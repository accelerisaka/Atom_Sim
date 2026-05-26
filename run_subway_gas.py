"""
地铁站有毒重气体泄漏 + 智能排风疏散 — 联合仿真主入口
========================================================
加载 config/topology_subway_gas.yaml，运行 Orchestrator，
并基于 orch.history 自定义 2×3 动态可视化面板。

可视化面板 (2x3)：
  ┌─────────────────────┬─────────────────────┬─────────────────────┐
  │ 气体浓度场 + 行人    │ 智能排风激活区       │ 毒性危害场           │
  ├─────────────────────┼─────────────────────┼─────────────────────┤
  │ 恐慌水平 (散点)      │ 疏散进度 (时序)      │ 排风强度 (时序)      │
  └─────────────────────┴─────────────────────┴─────────────────────┘
导出 PNG / GIF / MP4 至 output_subway_gas/。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from core.base import AtomicSimulator
from core.orchestrator import Orchestrator
from core.protocol import (
    EventMessage,
    ExchangeStrategy,
    PortAddress,
    S2SConnection,
)
from core.transforms import TRANSFORM_REGISTRY, TransformContext, get_transform

from atoms.physical.fl5_heavy_gas import HeavyGasDiffusion2D
from atoms.physical.vt1_smart_vent import IntelligentVentilation
from atoms.physical.fl4_crowd_fluid import CrowdFluid
from atoms.physical.cm1_rigid_body import RigidBody2D
from atoms.physical.p6_bottleneck import BottleneckGate

from atoms.social.pd3_emotion import EmotionAppraisal
from atoms.social.bp6_stress import StressPerformanceCurve
from atoms.social.s2_herd import HerdBehavior
from atoms.social.bp5_attention import AttentionAllocator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_subway_gas")


_SIM_REGISTRY: Dict[str, type] = {
    "atoms.physical.HeavyGasDiffusion2D": HeavyGasDiffusion2D,
    "atoms.physical.IntelligentVentilation": IntelligentVentilation,
    "atoms.physical.CrowdFluid": CrowdFluid,
    "atoms.physical.RigidBody2D": RigidBody2D,
    "atoms.physical.BottleneckGate": BottleneckGate,
    "atoms.social.EmotionAppraisal": EmotionAppraisal,
    "atoms.social.StressPerformanceCurve": StressPerformanceCurve,
    "atoms.social.HerdBehavior": HerdBehavior,
    "atoms.social.AttentionAllocator": AttentionAllocator,
}


def _build_simulator(sim_id: str, cfg: Dict[str, Any]) -> AtomicSimulator:
    cls_name = cfg["class"]
    cls = _SIM_REGISTRY.get(cls_name)
    if cls is None:
        raise ValueError(
            f"Unknown simulator class: {cls_name}. "
            f"Available: {list(_SIM_REGISTRY.keys())}"
        )
    params = cfg.get("params", {})
    for key in ("grid_size", "world_size"):
        if key in params and isinstance(params[key], list):
            params[key] = tuple(params[key])
    return cls(sim_id=sim_id, **params)


def _build_connection(cfg: Dict[str, Any]) -> S2SConnection:
    conn_id = cfg["connection_id"]
    transform_name = cfg.get("transform")
    if transform_name is None:
        raise ValueError(
            f"Connection '{conn_id}' missing required field 'transform'. "
            f"Available: {sorted(TRANSFORM_REGISTRY.keys())}"
        )
    transform_fn = get_transform(transform_name)
    return S2SConnection(
        connection_id=conn_id,
        source=PortAddress(cfg["source"]["sim_id"], cfg["source"]["port"]),
        target=PortAddress(cfg["target"]["sim_id"], cfg["target"]["port"]),
        strategy=ExchangeStrategy(cfg["strategy"]),
        transform=transform_fn,
        transform_name=transform_name,
        description=cfg.get("description", ""),
    )


def _rasterize_wall(
    mask: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cell_size: float,
    grid_size: Tuple[int, int],
) -> None:
    steps = int(np.linalg.norm(b - a) / cell_size * 2) + 1
    for i in range(steps + 1):
        t = i / max(steps, 1)
        p = a + t * (b - a)
        r, c = int(p[0] / cell_size), int(p[1] / cell_size)
        if 0 <= r < grid_size[0] and 0 <= c < grid_size[1]:
            mask[r, c] = False


def _apply_scenario(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    cm1: RigidBody2D = orch.simulators.get("CM1")  # type: ignore
    fl5: HeavyGasDiffusion2D = orch.simulators.get("FL5")  # type: ignore
    bp5: AttentionAllocator = orch.simulators.get("BP5")  # type: ignore

    leak_points = scenario.get("leak_points", [])
    if fl5 is not None and leak_points:
        ctx = TransformContext(
            simulators=orch.simulators,
            bus=orch.global_bus,
            source_sim_id="scenario",
            target_sim_id="FL5",
            target_time=0.0,
        )
        rate_fn = get_transform("leak_points_to_gas_source_rate")
        rate_field = rate_fn(leak_points, ctx)
        if rate_field is not None:
            fl5.inputs["gas_source_rate"] = rate_field

    walls = scenario.get("walls", [])
    if cm1 is not None and walls:
        cm1.set_walls([(tuple(w[0]), tuple(w[1])) for w in walls])

    if fl5 is not None and walls:
        mask = np.ones(fl5.grid_size, dtype=bool)
        for w in walls:
            a, b = np.array(w[0]), np.array(w[1])
            _rasterize_wall(mask, a, b, fl5.cell_size, fl5.grid_size)
        fl5.set_obstacles(mask)

    exits = scenario.get("exits", [])
    if cm1 is not None and exits:
        cm1.set_exits(exits)

    noise = float(scenario.get("ambient_noise_level", 0.0))
    if bp5 is not None:
        bp5.inputs["noise_level"] = noise


def _setup_gas_spike_monitor(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    """监测 FL5 全局毒性危害突变，异步唤醒 PD3。"""
    fl5: HeavyGasDiffusion2D = orch.simulators.get("FL5")  # type: ignore
    if fl5 is None:
        return

    threshold = float(scenario.get("gas_spike_threshold", 0.55))
    prev_max = [0.0]
    hazard_to_danger = get_transform("gas_hazard_field_to_per_agent_danger")

    original_step = fl5.step

    def _patched_step(dt: float) -> None:
        original_step(dt)
        hazard = fl5.state["gas_hazard_field"]
        current_max = float(np.max(hazard))
        delta = current_max - prev_max[0]
        if current_max >= threshold and delta > 0.08:
            event = EventMessage(
                event_type="GAS_SPIKE",
                source_sim_id="FL5",
                timestamp=fl5.current_time,
                payload={
                    "gas_hazard_field": hazard.copy(),
                    "max_hazard": current_max,
                    "delta": delta,
                },
            )
            orch.inject_event(event)
            logger.info(
                f"  GAS_SPIKE @ t={fl5.current_time:.2f}s  "
                f"max_hazard={current_max:.3f}  delta={delta:.3f}"
            )
        prev_max[0] = current_max

    fl5.step = _patched_step  # type: ignore

    def _handle_spike(event: EventMessage) -> None:
        pd3: EmotionAppraisal = orch.simulators.get("PD3")  # type: ignore
        if pd3 is None:
            return
        raw_field = event.payload.get("gas_hazard_field")
        ctx = TransformContext(
            simulators=orch.simulators,
            bus=orch.global_bus,
            source_sim_id="FL5",
            target_sim_id="PD3",
            target_time=event.timestamp,
        )
        pd3.inputs["local_danger"] = hazard_to_danger(raw_field, ctx)
        pd3.step(0.0)
        pd3.record_output()

    orch.register_event_handler("GAS_SPIKE", _handle_spike)


def _build_time_series(snapshots: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    times, escaped, inside, fallen = [], [], [], []
    avg_panic, avg_eff, active_fans, total_vent = [], [], [], []
    max_conc, max_hazard = [], []

    for s in snapshots:
        times.append(s["time"])
        cm1 = s.get("CM1", {}) or {}
        pd3 = s.get("PD3", {}) or {}
        bp6 = s.get("BP6", {}) or {}
        vt1 = s.get("VT1", {}) or {}
        fl5 = s.get("FL5", {}) or {}

        active = np.asarray(cm1.get("active", []), dtype=bool)
        fall = np.asarray(cm1.get("fallen", []), dtype=bool)
        n = len(active) if active.size else 0
        if n:
            escaped.append(int(np.sum(~active & ~fall)))
            inside.append(int(np.sum(active)))
            fallen.append(int(np.sum(fall)))
        else:
            escaped.append(0)
            inside.append(0)
            fallen.append(0)

        panic = np.asarray(pd3.get("panic_level", [0.0]))
        eff = np.asarray(bp6.get("cognitive_efficiency_multiplier", [1.0]))
        avg_panic.append(float(np.mean(panic)) if panic.size else 0.0)
        avg_eff.append(float(np.mean(eff)) if eff.size else 1.0)

        active_fans.append(int(vt1.get("active_fan_count", 0)))
        total_vent.append(float(vt1.get("total_extraction_rate", 0.0)))

        conc = fl5.get("gas_concentration_field")
        hazard = fl5.get("gas_hazard_field")
        max_conc.append(float(np.max(conc)) if conc is not None else 0.0)
        max_hazard.append(float(np.max(hazard)) if hazard is not None else 0.0)

    return {
        "time": np.asarray(times, dtype=np.float64),
        "escaped": np.asarray(escaped),
        "inside": np.asarray(inside),
        "fallen": np.asarray(fallen),
        "avg_panic": np.asarray(avg_panic),
        "avg_efficiency": np.asarray(avg_eff),
        "active_fans": np.asarray(active_fans),
        "total_vent": np.asarray(total_vent),
        "max_conc": np.asarray(max_conc),
        "max_hazard": np.asarray(max_hazard),
    }


def _draw_walls(ax, walls: List, color: str = "#444444") -> None:
    for w in walls:
        ax.plot(
            [w[0][1], w[1][1]], [w[0][0], w[1][0]],
            color=color, linewidth=1.2, zorder=3,
        )


def _setup_axes(fig, static: Dict[str, Any], series: Dict[str, np.ndarray], walls: List):
    axes = fig.subplots(2, 3)
    ax_gas, ax_vent, ax_hazard = axes[0]
    ax_panic, ax_evac, ax_vent_ts = axes[1]

    H, W = static["grid_size"]
    cell = static["cell_size"]
    extent = (0, W * cell, H * cell, 0)

    # —— 1. 气体浓度 + 行人 ——
    ax_gas.set_title("气体浓度场 + 行人分布", fontsize=10)
    ax_gas.set_xlabel("y (m)")
    ax_gas.set_ylabel("x (m)")
    im_gas = ax_gas.imshow(
        np.zeros((H, W)), origin="upper", cmap="YlOrRd",
        vmin=0.0, vmax=0.8, extent=extent, aspect="equal",
    )
    fig.colorbar(im_gas, ax=ax_gas, fraction=0.046, pad=0.04, label="浓度")
    sc = ax_gas.scatter([], [], s=12, c=[], cmap="coolwarm", vmin=0, vmax=1,
                        edgecolors="black", linewidths=0.3, zorder=5)
    _draw_walls(ax_gas, walls)
    for ex in static.get("exits", []):
        ax_gas.plot(ex[1], ex[0], "g^", markersize=10, zorder=6)
    title_gas = ax_gas.text(
        0.02, 1.04, "", transform=ax_gas.transAxes, fontsize=9, fontweight="bold",
    )

    # —— 2. 排风场 ——
    ax_vent.set_title("智能排风激活区", fontsize=10)
    ax_vent.set_xlabel("y (m)")
    ax_vent.set_ylabel("x (m)")
    im_vent = ax_vent.imshow(
        np.zeros((H, W)), origin="upper", cmap="Blues",
        vmin=0.0, vmax=0.5, extent=extent, aspect="equal",
    )
    fig.colorbar(im_vent, ax=ax_vent, fraction=0.046, pad=0.04, label="抽取率 (1/s)")
    _draw_walls(ax_vent, walls)
    title_vent = ax_vent.text(
        0.02, 1.04, "", transform=ax_vent.transAxes, fontsize=9, fontweight="bold",
    )

    # —— 3. 毒性危害 ——
    ax_hazard.set_title("毒性危害场", fontsize=10)
    ax_hazard.set_xlabel("y (m)")
    ax_hazard.set_ylabel("x (m)")
    im_haz = ax_hazard.imshow(
        np.zeros((H, W)), origin="upper", cmap="magma",
        vmin=0.0, vmax=1.0, extent=extent, aspect="equal",
    )
    fig.colorbar(im_haz, ax=ax_hazard, fraction=0.046, pad=0.04, label="危害")
    _draw_walls(ax_hazard, walls)
    title_haz = ax_hazard.text(
        0.02, 1.04, "", transform=ax_hazard.transAxes, fontsize=9, fontweight="bold",
    )

    # —— 4. 恐慌散点 ——
    ax_panic.set_title("个体恐慌水平 (颜色=panic)", fontsize=10)
    ax_panic.set_xlabel("y (m)")
    ax_panic.set_ylabel("x (m)")
    ax_panic.set_xlim(0, W * cell)
    ax_panic.set_ylim(H * cell, 0)
    sc_panic = ax_panic.scatter([], [], s=18, c=[], cmap="Reds", vmin=0, vmax=1,
                                edgecolors="gray", linewidths=0.3)
    _draw_walls(ax_panic, walls)
    title_panic = ax_panic.text(
        0.02, 1.04, "", transform=ax_panic.transAxes, fontsize=9, fontweight="bold",
    )

    # —— 5. 疏散时序 ——
    ax_evac.set_title("疏散进度")
    ax_evac.set_xlabel("time (s)")
    ax_evac.set_ylabel("人数")
    ax_evac.set_xlim(series["time"][0], series["time"][-1])
    n_agents = static["num_agents"]
    ax_evac.set_ylim(0, n_agents * 1.05)
    (line_esc,) = ax_evac.plot([], [], color="#2E7D32", linewidth=1.8, label="已疏散")
    (line_in,) = ax_evac.plot([], [], color="#1565C0", linewidth=1.6, label="站内")
    (line_fall,) = ax_evac.plot([], [], color="#C62828", linewidth=1.2, linestyle="--",
                                label="倒地")
    cursor_e = ax_evac.axvline(series["time"][0], color="black", linewidth=0.5, alpha=0.5)
    ax_evac.legend(loc="upper right", fontsize=8)

    # —— 6. 排风时序 ——
    ax_vent_ts.set_title("排风系统响应")
    ax_vent_ts.set_xlabel("time (s)")
    ax_vent_ts.set_ylabel("激活分区数", color="#1565C0")
    ax_vent_ts.set_xlim(series["time"][0], series["time"][-1])
    ax_vent_ts.set_ylim(0, max(series["active_fans"].max() * 1.2, 5))
    (line_fans,) = ax_vent_ts.plot([], [], color="#1565C0", linewidth=1.6, label="active zones")
    ax_vent_ts.tick_params(axis="y", labelcolor="#1565C0")
    cursor_v = ax_vent_ts.axvline(series["time"][0], color="black", linewidth=0.5, alpha=0.5)

    ax_vent_ts2 = ax_vent_ts.twinx()
    ax_vent_ts2.set_ylabel("总抽取率", color="#E65100")
    v_max = max(series["total_vent"].max() * 1.15, 1.0)
    ax_vent_ts2.set_ylim(0, v_max)
    (line_vrate,) = ax_vent_ts2.plot([], [], color="#E65100", linewidth=1.4,
                                     linestyle="-.", label="total extraction")
    ax_vent_ts2.tick_params(axis="y", labelcolor="#E65100")
    ax_vent_ts.legend(handles=[
        Patch(facecolor="#1565C0", label="active_fan_count"),
        Patch(facecolor="#E65100", label="total_extraction_rate"),
    ], loc="upper left", fontsize=8)

    return {
        "im_gas": im_gas, "im_vent": im_vent, "im_haz": im_haz,
        "sc": sc, "sc_panic": sc_panic,
        "title_gas": title_gas, "title_vent": title_vent,
        "title_haz": title_haz, "title_panic": title_panic,
        "line_esc": line_esc, "line_in": line_in, "line_fall": line_fall,
        "line_fans": line_fans, "line_vrate": line_vrate,
        "cursor_e": cursor_e, "cursor_v": cursor_v,
    }


def _update_frame(idx, snapshots, series, static, walls, artists):
    s = snapshots[idx]
    t = s["time"]
    fl5 = s.get("FL5", {}) or {}
    vt1 = s.get("VT1", {}) or {}
    cm1 = s.get("CM1", {}) or {}
    pd3 = s.get("PD3", {}) or {}
    bp6 = s.get("BP6", {}) or {}

    conc = fl5.get("gas_concentration_field")
    hazard = fl5.get("gas_hazard_field")
    vent = vt1.get("ventilation_field")
    pos = cm1.get("positions")
    active = np.asarray(cm1.get("active", []), dtype=bool)
    panic = np.asarray(pd3.get("panic_level", []))
    eff = np.asarray(bp6.get("cognitive_efficiency_multiplier", []))

    if conc is not None:
        artists["im_gas"].set_data(np.asarray(conc))
    if vent is not None:
        artists["im_vent"].set_data(np.asarray(vent))
    if hazard is not None:
        artists["im_haz"].set_data(np.asarray(hazard))

    if pos is not None:
        pos_arr = np.asarray(pos)
        mask = active if active.size == pos_arr.shape[0] else np.ones(len(pos_arr), dtype=bool)
        vis_pos = pos_arr[mask]
        if vis_pos.size:
            artists["sc"].set_offsets(np.column_stack([vis_pos[:, 1], vis_pos[:, 0]]))
            if eff.size >= len(pos_arr):
                artists["sc"].set_array(eff[mask])
            artists["sc_panic"].set_offsets(np.column_stack([vis_pos[:, 1], vis_pos[:, 0]]))
            if panic.size >= len(pos_arr):
                artists["sc_panic"].set_array(panic[mask])

    n_esc = int(series["escaped"][idx])
    n_in = int(series["inside"][idx])
    n_fall = int(series["fallen"][idx])
    artists["title_gas"].set_text(
        f"t={t:5.1f}s  站内={n_in}  已疏散={n_esc}  maxC={series['max_conc'][idx]:.3f}"
    )
    artists["title_vent"].set_text(
        f"激活分区={int(series['active_fans'][idx])}  "
        f"总抽取={series['total_vent'][idx]:.1f}"
    )
    artists["title_haz"].set_text(
        f"max危害={series['max_hazard'][idx]:.3f}  "
        f"avg认知={series['avg_efficiency'][idx]:.2f}"
    )
    artists["title_panic"].set_text(
        f"avg恐慌={series['avg_panic'][idx]:.3f}"
    )

    times = series["time"][: idx + 1]
    artists["line_esc"].set_data(times, series["escaped"][: idx + 1])
    artists["line_in"].set_data(times, series["inside"][: idx + 1])
    artists["line_fall"].set_data(times, series["fallen"][: idx + 1])
    artists["line_fans"].set_data(times, series["active_fans"][: idx + 1])
    artists["line_vrate"].set_data(times, series["total_vent"][: idx + 1])
    artists["cursor_e"].set_xdata([t, t])
    artists["cursor_v"].set_xdata([t, t])

    return list(artists.values())


def _render_outputs(snapshots, series, static, walls, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "地铁站有毒重气体泄漏 + 智能排风疏散 — 联合仿真",
        fontsize=14, fontweight="bold",
    )
    artists = _setup_axes(fig, static, series, walls)
    _update_frame(len(snapshots) - 1, snapshots, series, static, walls, artists)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = output_dir / "subway_gas_summary.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    logger.info(f"Summary PNG: {png_path}")

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "地铁站有毒重气体泄漏 + 智能排风疏散 — 动态过程",
        fontsize=14, fontweight="bold",
    )
    artists = _setup_axes(fig, static, series, walls)
    n_frames = len(snapshots)
    fps = 8

    def _frame(i):
        return _update_frame(i, snapshots, series, static, walls, artists)

    anim = FuncAnimation(fig, _frame, frames=n_frames, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    gif_path = output_dir / "subway_gas_dynamic.gif"
    try:
        anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=90)
        logger.info(f"GIF: {gif_path}")
    except Exception as e:
        logger.warning(f"GIF export failed: {e}")

    mp4_path = output_dir / "subway_gas_dynamic.mp4"
    try:
        from matplotlib.animation import FFMpegWriter
        anim.save(str(mp4_path), writer=FFMpegWriter(fps=fps, codec="libx264"), dpi=100)
        logger.info(f"MP4: {mp4_path}")
    except Exception as e:
        logger.warning(f"MP4 export failed: {e}")

    plt.close(fig)


def main(config_path: str = "config/topology_subway_gas.yaml") -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    orch = Orchestrator()
    for group_name, group_cfg in config["time_groups"].items():
        orch.register_time_group(group_name, group_cfg["dt"])

    for sim_id, sim_cfg in config["simulators"].items():
        sim = _build_simulator(sim_id, sim_cfg)
        orch.register_simulator(sim, sim_cfg["group"])
        logger.info(f"  Registered {sim_id} -> group={sim_cfg['group']}")

    for conn_cfg in config["connections"]:
        conn = _build_connection(conn_cfg)
        orch.add_connection(conn)
    logger.info(f"S2S connections: {len(orch.connections)}")

    loops = orch.causality_guard.detect_algebraic_loops()
    if loops:
        for loop in loops:
            logger.info(f"Detected (intentional) feedback loop: {loop}")

    scenario = config.get("scenario", {})
    _apply_scenario(orch, scenario)
    _setup_gas_spike_monitor(orch, scenario)

    logger.info("\n" + orch.summary())

    duration = float(scenario.get("duration", 90.0))
    snap_interval = float(scenario.get("snapshot_interval", 1.0))

    t0 = time.perf_counter()
    orch.run(duration=duration, snapshot_interval=snap_interval)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Simulation finished in {elapsed:.2f}s, "
        f"{len(orch.history)} snapshots."
    )

    cm1: RigidBody2D = orch.simulators.get("CM1")  # type: ignore
    fl5: HeavyGasDiffusion2D = orch.simulators.get("FL5")  # type: ignore
    if cm1 is not None:
        active = cm1.state["active"]
        fallen = cm1.state["fallen"]
        logger.info(
            f"Results: escaped={int(np.sum(~active & ~fallen))}, "
            f"inside={int(np.sum(active))}, fallen={int(np.sum(fallen))}"
        )

    if not orch.history:
        logger.warning("No snapshots collected.")
        return

    series = _build_time_series(orch.history)
    static = {
        "grid_size": fl5.grid_size if fl5 else (50, 50),
        "cell_size": fl5.cell_size if fl5 else 0.5,
        "num_agents": cm1.num_agents if cm1 else 180,
        "exits": scenario.get("exits", []),
    }
    walls = scenario.get("walls", [])
    _render_outputs(orch.history, series, static, walls, Path("output_subway_gas"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subway Heavy Gas Leak + Smart Ventilation Evacuation Co-Simulation",
    )
    parser.add_argument(
        "-c", "--config",
        default="config/topology_subway_gas.yaml",
        help="Path to topology YAML config",
    )
    args = parser.parse_args()
    main(args.config)
