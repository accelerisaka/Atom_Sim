"""
体育场突发爆炸 → 极端踩踏 + 应急广播引导 — 联合仿真主入口
========================================================
加载 config/topology_stadium_crush.yaml，运行 Orchestrator，
并基于 orch.history 自定义 2×3 动态可视化面板。

可视化面板 (2x3)：
  ┌─────────────────────┬─────────────────────┬─────────────────────┐
  │ 爆炸危害场 + 行人    │ 人群密度 + 挤压应力  │ 广播服从度 (散点)    │
  ├─────────────────────┼─────────────────────┼─────────────────────┤
  │ 恐慌水平 (散点)      │ 疏散进度 (时序)      │ 挤压峰值 (时序)      │
  └─────────────────────┴─────────────────────┴─────────────────────┘
导出 PNG / GIF / MP4 至 output_stadium_crush/。
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

from atoms.physical.ex1_blast_wave import ExplosionBlastWave
from atoms.physical.cf1_crush_stress import CrushStressField
from atoms.physical.fl4_crowd_fluid import CrowdFluid
from atoms.physical.cm1_rigid_body import RigidBody2D
from atoms.physical.p6_bottleneck import BottleneckGate

from atoms.social.bc1_emergency_broadcast import EmergencyBroadcast
from atoms.social.eg1_broadcast_compliance import BroadcastCompliance
from atoms.social.pd3_emotion import EmotionAppraisal
from atoms.social.bp6_stress import StressPerformanceCurve
from atoms.social.s2_herd import HerdBehavior
from atoms.social.bp5_attention import AttentionAllocator
from atoms.social.sd4_bystander import BystanderEffect


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_stadium_crush")


_SIM_REGISTRY: Dict[str, type] = {
    "atoms.physical.ExplosionBlastWave": ExplosionBlastWave,
    "atoms.physical.CrushStressField": CrushStressField,
    "atoms.physical.CrowdFluid": CrowdFluid,
    "atoms.physical.RigidBody2D": RigidBody2D,
    "atoms.physical.BottleneckGate": BottleneckGate,
    "atoms.social.EmergencyBroadcast": EmergencyBroadcast,
    "atoms.social.BroadcastCompliance": BroadcastCompliance,
    "atoms.social.EmotionAppraisal": EmotionAppraisal,
    "atoms.social.StressPerformanceCurve": StressPerformanceCurve,
    "atoms.social.HerdBehavior": HerdBehavior,
    "atoms.social.AttentionAllocator": AttentionAllocator,
    "atoms.social.BystanderEffect": BystanderEffect,
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
    ex1: ExplosionBlastWave = orch.simulators.get("EX1")  # type: ignore
    bc1: EmergencyBroadcast = orch.simulators.get("BC1")  # type: ignore
    bp5: AttentionAllocator = orch.simulators.get("BP5")  # type: ignore

    blast_points = scenario.get("blast_points", [])
    if ex1 is not None and blast_points:
        ctx = TransformContext(
            simulators=orch.simulators,
            bus=orch.global_bus,
            source_sim_id="scenario",
            target_sim_id="EX1",
            target_time=0.0,
        )
        seed_fn = get_transform("blast_points_to_seed_field")
        seed_field = seed_fn(blast_points, ctx)
        if seed_field is not None:
            ex1.inputs["blast_seed_field"] = seed_field
        if blast_points:
            ex1.set_epicenter_grid(
                int(blast_points[0][0]),
                int(blast_points[0][1]),
                float(blast_points[0][2]) if len(blast_points[0]) > 2 else 1.0,
            )

    walls = scenario.get("walls", [])
    if cm1 is not None and walls:
        cm1.set_walls([(tuple(w[0]), tuple(w[1])) for w in walls])

    exits = scenario.get("exits", [])
    if cm1 is not None and exits:
        cm1.set_exits(exits)

    if bc1 is not None and exits:
        idx = int(scenario.get("announced_exit_index", 0))
        bc1.set_announced_exit(idx, exits)
        delay = float(scenario.get("broadcast_delay", 4.0))
        bc1.broadcast_delay = delay

    noise = float(scenario.get("ambient_noise_level", 0.0))
    if bp5 is not None:
        bp5.inputs["noise_level"] = noise

    # 初始人群聚集在爆炸点附近（东侧看台，模拟数千人涌向通道）
    spawn = scenario.get("initial_spawn_region")
    if cm1 is not None and spawn:
        x0, x1, y0, y1 = spawn
        rng = np.random.default_rng(7)
        n = cm1.num_agents
        cm1.state["positions"] = np.column_stack([
            rng.uniform(x0, x1, n),
            rng.uniform(y0, y1, n),
        ])


def _setup_fall_event_bridge(orch: Orchestrator) -> None:
    cm1: RigidBody2D = orch.simulators.get("CM1")  # type: ignore
    if cm1 is None:
        return

    prev_fallen = np.zeros(cm1.num_agents, dtype=bool)
    original_step = cm1.step

    def _patched_step(dt: float) -> None:
        nonlocal prev_fallen
        original_step(dt)
        current = cm1.state["fallen"]
        new_falls = current & ~prev_fallen
        if np.any(new_falls):
            event = EventMessage(
                event_type="FALL_EVENT",
                source_sim_id="CM1",
                timestamp=cm1.current_time,
                payload={
                    "fallen": current.copy(),
                    "active": cm1.state["active"].copy(),
                    "positions": cm1.state["positions"].copy(),
                    "new_fall_indices": np.where(new_falls)[0].tolist(),
                },
            )
            orch.inject_event(event)
            logger.info(
                f"  FALL_EVENT @ t={cm1.current_time:.3f}s  "
                f"new={np.where(new_falls)[0].tolist()}"
            )
        prev_fallen = current.copy()

    cm1.step = _patched_step  # type: ignore

    def _handle_fall(event: EventMessage) -> None:
        sd4: BystanderEffect = orch.simulators.get("SD4")  # type: ignore
        if sd4 is None:
            return
        sd4.inputs["fallen_mask"] = event.payload.get("fallen")
        sd4.inputs["active_mask"] = event.payload.get("active")
        sd4.inputs["agent_positions"] = event.payload.get("positions")
        sd4.step(0.0)
        sd4.record_output()

    orch.register_event_handler("FALL_EVENT", _handle_fall)


def _setup_crush_spike_monitor(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    cf1: CrushStressField = orch.simulators.get("CF1")  # type: ignore
    if cf1 is None:
        return

    threshold = float(scenario.get("crush_spike_threshold", 0.65))
    prev_peak = [0.0]
    crush_to_exposure = get_transform("crush_stress_field_to_per_agent_danger")
    original_step = cf1.step

    def _patched_step(dt: float) -> None:
        original_step(dt)
        peak = float(cf1.state["peak_stress"])
        delta = peak - prev_peak[0]
        if peak >= threshold and delta > 0.12:
            event = EventMessage(
                event_type="CRUSH_SPIKE",
                source_sim_id="CF1",
                timestamp=cf1.current_time,
                payload={
                    "crush_stress_field": cf1.state["crush_stress_field"].copy(),
                    "peak_stress": peak,
                },
            )
            orch.inject_event(event)
            logger.info(
                f"  CRUSH_SPIKE @ t={cf1.current_time:.2f}s  peak={peak:.3f}"
            )
        prev_peak[0] = peak

    cf1.step = _patched_step  # type: ignore

    def _handle_crush(event: EventMessage) -> None:
        pd3: EmotionAppraisal = orch.simulators.get("PD3")  # type: ignore
        if pd3 is None:
            return
        field = event.payload.get("crush_stress_field")
        ctx = TransformContext(
            simulators=orch.simulators,
            bus=orch.global_bus,
            source_sim_id="CF1",
            target_sim_id="PD3",
            target_time=event.timestamp,
        )
        pd3.inputs["smoke_exposure"] = crush_to_exposure(field, ctx)
        pd3.step(0.0)
        pd3.record_output()

    orch.register_event_handler("CRUSH_SPIKE", _handle_crush)


def _build_time_series(snapshots: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    times, escaped, inside, fallen = [], [], [], []
    avg_panic, avg_eff, avg_compliance = [], [], []
    peak_crush, peak_blast, broadcast_on = [], [], []

    for s in snapshots:
        times.append(s["time"])
        cm1 = s.get("CM1", {}) or {}
        pd3 = s.get("PD3", {}) or {}
        bp6 = s.get("BP6", {}) or {}
        eg1 = s.get("EG1", {}) or {}
        cf1 = s.get("CF1", {}) or {}
        ex1 = s.get("EX1", {}) or {}
        bc1 = s.get("BC1", {}) or {}

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
        comp = np.asarray(eg1.get("compliance_weight", [0.0]))
        avg_panic.append(float(np.mean(panic)) if panic.size else 0.0)
        avg_eff.append(float(np.mean(eff)) if eff.size else 1.0)
        avg_compliance.append(float(np.mean(comp)) if comp.size else 0.0)

        stress = cf1.get("crush_stress_field")
        blast = ex1.get("blast_hazard_field")
        peak_crush.append(float(np.max(stress)) if stress is not None else 0.0)
        peak_blast.append(float(np.max(blast)) if blast is not None else 0.0)
        broadcast_on.append(float(bc1.get("broadcast_active", 0.0)))

    return {
        "time": np.asarray(times, dtype=np.float64),
        "escaped": np.asarray(escaped),
        "inside": np.asarray(inside),
        "fallen": np.asarray(fallen),
        "avg_panic": np.asarray(avg_panic),
        "avg_efficiency": np.asarray(avg_eff),
        "avg_compliance": np.asarray(avg_compliance),
        "peak_crush": np.asarray(peak_crush),
        "peak_blast": np.asarray(peak_blast),
        "broadcast_on": np.asarray(broadcast_on),
    }


def _draw_walls(ax, walls: List, color: str = "#444444") -> None:
    for w in walls:
        ax.plot(
            [w[0][1], w[1][1]], [w[0][0], w[1][0]],
            color=color, linewidth=1.2, zorder=3,
        )


def _setup_axes(fig, static: Dict[str, Any], series: Dict[str, np.ndarray], walls: List):
    axes = fig.subplots(2, 3)
    ax_blast, ax_crush, ax_comp = axes[0]
    ax_panic, ax_evac, ax_ts = axes[1]

    H, W = static["grid_size"]
    cell = static["cell_size"]
    extent = (0, W * cell, H * cell, 0)

    ax_blast.set_title("爆炸危害场 + 行人", fontsize=10)
    ax_blast.set_xlabel("y (m)")
    ax_blast.set_ylabel("x (m)")
    im_blast = ax_blast.imshow(
        np.zeros((H, W)), origin="upper", cmap="hot",
        vmin=0.0, vmax=1.0, extent=extent, aspect="equal",
    )
    fig.colorbar(im_blast, ax=ax_blast, fraction=0.046, pad=0.04, label="危害")
    sc_blast = ax_blast.scatter([], [], s=10, c=[], cmap="viridis", vmin=0, vmax=1,
                                edgecolors="k", linewidths=0.2, zorder=5)
    _draw_walls(ax_blast, walls)
    for ex in static.get("exits", []):
        ax_blast.plot(ex[1], ex[0], "g^", markersize=9, zorder=6)
    ann_ex = static.get("announced_exit")
    if ann_ex is not None:
        ax_blast.plot(ann_ex[1], ann_ex[0], "c*", markersize=14, zorder=7, label="广播出口")
    title_blast = ax_blast.text(0.02, 1.04, "", transform=ax_blast.transAxes, fontsize=9)

    ax_crush.set_title("人群密度 + 挤压应力叠加", fontsize=10)
    ax_crush.set_xlabel("y (m)")
    ax_crush.set_ylabel("x (m)")
    im_dens = ax_crush.imshow(
        np.zeros((H, W)), origin="upper", cmap="Blues",
        vmin=0.0, vmax=8.0, extent=extent, aspect="equal", alpha=0.7,
    )
    im_stress = ax_crush.imshow(
        np.zeros((H, W)), origin="upper", cmap="Reds",
        vmin=0.0, vmax=1.0, extent=extent, aspect="equal", alpha=0.45,
    )
    _draw_walls(ax_crush, walls)
    title_crush = ax_crush.text(0.02, 1.04, "", transform=ax_crush.transAxes, fontsize=9)

    ax_comp.set_title("应急广播服从度", fontsize=10)
    ax_comp.set_xlabel("y (m)")
    ax_comp.set_ylabel("x (m)")
    ax_comp.set_xlim(0, W * cell)
    ax_comp.set_ylim(H * cell, 0)
    sc_comp = ax_comp.scatter([], [], s=16, c=[], cmap="GnBu", vmin=0, vmax=1,
                              edgecolors="gray", linewidths=0.3)
    _draw_walls(ax_comp, walls)
    title_comp = ax_comp.text(0.02, 1.04, "", transform=ax_comp.transAxes, fontsize=9)

    ax_panic.set_title("个体恐慌水平", fontsize=10)
    ax_panic.set_xlabel("y (m)")
    ax_panic.set_ylabel("x (m)")
    ax_panic.set_xlim(0, W * cell)
    ax_panic.set_ylim(H * cell, 0)
    sc_panic = ax_panic.scatter([], [], s=16, c=[], cmap="Reds", vmin=0, vmax=1,
                                edgecolors="gray", linewidths=0.3)
    _draw_walls(ax_panic, walls)
    title_panic = ax_panic.text(0.02, 1.04, "", transform=ax_panic.transAxes, fontsize=9)

    ax_evac.set_title("疏散进度")
    ax_evac.set_xlabel("time (s)")
    ax_evac.set_ylabel("人数")
    ax_evac.set_xlim(series["time"][0], series["time"][-1])
    n_agents = static["num_agents"]
    ax_evac.set_ylim(0, n_agents * 1.05)
    (line_esc,) = ax_evac.plot([], [], color="#2E7D32", linewidth=1.8, label="已疏散")
    (line_in,) = ax_evac.plot([], [], color="#1565C0", linewidth=1.6, label="场内")
    (line_fall,) = ax_evac.plot([], [], color="#C62828", linewidth=1.2, linestyle="--",
                                label="倒地")
    cursor_e = ax_evac.axvline(series["time"][0], color="k", linewidth=0.5, alpha=0.5)
    ax_evac.legend(loc="upper right", fontsize=8)

    ax_ts.set_title("挤压应力峰值 / 广播状态")
    ax_ts.set_xlabel("time (s)")
    ax_ts.set_ylabel("peak crush stress", color="#C62828")
    ax_ts.set_xlim(series["time"][0], series["time"][-1])
    ax_ts.set_ylim(0, max(series["peak_crush"].max() * 1.15, 0.5))
    (line_crush,) = ax_ts.plot([], [], color="#C62828", linewidth=1.6, label="peak crush")
    ax_ts2 = ax_ts.twinx()
    ax_ts2.set_ylabel("broadcast active", color="#1565C0")
    ax_ts2.set_ylim(-0.05, 1.15)
    (line_bc,) = ax_ts2.plot([], [], color="#1565C0", linewidth=1.2, linestyle="--",
                              label="broadcast")
    cursor_t = ax_ts.axvline(series["time"][0], color="k", linewidth=0.5, alpha=0.5)
    ax_ts.legend(loc="upper left", fontsize=8)

    return {
        "im_blast": im_blast, "sc_blast": sc_blast, "title_blast": title_blast,
        "im_dens": im_dens, "im_stress": im_stress, "title_crush": title_crush,
        "sc_comp": sc_comp, "title_comp": title_comp,
        "sc_panic": sc_panic, "title_panic": title_panic,
        "line_esc": line_esc, "line_in": line_in, "line_fall": line_fall,
        "line_crush": line_crush, "line_bc": line_bc,
        "cursor_e": cursor_e, "cursor_t": cursor_t,
    }


def _update_frame(idx, snapshots, series, static, walls, artists):
    s = snapshots[idx]
    t = s["time"]
    ex1 = s.get("EX1", {}) or {}
    cf1 = s.get("CF1", {}) or {}
    fl4 = s.get("FL4", {}) or {}
    cm1 = s.get("CM1", {}) or {}
    pd3 = s.get("PD3", {}) or {}
    eg1 = s.get("EG1", {}) or {}

    blast = ex1.get("blast_hazard_field")
    stress = cf1.get("crush_stress_field")
    dens = fl4.get("density_map")
    pos = cm1.get("positions")
    active = np.asarray(cm1.get("active", []), dtype=bool)
    panic = np.asarray(pd3.get("panic_level", []))
    comp = np.asarray(eg1.get("compliance_weight", []))

    if blast is not None:
        artists["im_blast"].set_data(np.asarray(blast))
    if dens is not None:
        artists["im_dens"].set_data(np.asarray(dens))
    if stress is not None:
        artists["im_stress"].set_data(np.asarray(stress))

    if pos is not None:
        pos_arr = np.asarray(pos)
        mask = active if active.size == pos_arr.shape[0] else np.ones(len(pos_arr), dtype=bool)
        vis = pos_arr[mask]
        if vis.size:
            xy = np.column_stack([vis[:, 1], vis[:, 0]])
            artists["sc_blast"].set_offsets(xy)
            if comp.size >= len(pos_arr):
                artists["sc_blast"].set_array(comp[mask])
            artists["sc_comp"].set_offsets(xy)
            if comp.size >= len(pos_arr):
                artists["sc_comp"].set_array(comp[mask])
            artists["sc_panic"].set_offsets(xy)
            if panic.size >= len(pos_arr):
                artists["sc_panic"].set_array(panic[mask])

    artists["title_blast"].set_text(
        f"t={t:5.1f}s  场内={int(series['inside'][idx])}  "
        f"max爆={series['peak_blast'][idx]:.2f}  广播={series['broadcast_on'][idx]:.0f}"
    )
    artists["title_crush"].set_text(
        f"peak挤压={series['peak_crush'][idx]:.3f}  倒地={int(series['fallen'][idx])}"
    )
    artists["title_comp"].set_text(
        f"avg服从={series['avg_compliance'][idx]:.3f}"
    )
    artists["title_panic"].set_text(
        f"avg恐慌={series['avg_panic'][idx]:.3f}  avg认知={series['avg_efficiency'][idx]:.2f}"
    )

    times = series["time"][: idx + 1]
    artists["line_esc"].set_data(times, series["escaped"][: idx + 1])
    artists["line_in"].set_data(times, series["inside"][: idx + 1])
    artists["line_fall"].set_data(times, series["fallen"][: idx + 1])
    artists["line_crush"].set_data(times, series["peak_crush"][: idx + 1])
    artists["line_bc"].set_data(times, series["broadcast_on"][: idx + 1])
    artists["cursor_e"].set_xdata([t, t])
    artists["cursor_t"].set_xdata([t, t])

    return list(artists.values())


def _render_outputs(snapshots, series, static, walls, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "体育场突发爆炸 → 极端踩踏 + 应急广播引导 — 联合仿真",
        fontsize=14, fontweight="bold",
    )
    artists = _setup_axes(fig, static, series, walls)
    _update_frame(len(snapshots) - 1, snapshots, series, static, walls, artists)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = output_dir / "stadium_crush_summary.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    logger.info(f"Summary PNG: {png_path}")

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "体育场突发爆炸 → 极端踩踏 + 应急广播引导 — 动态过程",
        fontsize=14, fontweight="bold",
    )
    artists = _setup_axes(fig, static, series, walls)
    n_frames = len(snapshots)
    fps = 8

    def _frame(i):
        return _update_frame(i, snapshots, series, static, walls, artists)

    anim = FuncAnimation(fig, _frame, frames=n_frames, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    gif_path = output_dir / "stadium_crush_dynamic.gif"
    try:
        anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=90)
        logger.info(f"GIF: {gif_path}")
    except Exception as e:
        logger.warning(f"GIF export failed: {e}")

    mp4_path = output_dir / "stadium_crush_dynamic.mp4"
    try:
        from matplotlib.animation import FFMpegWriter
        anim.save(str(mp4_path), writer=FFMpegWriter(fps=fps, codec="libx264"), dpi=100)
        logger.info(f"MP4: {mp4_path}")
    except Exception as e:
        logger.warning(f"MP4 export failed: {e}")

    plt.close(fig)


def main(config_path: str = "config/topology_stadium_crush.yaml") -> None:
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
    scenario.setdefault(
        "initial_spawn_region",
        [18.0, 24.0, 18.0, 24.0],
    )
    _apply_scenario(orch, scenario)
    _setup_fall_event_bridge(orch)
    _setup_crush_spike_monitor(orch, scenario)

    logger.info("\n" + orch.summary())

    duration = float(scenario.get("duration", 75.0))
    snap_interval = float(scenario.get("snapshot_interval", 1.0))

    t0 = time.perf_counter()
    orch.run(duration=duration, snapshot_interval=snap_interval)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Simulation finished in {elapsed:.2f}s, "
        f"{len(orch.history)} snapshots."
    )

    cm1: RigidBody2D = orch.simulators.get("CM1")  # type: ignore
    bc1: EmergencyBroadcast = orch.simulators.get("BC1")  # type: ignore
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
    exits = scenario.get("exits", [])
    ann_idx = int(scenario.get("announced_exit_index", 0))
    ann_exit = exits[ann_idx] if exits and ann_idx < len(exits) else None

    static = {
        "grid_size": (50, 50),
        "cell_size": 0.5,
        "num_agents": cm1.num_agents if cm1 else 350,
        "exits": exits,
        "announced_exit": ann_exit,
    }
    _render_outputs(
        orch.history, series, static,
        scenario.get("walls", []),
        Path("output_stadium_crush"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stadium Explosion Crush + Emergency Broadcast Evacuation",
    )
    parser.add_argument(
        "-c", "--config",
        default="config/topology_stadium_crush.yaml",
        help="Path to topology YAML config",
    )
    args = parser.parse_args()
    main(args.config)
