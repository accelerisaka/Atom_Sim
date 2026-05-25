"""
行人过马路从众心理联合仿真 — 主入口
========================================
加载 config/topology_crosswalk.yaml，运行 Orchestrator，
并基于 orch.history 自定义动态可视化面板。

可视化面板 (2x2)：
  ┌────────────────────────┬────────────────────────┐
  │ 路口俯视图              │ 性格分布 & 过街状态     │
  │ (行人/车辆/信号灯/斑马线)│ (闯红灯倾向/过街人数)   │
  ├────────────────────────┼────────────────────────┤
  │ 信号相位 & 碰撞计数     │ 心理指标时序            │
  │ (红绿切换/累计碰撞)      │ (焦虑/从众/决策)        │
  └────────────────────────┴────────────────────────┘
导出 GIF / MP4 至 output_crosswalk/。
"""

from __future__ import annotations

import argparse
import json
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
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

from core.base import AtomicSimulator
from core.orchestrator import Orchestrator
from core.protocol import (
    EventMessage,
    ExchangeStrategy,
    PortAddress,
    S2SConnection,
)
from core.transforms import TRANSFORM_REGISTRY, get_transform

from atoms.physical.tl1_traffic_light import TrafficLightController
from atoms.physical.veh1_lane_traffic import LaneVehicleTraffic
from atoms.physical.ped1_crossing_body import PedestrianCrossingBody
from atoms.physical.cl1_collision import CollisionDetector
from atoms.physical.fl4_crowd_fluid import CrowdFluid
from atoms.physical.p6_bottleneck import BottleneckGate

from atoms.social.pp1_pedestrian_profile import PedestrianProfile
from atoms.social.pc1_crosswalk_decision import CrosswalkDecision
from atoms.social.s2_herd import HerdBehavior
from atoms.social.bp5_attention import AttentionAllocator
from atoms.social.pd3_emotion import EmotionAppraisal
from atoms.social.bp6_stress import StressPerformanceCurve


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_crosswalk")


_SIM_REGISTRY: Dict[str, type] = {
    "atoms.physical.TrafficLightController": TrafficLightController,
    "atoms.physical.LaneVehicleTraffic": LaneVehicleTraffic,
    "atoms.physical.PedestrianCrossingBody": PedestrianCrossingBody,
    "atoms.physical.CollisionDetector": CollisionDetector,
    "atoms.physical.CrowdFluid": CrowdFluid,
    "atoms.physical.BottleneckGate": BottleneckGate,
    "atoms.social.PedestrianProfile": PedestrianProfile,
    "atoms.social.CrosswalkDecision": CrosswalkDecision,
    "atoms.social.HerdBehavior": HerdBehavior,
    "atoms.social.AttentionAllocator": AttentionAllocator,
    "atoms.social.EmotionAppraisal": EmotionAppraisal,
    "atoms.social.StressPerformanceCurve": StressPerformanceCurve,
}


def _build_simulator(sim_id: str, cfg: Dict[str, Any]) -> AtomicSimulator:
    cls_name = cfg["class"]
    cls = _SIM_REGISTRY.get(cls_name)
    if cls is None:
        raise ValueError(
            f"Unknown simulator class: {cls_name}. "
            f"Available: {list(_SIM_REGISTRY.keys())}"
        )
    params = dict(cfg.get("params", {}))
    for key in ("grid_size", "world_size", "jaywalk_range", "crosswalk"):
        if key in params and isinstance(params[key], list):
            if key == "crosswalk":
                continue
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


def _apply_scenario(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    ped1: PedestrianCrossingBody = orch.simulators.get("PED1")  # type: ignore
    veh1: LaneVehicleTraffic = orch.simulators.get("VEH1")  # type: ignore

    walls = scenario.get("walls", [])
    if ped1 is not None and walls:
        ped1.set_walls([(tuple(w[0]), tuple(w[1])) for w in walls])

    exits = scenario.get("exits", [])
    if ped1 is not None and exits:
        ped1.set_exits([tuple(e) for e in exits])

    crosswalk = scenario.get("crosswalk", {})
    if ped1 is not None and crosswalk:
        x_range = crosswalk.get("x_range", [12.0, 18.0])
        north_y = crosswalk.get("north_target_y", 14.0)
        ped1.set_crosswalk((float(x_range[0]), float(x_range[1])), float(north_y))

    stop_line = scenario.get("stop_line_x")
    if veh1 is not None and stop_line is not None:
        veh1.stop_line_x = float(stop_line)


def _setup_collision_event_bridge(orch: Orchestrator) -> None:
    """CL1 检测到新碰撞 → 注入 COLLISION_EVENT → 标记 PED1 跌倒。"""

    cl1: CollisionDetector = orch.simulators.get("CL1")  # type: ignore
    if cl1 is None:
        return

    original_step = cl1.step

    def _patched_step(dt: float) -> None:
        original_step(dt)
        new_hits = cl1.state.get("new_collisions")
        if new_hits is not None and np.any(new_hits):
            event = EventMessage(
                event_type="COLLISION_EVENT",
                source_sim_id="CL1",
                timestamp=cl1.current_time,
                payload={
                    "new_collisions": new_hits.copy(),
                    "hit_mask": cl1.state["hit_mask"].copy(),
                    "collision_count": int(cl1.state["collision_count"]),
                },
            )
            orch.inject_event(event)
            indices = np.where(new_hits)[0].tolist()
            logger.info(
                f"  COLLISION_EVENT @ t={cl1.current_time:.3f}s  "
                f"pedestrians={indices}  total={cl1.state['collision_count']}"
            )

    cl1.step = _patched_step  # type: ignore

    def _handle_collision(event: EventMessage) -> None:
        ped1: PedestrianCrossingBody = orch.simulators.get("PED1")  # type: ignore
        if ped1 is None:
            return
        new_hits = event.payload.get("new_collisions")
        if new_hits is None:
            return
        hits = np.asarray(new_hits, dtype=bool)
        n = min(hits.shape[0], ped1.num_agents)
        fallen = ped1.state["fallen"]
        active = ped1.state["active"]
        for i in range(n):
            if hits[i]:
                fallen[i] = True
                active[i] = False
                ped1.state["velocities"][i] = 0.0

    orch.register_event_handler("COLLISION_EVENT", _handle_collision)


def _build_time_series(snapshots: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    times, ped_sig, veh_sig, collisions = [], [], [], []
    avg_panic, avg_cross, avg_herd, n_crossing = [], [], [], []

    for s in snapshots:
        times.append(float(s["time"]))
        tl1 = s.get("TL1", {}) or {}
        cl1 = s.get("CL1", {}) or {}
        pd3 = s.get("PD3", {}) or {}
        pc1 = s.get("PC1", {}) or {}
        s2 = s.get("S2", {}) or {}
        ped1 = s.get("PED1", {}) or {}

        ped_sig.append(float(tl1.get("pedestrian_signal", 0)))
        veh_sig.append(float(tl1.get("vehicle_signal", 0)))
        collisions.append(int(cl1.get("collision_count", 0)))

        panic = pd3.get("panic_level", [])
        cross = pc1.get("cross_intent", [])
        herd = s2.get("herd_velocity_bias", [])
        crossing = ped1.get("crossing", [])

        avg_panic.append(float(np.mean(panic)) if len(panic) else 0.0)
        avg_cross.append(float(np.mean(cross)) if len(cross) else 0.0)
        if isinstance(herd, list) and len(herd) > 0:
            hm = np.linalg.norm(np.asarray(herd), axis=1)
            avg_herd.append(float(np.mean(hm)))
        else:
            avg_herd.append(0.0)
        n_crossing.append(int(np.sum(np.asarray(crossing, dtype=bool))) if len(crossing) else 0)

    return {
        "time": np.asarray(times),
        "ped_signal": np.asarray(ped_sig),
        "veh_signal": np.asarray(veh_sig),
        "collisions": np.asarray(collisions),
        "avg_panic": np.asarray(avg_panic),
        "avg_cross_intent": np.asarray(avg_cross),
        "avg_herd": np.asarray(avg_herd),
        "n_crossing": np.asarray(n_crossing),
    }


def _render_summary_png(
    snapshots: List[Dict[str, Any]],
    series: Dict[str, np.ndarray],
    static: Dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("行人过马路从众心理仿真 — 汇总", fontsize=14, fontweight="bold")

    jaywalk = static["jaywalk_tendency"]
    axes[0, 0].hist(jaywalk, bins=15, color="#e74c3c", alpha=0.75, edgecolor="white")
    axes[0, 0].set_title("行人闯红灯倾向分布")
    axes[0, 0].set_xlabel("jaywalk_tendency")
    axes[0, 0].set_ylabel("人数")

    t = series["time"]
    axes[0, 1].plot(t, series["ped_signal"], "g-", label="行人信号", linewidth=2)
    axes[0, 1].plot(t, series["veh_signal"], "r-", label="车辆信号", linewidth=2)
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].set_title("交通信号时序")
    axes[0, 1].set_xlabel("时间 (s)")
    axes[0, 1].legend()

    axes[1, 0].plot(t, series["n_crossing"], color="#3498db", linewidth=2)
    axes[1, 0].set_title("正在过街人数")
    axes[1, 0].set_xlabel("时间 (s)")

    ax2 = axes[1, 0].twinx()
    ax2.plot(t, series["collisions"], "k--", linewidth=1.5, label="累计碰撞")
    ax2.set_ylabel("碰撞次数")
    ax2.legend(loc="upper right")

    axes[1, 1].plot(t, series["avg_panic"], label="平均焦虑", color="#9b59b6")
    axes[1, 1].plot(t, series["avg_herd"], label="平均从众偏置", color="#e67e22")
    axes[1, 1].plot(t, series["avg_cross_intent"], label="平均过街意图", color="#2ecc71")
    axes[1, 1].set_title("社会心理指标")
    axes[1, 1].set_xlabel("时间 (s)")
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Summary PNG: {out_path}")


def _render_animation(
    snapshots: List[Dict[str, Any]],
    series: Dict[str, np.ndarray],
    static: Dict[str, Any],
    gif_path: Path,
    mp4_path: Path,
    fps: int = 8,
) -> None:
    world_size = static["world_size"]
    crosswalk = static["crosswalk"]
    jaywalk = static["jaywalk_tendency"]
    walls = static.get("walls", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("行人过马路从众心理联合仿真", fontsize=13, fontweight="bold")

    ax_map = axes[0, 0]
    ax_profile = axes[0, 1]
    ax_signal = axes[1, 0]
    ax_psych = axes[1, 1]

    # --- 地图轴 ---
    ax_map.set_xlim(0, world_size[0])
    ax_map.set_ylim(0, world_size[1])
    ax_map.set_aspect("equal")
    ax_map.set_title("路口俯视图")
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")

    # 道路
    road = Rectangle((0, 8.5), world_size[0], 3.0, facecolor="#555555", alpha=0.35, zorder=0)
    ax_map.add_patch(road)
    # 斑马线
    x0, x1 = crosswalk["x_range"]
    cross_rect = Rectangle(
        (x0, crosswalk["y_range"][0]), x1 - x0,
        crosswalk["y_range"][1] - crosswalk["y_range"][0],
        facecolor="#ecf0f1", alpha=0.6, edgecolor="white", linewidth=1, zorder=1,
    )
    ax_map.add_patch(cross_rect)
    for xi in np.linspace(x0, x1, 8):
        ax_map.plot([xi, xi], [crosswalk["y_range"][0], crosswalk["y_range"][1]],
                    color="white", linewidth=0.8, alpha=0.5, zorder=2)

    if walls:
        segs = [[w[0], w[1]] for w in walls]
        ax_map.add_collection(LineCollection(segs, colors="#2c3e50", linewidths=2, zorder=3))

    ped_scatter = ax_map.scatter([], [], s=30, c=[], cmap="RdYlGn_r", vmin=0, vmax=1,
                                 edgecolors="black", linewidths=0.3, zorder=5, label="行人")
    veh_scatter = ax_map.scatter([], [], s=120, c="#e74c3c", marker="s",
                                 edgecolors="black", linewidths=0.5, zorder=6, label="车辆")
    fallen_scatter = ax_map.scatter([], [], s=60, c="black", marker="x", zorder=7, label="事故")

    light_text = ax_map.text(1.0, 18.5, "", fontsize=10, fontweight="bold",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # --- 性格轴 ---
    ax_profile.set_title("性格 vs 过街状态")
    ax_profile.set_xlabel("闯红灯倾向")
    ax_profile.set_ylabel("索引")
    ax_profile.scatter(jaywalk, np.arange(len(jaywalk)), c=jaywalk, cmap="RdYlGn_r",
                       vmin=0, vmax=1, s=40, edgecolors="gray", linewidths=0.3)
    status_text = ax_profile.text(0.02, 0.98, "", transform=ax_profile.transAxes,
                                  va="top", fontsize=9,
                                  bbox=dict(boxstyle="round", facecolor="#fef9e7", alpha=0.9))

    # --- 信号轴 ---
    ax_signal.set_xlim(0, series["time"][-1] if len(series["time"]) else 90)
    ax_signal.set_ylim(-0.1, 1.1)
    ax_signal.set_title("信号 & 碰撞")
    ax_signal.set_xlabel("时间 (s)")
    line_ped, = ax_signal.plot([], [], "g-", linewidth=2, label="行人信号")
    line_veh, = ax_signal.plot([], [], "r-", linewidth=2, label="车辆信号")
    line_col, = ax_signal.plot([], [], "k--", linewidth=1.5, label="累计碰撞")
    vline = ax_signal.axvline(0, color="blue", linestyle=":", alpha=0.6)
    ax_signal.legend(loc="upper right", fontsize=8)

    # --- 心理轴 ---
    ax_psych.set_xlim(0, series["time"][-1] if len(series["time"]) else 90)
    ax_psych.set_ylim(0, max(1.2, float(series["avg_herd"].max()) + 0.2))
    ax_psych.set_title("社会心理时序")
    ax_psych.set_xlabel("时间 (s)")
    ln_panic, = ax_psych.plot([], [], color="#9b59b6", label="焦虑")
    ln_herd, = ax_psych.plot([], [], color="#e67e22", label="从众")
    ln_cross, = ax_psych.plot([], [], color="#2ecc71", label="过街意图")
    ln_ncross, = ax_psych.plot([], [], color="#3498db", label="过街人数/40")
    vline2 = ax_psych.axvline(0, color="blue", linestyle=":", alpha=0.6)
    ax_psych.legend(loc="upper right", fontsize=8)

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=11)

    def _update(frame_idx: int):
        snap = snapshots[frame_idx]
        t = snap["time"]
        ped1 = snap.get("PED1", {}) or {}
        veh1 = snap.get("VEH1", {}) or {}
        tl1 = snap.get("TL1", {}) or {}
        pc1 = snap.get("PC1", {}) or {}

        ppos = np.asarray(ped1.get("positions", []))
        pfallen = np.asarray(ped1.get("fallen", []), dtype=bool)
        pactive = np.asarray(ped1.get("active", []), dtype=bool)
        vpos = np.asarray(veh1.get("positions", []))
        vactive = np.asarray(veh1.get("active", []), dtype=bool)

        if ppos.ndim == 2 and ppos.shape[0] > 0:
            n = min(ppos.shape[0], len(jaywalk))
            colors = jaywalk[:n]
            ped_scatter.set_offsets(ppos[:n])
            ped_scatter.set_array(colors)
            mask_fallen = pfallen[:n] if pfallen.shape[0] >= n else np.zeros(n, dtype=bool)
            if np.any(mask_fallen):
                fallen_scatter.set_offsets(ppos[:n][mask_fallen])
            else:
                fallen_scatter.set_offsets(np.empty((0, 2)))

        if vpos.ndim == 2 and vpos.shape[0] > 0:
            if vactive.shape[0] == vpos.shape[0]:
                vpos = vpos[vactive]
            veh_scatter.set_offsets(vpos)

        ped_sig = float(tl1.get("pedestrian_signal", 0))
        phase = tl1.get("phase", "?")
        remaining = float(tl1.get("time_remaining", 0))
        color = "#27ae60" if ped_sig > 0.5 else "#c0392b"
        light_text.set_text(f"行人: {'通行' if ped_sig > 0.5 else '禁止'}\n"
                            f"相位: {phase}\n剩余: {remaining:.1f}s")
        light_text.set_color(color)

        cross_intent = pc1.get("cross_intent", [])
        n_cross = int(np.sum(np.asarray(cross_intent, dtype=bool))) if len(cross_intent) else 0
        n_jay = int(np.sum(jaywalk > 0.6))
        status_text.set_text(
            f"t={t:.1f}s\n过街意图: {n_cross}/{len(jaywalk)}\n"
            f"高闯红灯倾向: {n_jay} 人\n累计碰撞: {int(series['collisions'][frame_idx])}"
        )

        tt = series["time"][: frame_idx + 1]
        line_ped.set_data(tt, series["ped_signal"][: frame_idx + 1])
        line_veh.set_data(tt, series["veh_signal"][: frame_idx + 1])
        col = series["collisions"][: frame_idx + 1].astype(float)
        line_col.set_data(tt, col / max(col.max(), 1))
        vline.set_xdata([t, t])

        ln_panic.set_data(tt, series["avg_panic"][: frame_idx + 1])
        ln_herd.set_data(tt, series["avg_herd"][: frame_idx + 1])
        ln_cross.set_data(tt, series["avg_cross_intent"][: frame_idx + 1])
        ln_ncross.set_data(tt, series["n_crossing"][: frame_idx + 1] / 40.0)
        vline2.set_xdata([t, t])

        time_text.set_text(f"仿真时间: {t:.1f} s")
        artists = [ped_scatter, veh_scatter, fallen_scatter, light_text, status_text,
                   line_ped, line_veh, line_col, vline, ln_panic, ln_herd, ln_cross,
                   ln_ncross, vline2, time_text]
        return artists

    anim = FuncAnimation(fig, _update, frames=len(snapshots), interval=1000 // fps, blit=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    try:
        anim.save(str(gif_path), writer=PillowWriter(fps=fps))
        logger.info(f"GIF exported: {gif_path}")
    except Exception as exc:
        logger.warning(f"GIF export failed: {exc}")

    try:
        anim.save(str(mp4_path), fps=fps)
        logger.info(f"MP4 exported: {mp4_path}")
    except Exception as exc:
        logger.warning(f"MP4 export failed (need ffmpeg): {exc}")

    plt.close(fig)


def main(config_path: str = "config/topology_crosswalk.yaml") -> None:
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
            logger.info(f"Feedback loop (expected): {loop}")

    scenario = config.get("scenario", {})
    _apply_scenario(orch, scenario)
    _setup_collision_event_bridge(orch)

    logger.info("\n" + orch.summary())

    duration = float(scenario.get("duration", 90.0))
    snap_interval = float(scenario.get("snapshot_interval", 0.5))

    t0 = time.perf_counter()
    orch.run(duration=duration, snapshot_interval=snap_interval)
    elapsed = time.perf_counter() - t0
    logger.info(f"Simulation finished in {elapsed:.2f}s, {len(orch.history)} snapshots.")

    # 统计
    cl1: CollisionDetector = orch.simulators.get("CL1")  # type: ignore
    pp1: PedestrianProfile = orch.simulators.get("PP1")  # type: ignore
    ped1: PedestrianCrossingBody = orch.simulators.get("PED1")  # type: ignore
    if cl1 and pp1 and ped1:
        jay = pp1.state["jaywalk_tendency"]
        logger.info(
            f"Results: collisions={cl1.state['collision_count']}, "
            f"fallen={int(np.sum(ped1.state['fallen']))}, "
            f"avg_jaywalk={float(np.mean(jay)):.3f}, "
            f"extreme_jaywalkers={int(np.sum(jay > 0.75))}"
        )

    output_dir = Path("output_crosswalk")
    output_dir.mkdir(exist_ok=True)

    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(orch.history, f, default=_json_default)
    logger.info(f"History JSON: {history_path}")

    pp1 = orch.simulators.get("PP1")
    static = {
        "world_size": tuple(scenario.get("world_size", [30.0, 20.0])),
        "crosswalk": scenario.get("crosswalk", {}),
        "walls": scenario.get("walls", []),
        "jaywalk_tendency": pp1.state["jaywalk_tendency"].copy() if pp1 else np.array([]),
    }

    if not orch.history:
        logger.warning("No snapshots collected.")
        return

    series = _build_time_series(orch.history)
    _render_summary_png(orch.history, series, static, output_dir / "crosswalk_summary.png")
    _render_animation(
        orch.history, series, static,
        gif_path=output_dir / "crosswalk_dynamic.gif",
        mp4_path=output_dir / "crosswalk_dynamic.mp4",
        fps=8,
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Crosswalk Co-Simulation")
    parser.add_argument(
        "-c", "--config",
        default="config/topology_crosswalk.yaml",
        help="Path to topology YAML config",
    )
    args = parser.parse_args()
    main(args.config)
