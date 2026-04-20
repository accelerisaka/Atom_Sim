"""
建筑火灾疏散联合仿真系统 — 主入口。
从 YAML 拓扑配置加载所有仿真器和 S2S 连接，运行 Orchestrator 主循环。
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

from core.base import AtomicSimulator
from core.orchestrator import Orchestrator
from core.protocol import (
    EventMessage,
    ExchangeStrategy,
    PortAddress,
    S2SConnection,
)

from atoms.physical.td1_heat import HeatConduction
from atoms.physical.fl3_smoke import Diffusion2D
from atoms.physical.fl4_crowd_fluid import CrowdFluid
from atoms.physical.cm1_rigid_body import RigidBody2D
from atoms.physical.p6_bottleneck import BottleneckGate

from atoms.social.pd3_emotion import EmotionAppraisal
from atoms.social.bp6_stress import StressPerformanceCurve
from atoms.social.s2_herd import HerdBehavior
from atoms.social.sd4_bystander import BystanderEffect
from atoms.social.bp5_attention import AttentionAllocator

from visualization.renderer import SimulationRenderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ------------------------------------------------------------------
# 仿真器工厂
# ------------------------------------------------------------------

_SIM_REGISTRY: Dict[str, type] = {
    "atoms.physical.HeatConduction": HeatConduction,
    "atoms.physical.Diffusion2D": Diffusion2D,
    "atoms.physical.CrowdFluid": CrowdFluid,
    "atoms.physical.RigidBody2D": RigidBody2D,
    "atoms.physical.BottleneckGate": BottleneckGate,
    "atoms.social.EmotionAppraisal": EmotionAppraisal,
    "atoms.social.StressPerformanceCurve": StressPerformanceCurve,
    "atoms.social.HerdBehavior": HerdBehavior,
    "atoms.social.BystanderEffect": BystanderEffect,
    "atoms.social.AttentionAllocator": AttentionAllocator,
}


def _build_simulator(sim_id: str, cfg: Dict[str, Any]) -> AtomicSimulator:
    cls_name = cfg["class"]
    cls = _SIM_REGISTRY.get(cls_name)
    if cls is None:
        raise ValueError(f"Unknown simulator class: {cls_name}")
    params = cfg.get("params", {})
    # 将 list 形式的 grid_size / world_size 转为 tuple
    for key in ("grid_size", "world_size"):
        if key in params and isinstance(params[key], list):
            params[key] = tuple(params[key])
    return cls(sim_id=sim_id, **params)


def _build_connection(cfg: Dict[str, Any]) -> S2SConnection:
    return S2SConnection(
        connection_id=cfg["connection_id"],
        source=PortAddress(cfg["source"]["sim_id"], cfg["source"]["port"]),
        target=PortAddress(cfg["target"]["sim_id"], cfg["target"]["port"]),
        strategy=ExchangeStrategy(cfg["strategy"]),
    )


# ------------------------------------------------------------------
# 场景初始化
# ------------------------------------------------------------------

def _apply_scenario(
    orch: Orchestrator,
    scenario: Dict[str, Any],
) -> None:
    """将场景配置（着火点、墙壁、出口）注入对应仿真器。"""

    cm1: RigidBody2D = orch.simulators.get("CM1")  # type: ignore
    td1: HeatConduction = orch.simulators.get("TD1")  # type: ignore
    fl3: Diffusion2D = orch.simulators.get("FL3")  # type: ignore

    # 设置着火点
    ignition = scenario.get("ignition_points", [])
    if td1 is not None and ignition:
        td1.inputs["ignition_points"] = ignition

    # 设置墙壁
    walls = scenario.get("walls", [])
    if cm1 is not None and walls:
        cm1.set_walls([(tuple(w[0]), tuple(w[1])) for w in walls])

    # 设置障碍物掩码（墙壁在烟雾网格中不可通行）
    if fl3 is not None and walls:
        mask = np.ones(fl3.grid_size, dtype=bool)
        for w in walls:
            a, b = np.array(w[0]), np.array(w[1])
            # 在网格上标记墙壁线段
            _rasterize_wall(mask, a, b, fl3.cell_size, fl3.grid_size)
        fl3.set_obstacles(mask)

    # 设置出口
    exits = scenario.get("exits", [])
    if cm1 is not None and exits:
        cm1.set_exits(exits)


def _rasterize_wall(
    mask: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cell_size: float,
    grid_size: Tuple[int, int],
) -> None:
    """用 Bresenham 思路将墙壁线段标记到网格上。"""
    steps = int(np.linalg.norm(b - a) / cell_size * 2) + 1
    for i in range(steps + 1):
        t = i / max(steps, 1)
        p = a + t * (b - a)
        r, c = int(p[0] / cell_size), int(p[1] / cell_size)
        if 0 <= r < grid_size[0] and 0 <= c < grid_size[1]:
            mask[r, c] = False


# ------------------------------------------------------------------
# 事件处理：跌倒 → 旁观者效应
# ------------------------------------------------------------------

def _setup_fall_event_bridge(orch: Orchestrator) -> None:
    """注册跌倒事件监听，在 CM1 每步后检测新跌倒并触发 SD4。"""

    cm1: RigidBody2D = orch.simulators.get("CM1")  # type: ignore
    if cm1 is None:
        return

    _prev_fallen = np.zeros(cm1.num_agents, dtype=bool)

    original_step = cm1.step

    def _patched_step(dt: float) -> None:
        nonlocal _prev_fallen
        original_step(dt)
        current_fallen = cm1.state["fallen"]
        new_falls = current_fallen & ~_prev_fallen
        if np.any(new_falls):
            event = EventMessage(
                event_type="FALL_EVENT",
                source_sim_id="CM1",
                timestamp=cm1.current_time,
                payload={
                    "fallen": current_fallen.copy(),
                    "active": cm1.state["active"].copy(),
                    "positions": cm1.state["positions"].copy(),
                    "new_fall_indices": np.where(new_falls)[0].tolist(),
                },
            )
            orch.inject_event(event)
            logger.info(
                f"  FALL_EVENT @ t={cm1.current_time:.3f}s  "
                f"agents={np.where(new_falls)[0].tolist()}"
            )
        _prev_fallen = current_fallen.copy()

    cm1.step = _patched_step  # type: ignore


def _handle_fall_event(orch: Orchestrator) -> None:
    """注册 FALL_EVENT 处理函数。"""

    sd4: BystanderEffect = orch.simulators.get("SD4")  # type: ignore
    if sd4 is None:
        return

    def handler(event: EventMessage) -> None:
        sd4.inputs["fallen_mask"] = event.payload.get("fallen")
        sd4.inputs["active_mask"] = event.payload.get("active")
        sd4.inputs["agent_positions"] = event.payload.get("positions")
        sd4.step(0.0)
        sd4.record_output()

        outputs = sd4.get_outputs()
        helpers = outputs.get("helper_ids", [])
        if helpers:
            cm1: RigidBody2D = orch.simulators.get("CM1")  # type: ignore
            if cm1 is not None:
                for hid in helpers:
                    if 0 <= hid < cm1.num_agents:
                        cm1.state["velocities"][hid] = 0.0
                logger.info(f"  Helpers stopped: {helpers}")

    orch.register_event_handler("FALL_EVENT", handler)


# ------------------------------------------------------------------
# 温度突变 → 异步唤醒 PD3
# ------------------------------------------------------------------

def _setup_temperature_spike_monitor(orch: Orchestrator) -> None:
    """监测 TD1 温度突变，超阈值时异步事件唤醒 PD3。"""

    td1: HeatConduction = orch.simulators.get("TD1")  # type: ignore
    if td1 is None:
        return

    _prev_max_temp = [20.0]
    SPIKE_THRESHOLD = 100.0  # 单步温升超过此值触发事件

    original_step = td1.step

    def _patched_step(dt: float) -> None:
        original_step(dt)
        current_max = float(np.max(td1.state["temperature_field"]))
        delta = current_max - _prev_max_temp[0]
        if delta > SPIKE_THRESHOLD:
            event = EventMessage(
                event_type="TEMP_SPIKE",
                source_sim_id="TD1",
                timestamp=td1.current_time,
                payload={
                    "temperature_field": td1.state["temperature_field"].copy(),
                    "max_temp": current_max,
                    "delta": delta,
                },
            )
            orch.inject_event(event)
            logger.info(
                f"  TEMP_SPIKE @ t={td1.current_time:.3f}s  "
                f"max={current_max:.1f}  delta={delta:.1f}"
            )
        _prev_max_temp[0] = current_max

    td1.step = _patched_step  # type: ignore

    def _handle_spike(event: EventMessage) -> None:
        pd3: EmotionAppraisal = orch.simulators.get("PD3")  # type: ignore
        if pd3 is None:
            return
        pd3.inputs["local_temperature"] = event.payload.get("temperature_field")
        pd3.step(0.0)
        pd3.record_output()

    orch.register_event_handler("TEMP_SPIKE", _handle_spike)


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------

def main(config_path: str = "config/topology.yaml") -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 1. 创建 Orchestrator ----
    orch = Orchestrator()

    # ---- 2. 注册时间分组 ----
    for group_name, group_cfg in config["time_groups"].items():
        orch.register_time_group(group_name, group_cfg["dt"])
    logger.info("Time groups registered.")

    # ---- 3. 实例化并注册仿真器 ----
    for sim_id, sim_cfg in config["simulators"].items():
        sim = _build_simulator(sim_id, sim_cfg)
        group = sim_cfg["group"]
        orch.register_simulator(sim, group)
        logger.info(f"  Registered {sim_id} -> group={group}")

    # ---- 4. 建立 S2S 连接 ----
    for conn_cfg in config["connections"]:
        conn = _build_connection(conn_cfg)
        orch.add_connection(conn)
    logger.info(f"S2S connections established: {len(orch.connections)}")

    # ---- 5. 因果性检查 ----
    loops = orch.causality_guard.detect_algebraic_loops()
    if loops:
        for loop in loops:
            logger.warning(f"Algebraic loop detected: {loop}")

    # ---- 6. 场景初始化 ----
    scenario = config.get("scenario", {})
    _apply_scenario(orch, scenario)

    # ---- 7. 注册事件桥 ----
    _setup_fall_event_bridge(orch)
    _handle_fall_event(orch)
    _setup_temperature_spike_monitor(orch)

    # ---- 8. 打印拓扑摘要 ----
    logger.info("\n" + orch.summary())

    # ---- 9. 运行仿真 ----
    duration = scenario.get("duration", 60.0)
    snap_interval = scenario.get("snapshot_interval", 0.5)

    logger.info(f"Starting simulation: duration={duration}s, snapshot_interval={snap_interval}s")
    t0 = time.perf_counter()
    orch.run(duration=duration, snapshot_interval=snap_interval)
    elapsed = time.perf_counter() - t0
    logger.info(f"Simulation finished in {elapsed:.2f}s (wall-clock)")

    # ---- 10. 统计 ----
    cm1: RigidBody2D = orch.simulators.get("CM1")  # type: ignore
    if cm1 is not None:
        active = cm1.state["active"]
        fallen = cm1.state["fallen"]
        escaped = int(np.sum(~active & ~fallen))
        still_inside = int(np.sum(active))
        on_ground = int(np.sum(fallen))
        logger.info(
            f"Results: escaped={escaped}, still_inside={still_inside}, fallen={on_ground}"
        )

    # ---- 11. 输出 ----
    renderer = SimulationRenderer(output_dir="output")

    json_path = renderer.export_history_json(orch.history)
    logger.info(f"History JSON exported: {json_path}")

    walls = scenario.get("walls", [])
    exits = scenario.get("exits", [])
    frames = renderer.render_all_to_arrays(
        orch.history,
        world_size=(25.0, 25.0),
        walls=walls,
        exits=exits,
    )
    logger.info(f"Rendered {len(frames)} frames in memory (no PNG disk I/O).")

    # ---- 12. 合成动图 / 视频 ----
    gif_path = renderer.export_gif(frames, fps=4, scale=0.6)
    if gif_path:
        logger.info(f"GIF exported: {gif_path}")

    video_path = renderer.export_video(frames, fps=8)
    if video_path:
        logger.info(f"Video exported: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building Fire Evacuation Co-Simulation")
    parser.add_argument(
        "-c", "--config",
        default="config/topology.yaml",
        help="Path to topology YAML config",
    )
    args = parser.parse_args()
    main(args.config)
