"""
CM1: RigidBody2D — 微观碰撞与动力学仿真。
物理规律：牛顿第二定律 F=ma，社会力模型 (Social Force Model)。
实现：个体间排斥力 + 障碍物排斥力 + 目标驱动力。
Natural DT = 0.02s (高频计算防穿模)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.base import AtomicSimulator


class RigidBody2D(AtomicSimulator):
    """基于社会力模型的微观人群碰撞动力学仿真器。"""

    # 社会力模型参数
    TAU = 0.5          # 期望速度松弛时间 (s)
    A_REP = 2000.0     # 个体间排斥力振幅 (N)
    B_REP = 0.08       # 排斥力衰减尺度 (m)
    A_WALL = 2000.0    # 墙壁排斥力振幅 (N)
    B_WALL = 0.08      # 墙壁排斥力衰减尺度 (m)
    AGENT_RADIUS = 0.25  # 个体半径 (m)
    MAX_SPEED = 2.0    # 最大速度限制 (m/s)
    MASS = 80.0        # 个体质量 (kg)
    HERD_WEIGHT = 0.3  # 从众偏置对期望速度的叠加权重 (0=忽略, 1=平均)
    EXIT_REACH_DIST = 1.2  # 到达出口判定半径 (m)

    def __init__(
        self,
        sim_id: str = "CM1",
        num_agents: int = 50,
        world_size: Tuple[float, float] = (25.0, 25.0),
    ):
        super().__init__(sim_id, natural_dt=0.02)
        self.num_agents = num_agents
        self.world_size = world_size

        rng = np.random.default_rng(42)
        self.state["positions"] = rng.uniform(
            [1.0, 1.0], [world_size[0] - 1, world_size[1] - 1],
            size=(num_agents, 2),
        )
        self.state["velocities"] = np.zeros((num_agents, 2), dtype=np.float64)
        self.state["desired_speed"] = np.full(num_agents, 1.3)  # m/s
        self.state["active"] = np.ones(num_agents, dtype=bool)
        self.state["fallen"] = np.zeros(num_agents, dtype=bool)

        self._wall_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        self._exit_positions: List[np.ndarray] = []

    def set_walls(self, segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        self._wall_segments = [
            (np.array(a, dtype=np.float64), np.array(b, dtype=np.float64))
            for a, b in segments
        ]

    def set_exits(self, exits: List[Tuple[float, float]]) -> None:
        self._exit_positions = [np.array(e, dtype=np.float64) for e in exits]

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        pos = self.state["positions"]
        vel = self.state["velocities"]
        active = self.state["active"]
        fallen = self.state["fallen"]
        n = self.num_agents

        # 外部输入
        desired_vel_input = self.inputs.get("desired_velocity")
        herd_bias_input = self.inputs.get("herd_bias")
        flow_constraint = self.inputs.get("flow_constraint", 1.0)
        ext_impulses = self.inputs.get("external_impulses")
        efficiency = self.inputs.get("cognitive_efficiency", 1.0)

        # 语义分离：
        #   desired_velocity  -> 完整覆盖（用于脚本化控制/测试）
        #   herd_bias         -> 叠加偏置（用于社会力耦合，如 S2 从众）
        # 二者可同时存在，从众偏置始终与出口驱动方向混合，避免出现
        # "整体期望速度被拉至 0" 的冻结人群伪影。
        if desired_vel_input is not None and isinstance(desired_vel_input, np.ndarray):
            desired_vel = desired_vel_input.astype(np.float64).copy()
        else:
            desired_vel = self._compute_desired_velocity(pos, active, efficiency)

        if herd_bias_input is not None and isinstance(herd_bias_input, np.ndarray):
            bias = np.asarray(herd_bias_input, dtype=np.float64)
            if bias.shape == desired_vel.shape:
                desired_vel = (1.0 - self.HERD_WEIGHT) * desired_vel + self.HERD_WEIGHT * bias

        # 应用瓶颈限速
        if isinstance(flow_constraint, (int, float)):
            desired_vel *= float(flow_constraint)
        elif isinstance(flow_constraint, np.ndarray):
            desired_vel *= flow_constraint.reshape(-1, 1) if flow_constraint.ndim == 1 else flow_constraint

        forces = np.zeros_like(pos)

        # 1) 驱动力
        f_drive = self.MASS * (desired_vel - vel) / self.TAU
        forces += f_drive

        # 2) 个体间排斥力
        for i in range(n):
            if not active[i] or fallen[i]:
                continue
            for j in range(i + 1, n):
                if not active[j]:
                    continue
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = np.random.randn(2) * 0.01
                    dist = np.linalg.norm(diff)
                overlap = 2 * self.AGENT_RADIUS - dist
                n_ij = diff / dist
                f_rep = self.A_REP * np.exp(overlap / self.B_REP) * n_ij
                forces[i] += f_rep
                forces[j] -= f_rep

                # 身体接触力
                if overlap > 0:
                    forces[i] += 1200.0 * overlap * n_ij
                    forces[j] -= 1200.0 * overlap * n_ij

        # 3) 墙壁排斥力
        for i in range(n):
            if not active[i]:
                continue
            for seg_a, seg_b in self._wall_segments:
                closest = self._closest_point_on_segment(pos[i], seg_a, seg_b)
                diff = pos[i] - closest
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                n_iw = diff / dist
                f_wall = self.A_WALL * np.exp((self.AGENT_RADIUS - dist) / self.B_WALL) * n_iw
                forces[i] += f_wall

        # 4) 外部冲击
        if ext_impulses is not None:
            forces += np.asarray(ext_impulses)

        # 跌倒的人不动
        forces[fallen] = 0.0

        # 积分 (半隐式欧拉)
        acc = forces / self.MASS
        vel_new = vel + acc * dt

        # 速度限制
        speed = np.linalg.norm(vel_new, axis=1, keepdims=True)
        max_spd = self.MAX_SPEED * flow_constraint if isinstance(flow_constraint, (int, float)) else self.MAX_SPEED
        too_fast = speed > max_spd
        vel_new = np.where(too_fast, vel_new / speed * max_spd, vel_new)

        pos_new = pos + vel_new * dt

        # 边界钳制
        pos_new[:, 0] = np.clip(pos_new[:, 0], 0.0, self.world_size[0])
        pos_new[:, 1] = np.clip(pos_new[:, 1], 0.0, self.world_size[1])

        # 检测是否到达出口（仅对仍然活跃的个体生效）
        for ex in self._exit_positions:
            dist_to_exit = np.linalg.norm(pos_new - ex, axis=1)
            escaped = (dist_to_exit < self.EXIT_REACH_DIST) & active & ~fallen
            active[escaped] = False

        # 检测跌倒（密集碰撞 + 随机概率）
        for i in range(n):
            if not active[i] or fallen[i]:
                continue
            if np.linalg.norm(acc[i]) > 15.0 and np.random.random() < 0.02:
                fallen[i] = True
                vel_new[i] = 0.0

        self.state["positions"] = pos_new
        self.state["velocities"] = vel_new
        self.state["active"] = active
        self.state["fallen"] = fallen
        self.current_time += dt

    def _compute_desired_velocity(
        self, pos: np.ndarray, active: np.ndarray, efficiency: Any
    ) -> np.ndarray:
        """默认：朝最近出口移动。efficiency 可为标量或 per-agent 数组。"""
        n = pos.shape[0]
        desired = np.zeros_like(pos)
        if not self._exit_positions:
            return desired

        eff_arr = np.asarray(efficiency, dtype=np.float64).ravel()
        if eff_arr.size == 0:
            eff_arr = np.ones(n)
        elif eff_arr.size == 1:
            eff_arr = np.full(n, float(eff_arr[0]))
        elif eff_arr.size != n:
            eff_arr = np.full(n, float(eff_arr.mean()))

        for i in range(n):
            if not active[i]:
                continue
            dists = [np.linalg.norm(pos[i] - ex) for ex in self._exit_positions]
            nearest = self._exit_positions[int(np.argmin(dists))]
            direction = nearest - pos[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            desired[i] = direction * self.state["desired_speed"][i] * eff_arr[i]

        return desired

    @staticmethod
    def _closest_point_on_segment(
        p: np.ndarray, a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-12:
            return a.copy()
        t = np.clip(np.dot(p - a, ab) / ab_sq, 0.0, 1.0)
        return a + t * ab

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "positions": self.state["positions"].copy(),
            "velocities": self.state["velocities"].copy(),
            "active": self.state["active"].copy(),
            "fallen": self.state["fallen"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [
                "desired_velocity", "herd_bias", "external_impulses",
                "flow_constraint", "cognitive_efficiency",
            ],
            "outputs": ["positions", "velocities", "active", "fallen"],
        }
