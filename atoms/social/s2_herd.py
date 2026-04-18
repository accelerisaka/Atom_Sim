"""
S2: HerdBehavior — 从众行为仿真。
理论依据：社会干扰理论。
当个体对路径不确定时，大概率跟随周围人群移动方向。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class HerdBehavior(AtomicSimulator):
    """从众行为：基于邻居速度和个体信心计算速度偏置。"""

    def __init__(
        self,
        sim_id: str = "S2",
        num_agents: int = 50,
        neighbor_radius: float = 3.0,
        max_bias_magnitude: float = 0.8,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.num_agents = num_agents
        self.neighbor_radius = neighbor_radius
        self.max_bias = max_bias_magnitude

        self.state["herd_velocity_bias"] = np.zeros((num_agents, 2), dtype=np.float64)

    def step(self, dt: float) -> None:
        positions = self.inputs.get("agent_positions")
        velocities = self.inputs.get("neighbor_velocities")
        confidence = self.inputs.get("individual_confidence")

        if positions is None or velocities is None:
            self.current_time += dt
            return

        pos = np.asarray(positions, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)
        n = min(pos.shape[0], self.num_agents)

        if confidence is not None:
            conf = np.asarray(confidence, dtype=np.float64)[:n]
        else:
            conf = np.full(n, 0.5)

        bias = np.zeros((self.num_agents, 2), dtype=np.float64)

        for i in range(n):
            diffs = pos[:n] - pos[i]
            dists = np.linalg.norm(diffs, axis=1)
            neighbors = (dists < self.neighbor_radius) & (dists > 1e-6)

            if not np.any(neighbors):
                continue

            avg_vel = np.mean(vel[:n][neighbors], axis=0)
            # 信心低 => 更易从众
            herd_strength = 1.0 - conf[i]
            bias[i] = avg_vel * herd_strength

        # 限幅
        mag = np.linalg.norm(bias, axis=1, keepdims=True)
        too_large = mag > self.max_bias
        bias = np.where(too_large, bias / (mag + 1e-9) * self.max_bias, bias)

        self.state["herd_velocity_bias"] = bias
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {"herd_velocity_bias": self.state["herd_velocity_bias"].copy()}

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["neighbor_velocities", "individual_confidence", "agent_positions"],
            "outputs": ["herd_velocity_bias"],
        }
