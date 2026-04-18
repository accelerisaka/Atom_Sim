"""
BP5: AttentionAllocator — 注意力分配仿真。
认知限制：浓烟和高温会强制占用注意力资源，导致个体忽略逃生指示牌。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class AttentionAllocator(AtomicSimulator):
    """注意力资源分配仿真器。"""

    def __init__(
        self,
        sim_id: str = "BP5",
        num_agents: int = 50,
        smoke_attention_cost: float = 0.6,
        noise_attention_cost: float = 0.3,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.num_agents = num_agents
        self.smoke_cost = smoke_attention_cost
        self.noise_cost = noise_attention_cost

        # perception_mask: 1.0 = 完全感知, 0.0 = 完全失去感知
        self.state["perception_mask"] = np.ones(num_agents, dtype=np.float64)

    def step(self, dt: float) -> None:
        n = self.num_agents

        smoke = self.inputs.get("smoke_density", 0.0)
        noise = self.inputs.get("noise_level", 0.0)

        if isinstance(smoke, np.ndarray):
            smoke_vals = np.clip(smoke.ravel()[:n], 0.0, 1.0)
            if smoke_vals.shape[0] < n:
                smoke_vals = np.pad(smoke_vals, (0, n - smoke_vals.shape[0]), constant_values=0.0)
        else:
            smoke_vals = np.full(n, np.clip(float(smoke), 0.0, 1.0))

        if isinstance(noise, np.ndarray):
            noise_vals = np.clip(noise.ravel()[:n], 0.0, 1.0)
            if noise_vals.shape[0] < n:
                noise_vals = np.pad(noise_vals, (0, n - noise_vals.shape[0]), constant_values=0.0)
        else:
            noise_vals = np.full(n, np.clip(float(noise), 0.0, 1.0))

        # 注意力被占用的比例
        attention_used = (
            self.smoke_cost * smoke_vals
            + self.noise_cost * noise_vals
        )

        # 剩余可用于感知逃生信息的注意力
        remaining = 1.0 - np.clip(attention_used, 0.0, 1.0)

        self.state["perception_mask"] = np.clip(remaining, 0.05, 1.0)
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {"perception_mask": self.state["perception_mask"].copy()}

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["smoke_density", "noise_level"],
            "outputs": ["perception_mask"],
        }
