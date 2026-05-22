"""
BP5: AttentionAllocator — 注意力分配仿真（逐个体）。
认知限制：每个个体所处位置的烟雾浓度和环境噪声独立占用其注意力资源，
导致其忽略逃生指示牌的程度各异。
Natural DT = 0.1s

端口契约 (Input Port Schema)
----------------------------
- smoke_exposure : ndarray[n] float in [0, 1]
    每个个体所处位置的烟雾暴露度；由 S2S 总线上的 transform 按 CM1.positions
    在烟雾浓度场中采样得到。
- noise_level    : float in [0, 1] 或 ndarray[n]
    环境噪声水平（目前假设场景均匀，标量即可）；如果未来要做"按位置的噪声场"，
    在 core/transforms.py 新增相应 transform 并递交 ndarray[n] 即可。

兼容性兜底：若输入为标量或 None，会被广播成长度 n 的数组。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class AttentionAllocator(AtomicSimulator):
    """逐个体注意力资源分配仿真器。"""

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

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _to_per_agent(self, value: Any, default: float) -> np.ndarray:
        n = self.num_agents
        if value is None:
            return np.full(n, default, dtype=np.float64)

        arr = np.asarray(value, dtype=np.float64).ravel()
        if arr.size == 0:
            return np.full(n, default, dtype=np.float64)
        if arr.size == 1:
            return np.full(n, float(arr[0]), dtype=np.float64)
        if arr.size < n:
            pad = np.full(n - arr.size, default, dtype=np.float64)
            return np.concatenate([arr, pad])
        return arr[:n]

    # ------------------------------------------------------------------
    # 主步进
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        smoke = np.clip(self._to_per_agent(self.inputs.get("smoke_exposure"), 0.0), 0.0, 1.0)
        noise = np.clip(self._to_per_agent(self.inputs.get("noise_level"), 0.0), 0.0, 1.0)

        # 逐个体注意力占用
        attention_used = self.smoke_cost * smoke + self.noise_cost * noise
        remaining = 1.0 - np.clip(attention_used, 0.0, 1.0)

        self.state["perception_mask"] = np.clip(remaining, 0.05, 1.0)
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {"perception_mask": self.state["perception_mask"].copy()}

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["smoke_exposure", "noise_level"],
            "outputs": ["perception_mask"],
        }
