"""
EG1: BroadcastCompliance — 应急广播服从度仿真。
社会心理：恐慌较低且仍保有认知资源的个体会听从广播规划的备用路线。
实现：综合广播强度、个体恐慌与感知掩码，输出逐个体服从权重。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class BroadcastCompliance(AtomicSimulator):
    """逐个体应急广播服从权重仿真器。"""

    def __init__(
        self,
        sim_id: str = "EG1",
        num_agents: int = 50,
        sanity_threshold: float = 0.55,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.num_agents = num_agents
        self.sanity_threshold = sanity_threshold

        self.state["compliance_weight"] = np.zeros(num_agents, dtype=np.float64)

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

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        active = float(self.inputs.get("broadcast_active", 0.0))
        guidance = float(self.inputs.get("guidance_intensity", 0.0))
        panic = np.clip(self._to_per_agent(self.inputs.get("panic_level"), 0.0), 0.0, 1.0)
        perception = np.clip(self._to_per_agent(self.inputs.get("perception_mask"), 1.0), 0.05, 1.0)

        # 恐慌过高则丧失听从广播的能力
        sanity = np.clip(1.0 - panic, 0.0, 1.0)
        sane_enough = sanity > (1.0 - self.sanity_threshold)

        raw = active * guidance * perception * sanity
        raw = np.where(sane_enough, raw, raw * 0.15)

        self.state["compliance_weight"] = np.clip(raw, 0.0, 1.0)
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {"compliance_weight": self.state["compliance_weight"].copy()}

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [
                "broadcast_active", "guidance_intensity",
                "panic_level", "perception_mask",
            ],
            "outputs": ["compliance_weight"],
        }
