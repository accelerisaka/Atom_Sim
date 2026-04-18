"""
BP6: StressPerformanceCurve — 压力-表现曲线。
理论依据：耶克斯-多德森定律 (Yerkes-Dodson Law)。
随着压力上升，表现先升后降；极高压力下导航能力和决策速度骤降。
Natural DT = 0.5s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class StressPerformanceCurve(AtomicSimulator):
    """Yerkes-Dodson 压力-效率映射仿真器。"""

    def __init__(
        self,
        sim_id: str = "BP6",
        num_agents: int = 50,
        optimal_arousal: float = 0.4,
        curve_width: float = 0.3,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_agents = num_agents
        self.optimal_arousal = optimal_arousal
        self.curve_width = curve_width

        self.state["cognitive_efficiency"] = np.ones(num_agents, dtype=np.float64)

    def step(self, dt: float) -> None:
        panic = self.inputs.get("panic_level")
        if panic is None:
            self.current_time += dt
            return

        panic = np.asarray(panic, dtype=np.float64)

        # Yerkes-Dodson 倒 U 型曲线：
        # efficiency = exp(-((arousal - optimal)^2) / (2 * sigma^2))
        sigma = self.curve_width
        eff = np.exp(-((panic - self.optimal_arousal) ** 2) / (2.0 * sigma ** 2))

        # 极高恐慌时增加额外惩罚（决策瘫痪）
        high_panic = panic > 0.8
        eff[high_panic] *= 0.5

        self.state["cognitive_efficiency"] = np.clip(eff, 0.1, 1.0)
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "cognitive_efficiency_multiplier": self.state["cognitive_efficiency"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["panic_level"],
            "outputs": ["cognitive_efficiency_multiplier"],
        }
