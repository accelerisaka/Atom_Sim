"""
PD3: EmotionAppraisal — 恐慌情绪生成。
理论依据：Scherer 的认知评估模型 (CPM)。
基于"新奇性"、"目标阻碍"和"应对潜力"评估恐慌值。
Natural DT = 0.5s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class EmotionAppraisal(AtomicSimulator):
    """基于 CPM 的恐慌情绪评估仿真器。"""

    def __init__(
        self,
        sim_id: str = "PD3",
        num_agents: int = 50,
        novelty_weight: float = 0.3,
        obstruction_weight: float = 0.4,
        coping_weight: float = 0.3,
        inertia: float = 0.7,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_agents = num_agents
        self.w_novelty = novelty_weight
        self.w_obstruct = obstruction_weight
        self.w_coping = coping_weight
        self.inertia = inertia

        self.state["panic_level"] = np.zeros(num_agents, dtype=np.float64)

    def step(self, dt: float) -> None:
        panic = self.state["panic_level"]
        n = self.num_agents

        local_temp = self.inputs.get("local_temperature", 20.0)
        smoke = self.inputs.get("smoke_density", 0.0)
        exit_visible = self.inputs.get("is_exit_visible", True)

        # --- 新奇性评估：温度 / 烟雾突然升高 ---
        if isinstance(local_temp, np.ndarray):
            temp_vals = np.clip((local_temp - 40.0) / 300.0, 0.0, 1.0)
        else:
            temp_vals = np.full(n, np.clip((float(local_temp) - 40.0) / 300.0, 0.0, 1.0))

        if isinstance(smoke, np.ndarray):
            smoke_vals = np.clip(smoke, 0.0, 1.0)
        else:
            smoke_vals = np.full(n, np.clip(float(smoke), 0.0, 1.0))

        novelty = np.maximum(temp_vals, smoke_vals)

        # --- 目标阻碍：出口不可见 = 高阻碍 ---
        if isinstance(exit_visible, np.ndarray):
            obstruction = 1.0 - exit_visible.astype(np.float64)
        elif isinstance(exit_visible, bool):
            obstruction = np.full(n, 0.0 if exit_visible else 1.0)
        else:
            obstruction = np.full(n, 1.0 - float(exit_visible))

        # --- 应对潜力：受体力和清醒度影响（简化为固定值 + 噪声）---
        coping = np.clip(1.0 - smoke_vals * 0.5, 0.0, 1.0)

        # --- CPM 综合评估 ---
        raw_panic = (
            self.w_novelty * novelty
            + self.w_obstruct * obstruction
            + self.w_coping * (1.0 - coping)
        )

        # 惯性平滑：避免恐慌瞬间剧变
        new_panic = self.inertia * panic + (1.0 - self.inertia) * raw_panic
        self.state["panic_level"] = np.clip(new_panic, 0.0, 1.0)

        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {"panic_level": self.state["panic_level"].copy()}

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["local_temperature", "smoke_density", "is_exit_visible"],
            "outputs": ["panic_level"],
        }
