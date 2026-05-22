"""
PD3: EmotionAppraisal — 恐慌情绪生成（逐个体）。
理论依据：Scherer 的认知评估模型 (CPM)。
每个个体基于其所处环境的"新奇性""目标阻碍""应对潜力"独立评估自身恐慌值。
Natural DT = 0.5s

设计原则
--------
本原子不负责 **任何** 物理量到 [0, 1] 的归一化、场聚合或空间采样；
所有来自上游仿真器的输入都应该已经是 **逐个体的** ndarray[num_agents]，
单位/语义符合本原子的端口契约。

具体的"场 → 逐个体"空间采样在 core/transforms.py 中定义，
由 S2S 总线按每个 tick 的 CM1.positions 在 transform 里完成。

端口契约 (Input Port Schema)
----------------------------
- local_danger     : ndarray[n] float in [0, 1]  — 每个个体位置处的热暴露危险度
- smoke_exposure   : ndarray[n] float in [0, 1]  — 每个个体位置处的烟雾暴露度
- exit_visibility  : ndarray[n] float in [0, 1]  — 每个个体位置处的可见度 (1=完全可见)

兼容性兜底：若输入为标量或 None，会被广播成长度 n 的数组。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class EmotionAppraisal(AtomicSimulator):
    """基于 CPM 的逐个体恐慌情绪评估仿真器。"""

    def __init__(
        self,
        sim_id: str = "PD3",
        num_agents: int = 50,
        novelty_weight: float = 0.3,
        obstruction_weight: float = 0.4,
        coping_weight: float = 0.3,
        inertia: float = 0.4,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_agents = num_agents
        self.w_novelty = novelty_weight
        self.w_obstruct = obstruction_weight
        self.w_coping = coping_weight
        self.inertia = inertia

        self.state["panic_level"] = np.zeros(num_agents, dtype=np.float64)

    # ------------------------------------------------------------------
    # 内部工具：把任意形式的输入规整为长度 n 的 float 数组
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
        panic = self.state["panic_level"]

        danger = np.clip(self._to_per_agent(self.inputs.get("local_danger"), 0.0), 0.0, 1.0)
        smoke = np.clip(self._to_per_agent(self.inputs.get("smoke_exposure"), 0.0), 0.0, 1.0)
        exit_vis = np.clip(self._to_per_agent(self.inputs.get("exit_visibility"), 1.0), 0.0, 1.0)

        # CPM 三要素（逐个体）
        novelty = np.maximum(danger, smoke)           # 热/烟中较严重者驱动新奇性
        obstruction = 1.0 - exit_vis                   # 可见度越低阻碍越大
        coping = np.clip(1.0 - smoke * 0.5, 0.0, 1.0)  # 烟雾削弱应对潜力

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
            "inputs": ["local_danger", "smoke_exposure", "exit_visibility"],
            "outputs": ["panic_level"],
        }
