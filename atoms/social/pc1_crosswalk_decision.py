"""
PC1: CrosswalkDecision — 行人过马路决策。
社会心理：综合信号灯、性格、从众、危险感知与等待焦虑，输出过街意图与期望速度。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class CrosswalkDecision(AtomicSimulator):
    """行人是否过马路的决策仿真器。"""

    def __init__(
        self,
        sim_id: str = "PC1",
        num_agents: int = 40,
        cross_speed: float = 1.2,
        herd_cross_threshold: float = 0.35,
        danger_block_threshold: float = 0.65,
        seed: int = 123,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.num_agents = num_agents
        self.cross_speed = cross_speed
        self.herd_threshold = herd_cross_threshold
        self.danger_threshold = danger_block_threshold
        self._rng = np.random.default_rng(seed)

        self.state["cross_intent"] = np.zeros(num_agents, dtype=bool)
        self.state["desired_velocity"] = np.zeros((num_agents, 2), dtype=np.float64)
        self.state["waiting_frustration"] = np.zeros(num_agents, dtype=np.float64)
        self.state["neighbors_crossing_ratio"] = np.zeros(num_agents, dtype=np.float64)

    def _to_per_agent(self, value: Any, default: float, n: int) -> np.ndarray:
        if value is None:
            return np.full(n, default, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).ravel()
        if arr.size == 1:
            return np.full(n, float(arr[0]), dtype=np.float64)
        if arr.size < n:
            pad = np.full(n - arr.size, default, dtype=np.float64)
            return np.concatenate([arr, pad])
        return arr[:n]

    def step(self, dt: float) -> None:
        n = self.num_agents
        signal = float(self._to_per_agent(self.inputs.get("pedestrian_signal"), 0.0, 1)[0])
        jaywalk = np.clip(self._to_per_agent(self.inputs.get("jaywalk_tendency"), 0.3, n), 0, 1)
        patience = np.clip(self._to_per_agent(self.inputs.get("patience"), 0.5, n), 0.05, 1)
        perception = np.clip(self._to_per_agent(self.inputs.get("perception_mask"), 0.8, n), 0.05, 1)
        danger = np.clip(self._to_per_agent(self.inputs.get("local_danger"), 0.0, n), 0, 1)
        impatience = np.clip(self._to_per_agent(self.inputs.get("impatience"), 0.0, n), 0, 1)
        waiting = np.clip(self._to_per_agent(self.inputs.get("waiting_time"), 0.0, n), 0, 120)
        herd_ratio = np.clip(self._to_per_agent(self.inputs.get("neighbors_crossing_ratio"), 0.0, n), 0, 1)

        herd_bias = self.inputs.get("herd_velocity_bias")
        if herd_bias is not None:
            hb = np.asarray(herd_bias, dtype=np.float64)
            if hb.shape[0] >= n:
                herd_north = np.clip(hb[:n, 1] / (self.cross_speed + 1e-6), 0, 1)
                herd_ratio = np.maximum(herd_ratio, herd_north)

        legal = signal > 0.5
        frustration = np.clip(
            (1.0 - signal) * (waiting / 30.0) * (1.0 - patience) + impatience * 0.5,
            0, 1,
        )

        cross_intent = np.zeros(n, dtype=bool)
        desired = np.zeros((n, 2), dtype=np.float64)

        for i in range(n):
            # 合法通行
            if legal:
                want = True
            else:
                # 闯红灯：性格 + 从众 + 焦虑
                jaywalk_drive = jaywalk[i] * (1.0 - perception[i] * 0.5)
                herd_drive = herd_ratio[i] * (1.0 - jaywalk[i] * 0.3)
                frustration_drive = frustration[i] * (1.0 - patience[i])

                p_jaywalk = np.clip(
                    jaywalk_drive * 0.5 + herd_drive * 0.35 + frustration_drive * 0.4,
                    0, 1,
                )
                want = p_jaywalk > self._rng.random()

                # 极端闯红灯者：高概率无视红灯
                if jaywalk[i] > 0.75:
                    want = want or (self._rng.random() < jaywalk[i] * 0.85)

                # 从众：看见足够多人过街则跟随
                if herd_ratio[i] > self.herd_threshold and jaywalk[i] > 0.15:
                    want = True

            # 危险阻挡（极端闯红灯者降低阻挡）
            danger_resist = danger[i] * (1.0 - jaywalk[i] * 0.6)
            if danger_resist > self.danger_threshold and jaywalk[i] < 0.7:
                want = False

            cross_intent[i] = want
            if want:
                speed_factor = 0.6 + 0.4 * (1.0 - danger[i] * 0.5)
                desired[i, 1] = self.cross_speed * speed_factor
            else:
                desired[i] = 0.0

        self.state["cross_intent"] = cross_intent
        self.state["desired_velocity"] = desired
        self.state["waiting_frustration"] = frustration
        self.state["neighbors_crossing_ratio"] = herd_ratio
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "cross_intent": self.state["cross_intent"].copy(),
            "desired_velocity": self.state["desired_velocity"].copy(),
            "waiting_frustration": self.state["waiting_frustration"].copy(),
            "neighbors_crossing_ratio": self.state["neighbors_crossing_ratio"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [
                "pedestrian_signal", "jaywalk_tendency", "patience",
                "perception_mask", "local_danger", "impatience",
                "waiting_time", "neighbors_crossing_ratio", "herd_velocity_bias",
            ],
            "outputs": [
                "cross_intent", "desired_velocity",
                "waiting_frustration", "neighbors_crossing_ratio",
            ],
        }
