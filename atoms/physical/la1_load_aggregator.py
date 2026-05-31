"""
LA1: LoadAggregator — 逐户多类型负荷汇总（取暖 + EV）。
物理规律：节点功率守恒  P_i = P_heat_i + P_ev_i
Natural DT = 0.5s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class LoadAggregator(AtomicSimulator):
    """逐户负荷汇总仿真器。"""

    def __init__(
        self,
        sim_id: str = "LA1",
        num_households: int = 24,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_households = num_households
        self.state["household_loads"] = np.zeros(num_households, dtype=np.float64)
        self.state["total_heating"] = 0.0
        self.state["total_ev"] = 0.0

    def _to_per_house(self, value: Any) -> np.ndarray:
        n = self.num_households
        if value is None:
            return np.zeros(n, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).ravel()
        if arr.size == 0:
            return np.zeros(n, dtype=np.float64)
        if arr.size < n:
            pad = np.zeros(n - arr.size, dtype=np.float64)
            return np.concatenate([arr, pad])
        return arr[:n]

    def step(self, dt: float) -> None:
        heating = self._to_per_house(self.inputs.get("heating_load"))
        ev = self._to_per_house(self.inputs.get("ev_load"))
        combined = heating + ev

        self.state["household_loads"] = combined
        self.state["total_heating"] = float(np.sum(heating))
        self.state["total_ev"] = float(np.sum(ev))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "household_loads": self.state["household_loads"].copy(),
            "total_heating": self.state["total_heating"],
            "total_ev": self.state["total_ev"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["heating_load", "ev_load"],
            "outputs": ["household_loads", "total_heating", "total_ev"],
        }
