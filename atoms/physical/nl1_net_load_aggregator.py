"""
NL1: NetLoadAggregator — 逐户净负荷汇总（基础负荷 - 光伏 + 储能净流）。
物理规律：P_net_i = P_base_i - P_pv_i + P_battery_net_i
Natural DT = 0.5s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class NetLoadAggregator(AtomicSimulator):
    """逐户净负荷汇总仿真器。"""

    def __init__(
        self,
        sim_id: str = "NL1",
        num_households: int = 24,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_households = num_households
        self.state["household_loads"] = np.zeros(num_households, dtype=np.float64)
        self.state["total_base"] = 0.0
        self.state["total_pv"] = 0.0
        self.state["total_battery_net"] = 0.0

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
        base = self._to_per_house(self.inputs.get("base_load"))
        pv = self._to_per_house(self.inputs.get("pv_power"))
        bat = self._to_per_house(self.inputs.get("battery_net_kw"))
        net = base - pv + bat

        self.state["household_loads"] = net
        self.state["total_base"] = float(np.sum(base))
        self.state["total_pv"] = float(np.sum(pv))
        self.state["total_battery_net"] = float(np.sum(bat))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "household_loads": self.state["household_loads"].copy(),
            "total_base": self.state["total_base"],
            "total_pv": self.state["total_pv"],
            "total_battery_net": self.state["total_battery_net"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["base_load", "pv_power", "battery_net_kw"],
            "outputs": [
                "household_loads",
                "total_base",
                "total_pv",
                "total_battery_net",
            ],
        }
