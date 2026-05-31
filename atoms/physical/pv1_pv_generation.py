"""
PV1: PVGeneration — 逐户屋顶光伏发电功率。
物理规律：P_pv_i = irradiance_factor · pv_capacity_i · performance_ratio
Natural DT = 0.5s

端口契约 (Input Port Schema)
----------------------------
- irradiance_factor : float in [0, 1]
- pv_capacity_kw    : ndarray[N] float, kW

端口契约 (Output Port Schema)
----------------------------
- pv_power      : ndarray[N] float, kW (发电功率，≥0)
- total_pv      : float, kW
- avg_pv_factor : float — 当前发电占装机比例
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class PVGeneration(AtomicSimulator):
    """逐户光伏发电仿真器。"""

    def __init__(
        self,
        sim_id: str = "PV1",
        num_households: int = 24,
        performance_ratio: float = 0.92,
        default_irradiance: float = 0.0,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_households = num_households
        self.performance_ratio = float(performance_ratio)
        self.default_irradiance = float(default_irradiance)

        self.state["pv_power"] = np.zeros(num_households, dtype=np.float64)
        self.state["total_pv"] = 0.0
        self.state["avg_pv_factor"] = 0.0

    def _to_per_house(self, value: Any, default: float) -> np.ndarray:
        n = self.num_households
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

    @staticmethod
    def _to_float(v: Any, default: float) -> float:
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def step(self, dt: float) -> None:
        irr = np.clip(
            self._to_float(self.inputs.get("irradiance_factor"), self.default_irradiance),
            0.0,
            1.0,
        )
        cap = self._to_per_house(self.inputs.get("pv_capacity_kw"), 0.0)
        power = irr * cap * self.performance_ratio

        self.state["pv_power"] = power
        self.state["total_pv"] = float(np.sum(power))
        cap_sum = float(np.sum(cap))
        self.state["avg_pv_factor"] = (
            float(self.state["total_pv"] / cap_sum) if cap_sum > 1e-9 else 0.0
        )

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "pv_power": self.state["pv_power"].copy(),
            "total_pv": self.state["total_pv"],
            "avg_pv_factor": self.state["avg_pv_factor"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["irradiance_factor", "pv_capacity_kw"],
            "outputs": ["pv_power", "total_pv", "avg_pv_factor"],
        }
