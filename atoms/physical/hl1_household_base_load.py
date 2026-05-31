"""
HL1: HouseholdBaseLoad — 逐户居民基础用电负荷（含早晚高峰）。
物理规律：日内双峰负荷曲线 × 逐户缩放系数
Natural DT = 0.5s

端口契约 (Input Port Schema)
----------------------------
- base_load_scale : ndarray[N] float

端口契约 (Output Port Schema)
----------------------------
- base_load   : ndarray[N] float, kW
- total_base  : float, kW
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class HouseholdBaseLoad(AtomicSimulator):
    """逐户基础用电负荷仿真器。"""

    def __init__(
        self,
        sim_id: str = "HL1",
        num_households: int = 24,
        base_kw: float = 0.45,
        morning_peak_kw: float = 0.35,
        evening_peak_kw: float = 1.1,
        morning_center: float = 22.0,
        evening_center: float = 108.0,
        peak_width: float = 12.0,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_households = num_households
        self.base_kw = float(base_kw)
        self.morning_peak_kw = float(morning_peak_kw)
        self.evening_peak_kw = float(evening_peak_kw)
        self.morning_center = float(morning_center)
        self.evening_center = float(evening_center)
        self.peak_width = max(float(peak_width), 1e-6)

        self.state["base_load"] = np.full(num_households, self.base_kw, dtype=np.float64)
        self.state["total_base"] = float(num_households * self.base_kw)

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

    def _gaussian_bump(self, t: float, center: float, amplitude: float) -> float:
        x = (t - center) / self.peak_width
        return amplitude * float(np.exp(-0.5 * x * x))

    def step(self, dt: float) -> None:
        scale = self._to_per_house(self.inputs.get("base_load_scale"), 1.0)
        t = self.current_time
        profile = (
            self.base_kw
            + self._gaussian_bump(t, self.morning_center, self.morning_peak_kw)
            + self._gaussian_bump(t, self.evening_center, self.evening_peak_kw)
        )
        load = scale * profile

        self.state["base_load"] = load
        self.state["total_base"] = float(np.sum(load))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "base_load": self.state["base_load"].copy(),
            "total_base": self.state["total_base"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["base_load_scale"],
            "outputs": ["base_load", "total_base"],
        }
