"""
EH1: ElectricHeatingLoad — 逐户电暖器/热泵取暖负荷模型。
物理规律：当室内体感温度低于设定值时开启取暖
        P_i = clip( idle + k_heat * max(0, T_set_i - T_local_i),  0, P_max )
Natural DT = 0.5s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class ElectricHeatingLoad(AtomicSimulator):
    """逐户电取暖负荷仿真器。"""

    def __init__(
        self,
        sim_id: str = "EH1",
        num_households: int = 24,
        heating_coeff: float = 0.65,
        max_load_per_house: float = 4.0,
        idle_load: float = 0.08,
        on_threshold: float = 0.4,
        default_local_temp: float = -5.0,
        default_setpoint: float = 22.0,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_households = num_households
        self.k = heating_coeff
        self.max_load = max_load_per_house
        self.idle_load = idle_load
        self.on_threshold = on_threshold
        self._default_local_temp = default_local_temp
        self._default_setpoint = default_setpoint

        self.state["heating_load"] = np.zeros(num_households, dtype=np.float64)
        self.state["heating_on"] = np.zeros(num_households, dtype=bool)
        self.state["avg_heat_demand"] = 0.0

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

    def step(self, dt: float) -> None:
        local_temp = self._to_per_house(
            self.inputs.get("local_temp"), self._default_local_temp
        )
        setpoint = self._to_per_house(
            self.inputs.get("heating_setpoint"), self._default_setpoint
        )

        deficit = setpoint - local_temp
        on_mask = deficit > self.on_threshold

        heat_power = self.k * np.maximum(deficit, 0.0)
        load = np.where(on_mask, self.idle_load + heat_power, 0.0)
        load = np.clip(load, 0.0, self.max_load)

        self.state["heating_load"] = load
        self.state["heating_on"] = on_mask
        self.state["avg_heat_demand"] = float(np.mean(np.maximum(deficit, 0.0)))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "heating_load": self.state["heating_load"].copy(),
            "heating_on": self.state["heating_on"].copy(),
            "avg_heat_demand": self.state["avg_heat_demand"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["local_temp", "heating_setpoint"],
            "outputs": ["heating_load", "heating_on", "avg_heat_demand"],
        }
