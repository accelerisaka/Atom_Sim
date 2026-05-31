"""
PD8: PeakLoadDemandResponse — 峰时电价下的需求侧响应（优先削减 EV，保护取暖）。
理论依据：价格弹性 + 负荷优先级（取暖 > EV 充电）。
Natural DT = 2.0s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class PeakLoadDemandResponse(AtomicSimulator):
    """逐户峰时需求响应：高价时削减 EV 充电，尽量保持取暖设定。"""

    def __init__(
        self,
        sim_id: str = "PD8",
        num_households: int = 24,
        reference_price: float = 0.55,
        ev_elasticity: float = 2.8,
        heating_trim_elasticity: float = 0.35,
        inertia: float = 0.5,
        cold_brake_temp: float = -2.0,
    ):
        super().__init__(sim_id, natural_dt=2.0)
        self.num_households = num_households
        self.ref_price = float(reference_price)
        self.ev_elasticity = float(ev_elasticity)
        self.heating_trim_elasticity = float(heating_trim_elasticity)
        self.inertia = float(inertia)
        self.cold_brake_temp = float(cold_brake_temp)

        self.state["heating_setpoint"] = np.full(num_households, 22.0, dtype=np.float64)
        self.state["ev_charge_limit"] = np.ones(num_households, dtype=np.float64)
        self.state["avg_ev_limit"] = 1.0
        self.state["avg_heating_trim"] = 0.0

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
        sensitivity = np.clip(
            self._to_per_house(self.inputs.get("price_sensitivity"), 0.5), 0.0, 1.0
        )
        base_sp = self._to_per_house(self.inputs.get("base_heating_setpoint"), 22.0)
        priority = np.clip(
            self._to_per_house(self.inputs.get("heating_priority"), 0.8), 0.0, 1.0
        )
        ev_flex = np.clip(
            self._to_per_house(self.inputs.get("ev_flexibility"), 0.5), 0.0, 1.0
        )
        local_temp = self._to_per_house(self.inputs.get("local_temp"), -5.0)
        price = self._to_float(self.inputs.get("current_price"), self.ref_price)

        premium = max(0.0, price / max(self.ref_price, 1e-9) - 1.0)

        target_ev_limit = 1.0 - sensitivity * ev_flex * self.ev_elasticity * premium
        target_ev_limit = np.clip(target_ev_limit, 0.0, 1.0)

        trim = (
            sensitivity
            * (1.0 - priority)
            * self.heating_trim_elasticity
            * premium
        )
        cold_mask = local_temp < self.cold_brake_temp
        trim = np.where(cold_mask, trim * 0.15, trim)
        target_heating_sp = base_sp - trim

        prev_limit = self.state["ev_charge_limit"]
        new_limit = self.inertia * prev_limit + (1.0 - self.inertia) * target_ev_limit

        self.state["ev_charge_limit"] = new_limit
        self.state["heating_setpoint"] = target_heating_sp
        self.state["avg_ev_limit"] = float(np.mean(new_limit))
        self.state["avg_heating_trim"] = float(np.mean(trim))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "heating_setpoint": self.state["heating_setpoint"].copy(),
            "ev_charge_limit": self.state["ev_charge_limit"].copy(),
            "avg_ev_limit": self.state["avg_ev_limit"],
            "avg_heating_trim": self.state["avg_heating_trim"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [
                "price_sensitivity",
                "base_heating_setpoint",
                "heating_priority",
                "ev_flexibility",
                "current_price",
                "local_temp",
            ],
            "outputs": [
                "heating_setpoint",
                "ev_charge_limit",
                "avg_ev_limit",
                "avg_heating_trim",
            ],
        }
