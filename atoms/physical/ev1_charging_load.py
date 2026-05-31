"""
EV1: EVChargingLoad — 逐户新能源汽车晚高峰充电负荷。
物理规律：在充电时间窗内按 charge_limit 比例抽取功率
        P_i = has_ev_i · charge_limit_i · P_max_i  (窗内)
Natural DT = 0.5s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class EVChargingLoad(AtomicSimulator):
    """逐户 EV 充电负荷仿真器。"""

    def __init__(
        self,
        sim_id: str = "EV1",
        num_households: int = 24,
        max_charge_kw: float = 7.0,
        evening_start_time: float = 25.0,
        evening_end_time: float = 115.0,
        ramp_seconds: float = 8.0,
        default_charge_limit: float = 1.0,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_households = num_households
        self.max_charge_kw = max_charge_kw
        self.evening_start = evening_start_time
        self.evening_end = evening_end_time
        self.ramp_seconds = ramp_seconds
        self._default_charge_limit = default_charge_limit

        self.state["ev_load"] = np.zeros(num_households, dtype=np.float64)
        self.state["charging"] = np.zeros(num_households, dtype=bool)
        self.state["window_active"] = False
        self.state["n_charging"] = 0

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

    def _to_per_house_bool(self, value: Any, default: bool = False) -> np.ndarray:
        n = self.num_households
        if value is None:
            return np.full(n, default, dtype=bool)
        arr = np.asarray(value, dtype=bool).ravel()
        if arr.size == 0:
            return np.full(n, default, dtype=bool)
        if arr.size == 1:
            return np.full(n, bool(arr[0]), dtype=bool)
        if arr.size < n:
            pad = np.full(n - arr.size, default, dtype=bool)
            return np.concatenate([arr, pad])
        return arr[:n]

    def _window_factor(self, t: float) -> float:
        if t < self.evening_start:
            return 0.0
        if t >= self.evening_end:
            return 0.0
        ramp = max(self.ramp_seconds, 1e-6)
        if t < self.evening_start + ramp:
            return float((t - self.evening_start) / ramp)
        return 1.0

    def step(self, dt: float) -> None:
        has_ev = self._to_per_house_bool(self.inputs.get("has_ev"), False)
        charge_limit = np.clip(
            self._to_per_house(self.inputs.get("charge_limit"), self._default_charge_limit),
            0.0,
            1.0,
        )

        win = self._window_factor(self.current_time)
        self.state["window_active"] = win > 0.01

        base_power = has_ev.astype(np.float64) * charge_limit * self.max_charge_kw * win
        charging = base_power > 0.05

        self.state["ev_load"] = base_power
        self.state["charging"] = charging
        self.state["n_charging"] = int(np.sum(charging))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "ev_load": self.state["ev_load"].copy(),
            "charging": self.state["charging"].copy(),
            "window_active": self.state["window_active"],
            "n_charging": self.state["n_charging"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["has_ev", "charge_limit"],
            "outputs": ["ev_load", "charging", "window_active", "n_charging"],
        }
