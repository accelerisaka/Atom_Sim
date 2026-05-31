"""
PD9: BatteryArbitrageAppraisal — 储能峰谷套利心理决策。
理论依据：期望收益最大化 + 个体套利敏感度异质性。
核心思想：
  电价高于参考价 → 提高向电网放电比例（售电套利）
  电价低于参考价 → 提高自用偏置、抑制上网售电（保留电量自用）
  电池 SOC 过低时自动提升自用偏置，避免停电风险
Natural DT = 2.0s

端口契约 (Input Port Schema)
----------------------------
- arbitrage_sensitivity : ndarray[N] float in [0, 1]
- current_price         : float, yuan/kWh
- battery_soc           : ndarray[N] float in [0, 1] (可选)

端口契约 (Output Port Schema)
----------------------------
- discharge_fraction : ndarray[N] float in [0, 1]
- self_use_bias      : ndarray[N] float in [0, 1]
- avg_discharge_frac : float
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class BatteryArbitrageAppraisal(AtomicSimulator):
    """逐户储能套利心理评估仿真器。"""

    def __init__(
        self,
        sim_id: str = "PD9",
        num_households: int = 24,
        reference_price: float = 0.55,
        peak_elasticity: float = 2.6,
        valley_self_use_boost: float = 1.8,
        inertia: float = 0.5,
        low_soc_brake: float = 0.25,
        evening_discharge_boost: float = 0.35,
        evening_start: float = 90.0,
        evening_end: float = 165.0,
    ):
        super().__init__(sim_id, natural_dt=2.0)
        self.num_households = num_households
        self.ref_price = float(reference_price)
        self.peak_elasticity = float(peak_elasticity)
        self.valley_boost = float(valley_self_use_boost)
        self.inertia = float(inertia)
        self.low_soc_brake = float(low_soc_brake)
        self.evening_boost = float(evening_discharge_boost)
        self.evening_start = float(evening_start)
        self.evening_end = float(evening_end)

        self.state["discharge_fraction"] = np.zeros(num_households, dtype=np.float64)
        self.state["self_use_bias"] = np.full(num_households, 0.65, dtype=np.float64)
        self.state["avg_discharge_frac"] = 0.0

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

    def _evening_factor(self, t: float) -> float:
        if t < self.evening_start or t >= self.evening_end:
            return 0.0
        mid = 0.5 * (self.evening_start + self.evening_end)
        width = max(0.5 * (self.evening_end - self.evening_start), 1e-6)
        x = (t - mid) / width
        return float(np.exp(-0.5 * x * x))

    def step(self, dt: float) -> None:
        sensitivity = np.clip(
            self._to_per_house(self.inputs.get("arbitrage_sensitivity"), 0.5),
            0.0,
            1.0,
        )
        soc = np.clip(
            self._to_per_house(self.inputs.get("battery_soc"), 0.5),
            0.0,
            1.0,
        )
        price = self._to_float(self.inputs.get("current_price"), self.ref_price)

        premium = max(0.0, price / max(self.ref_price, 1e-9) - 1.0)
        discount = max(0.0, 1.0 - price / max(self.ref_price, 1e-9))

        evening = self._evening_factor(self.current_time)
        target_discharge = (
            sensitivity * self.peak_elasticity * premium
            + self.evening_boost * evening
        )
        target_discharge = np.clip(target_discharge, 0.0, 1.0)

        target_self_use = 0.55 + self.valley_boost * sensitivity * discount
        low_soc_mask = soc < self.low_soc_brake
        target_self_use = np.where(low_soc_mask, np.maximum(target_self_use, 0.92), target_self_use)
        target_self_use = np.clip(target_self_use, 0.0, 1.0)

        prev_d = self.state["discharge_fraction"]
        prev_s = self.state["self_use_bias"]
        new_d = self.inertia * prev_d + (1.0 - self.inertia) * target_discharge
        new_s = self.inertia * prev_s + (1.0 - self.inertia) * target_self_use

        self.state["discharge_fraction"] = new_d
        self.state["self_use_bias"] = new_s
        self.state["avg_discharge_frac"] = float(np.mean(new_d))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "discharge_fraction": self.state["discharge_fraction"].copy(),
            "self_use_bias": self.state["self_use_bias"].copy(),
            "avg_discharge_frac": self.state["avg_discharge_frac"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["arbitrage_sensitivity", "current_price", "battery_soc"],
            "outputs": ["discharge_fraction", "self_use_bias", "avg_discharge_frac"],
        }
