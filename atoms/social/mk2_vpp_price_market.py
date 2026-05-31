"""
MK2: VPPPriceMarket — 双向动态电价（支持净负荷过低时的"负电价"激励）。
经济规律：
  ρ = P_total / P_capacity （可为负，表示净上网）
  高 ρ → 峰时加价；低 ρ → 谷时/负电价折扣
Natural DT = 2.0s

端口契约 (Input Port Schema)
----------------------------
- net_load_ratio  : float — 净负荷比（可 < 0）
- overload_pulse  : float — 过载边沿脉冲
- export_pulse    : float — 大规模上网边沿脉冲（压低电价）

端口契约 (Output Port Schema)
----------------------------
- electricity_price : float, yuan/kWh
- price_multiplier  : float
- target_multiplier : float
- is_negative_price : bool — 是否处于激励性低价/负电价区间
"""

from __future__ import annotations

from typing import Any, Dict

from core.base import AtomicSimulator


class VPPPriceMarket(AtomicSimulator):
    """虚拟电厂双向电价市场仿真器。"""

    def __init__(
        self,
        sim_id: str = "MK2",
        base_price: float = 0.55,
        max_multiplier: float = 4.0,
        min_multiplier: float = 0.12,
        peak_threshold: float = 0.82,
        valley_threshold: float = 0.22,
        peak_sensitivity: float = 3.0,
        valley_sensitivity: float = 2.2,
        nonlinearity: float = 1.35,
        smoothing: float = 0.32,
        pulse_boost: float = 0.65,
        export_pulse_cut: float = 0.35,
    ):
        super().__init__(sim_id, natural_dt=2.0)
        self.base_price = float(base_price)
        self.max_multiplier = float(max_multiplier)
        self.min_multiplier = float(min_multiplier)
        self.peak_thr = float(peak_threshold)
        self.valley_thr = float(valley_threshold)
        self.gamma_peak = float(peak_sensitivity)
        self.gamma_valley = float(valley_sensitivity)
        self.p = float(nonlinearity)
        self.smoothing = float(smoothing)
        self.pulse_boost = float(pulse_boost)
        self.export_pulse_cut = float(export_pulse_cut)

        self.state["price_multiplier"] = 1.0
        self.state["target_multiplier"] = 1.0
        self.state["electricity_price"] = self.base_price
        self.state["is_negative_price"] = False

    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def step(self, dt: float) -> None:
        rho = self._to_float(self.inputs.get("net_load_ratio"), 0.0)
        pulse = max(0.0, self._to_float(self.inputs.get("overload_pulse"), 0.0))
        export_pulse = max(0.0, self._to_float(self.inputs.get("export_pulse"), 0.0))

        peak_excess = max(0.0, rho - self.peak_thr)
        valley_depth = max(0.0, self.valley_thr - rho)

        target_mult = 1.0 + self.gamma_peak * (peak_excess ** self.p)
        target_mult -= self.gamma_valley * (valley_depth ** self.p)

        if pulse > self.peak_thr:
            target_mult += self.pulse_boost * (pulse - self.peak_thr)
        if export_pulse > 0.0:
            target_mult -= self.export_pulse_cut * export_pulse

        target_mult = max(self.min_multiplier, min(target_mult, self.max_multiplier))
        self.state["target_multiplier"] = target_mult

        cur = self.state["price_multiplier"]
        new_mult = cur + (target_mult - cur) * self.smoothing
        new_mult = max(self.min_multiplier, min(new_mult, self.max_multiplier))

        self.state["price_multiplier"] = new_mult
        self.state["electricity_price"] = self.base_price * new_mult
        self.state["is_negative_price"] = new_mult < 0.45

        self.inputs["overload_pulse"] = 0.0
        self.inputs["export_pulse"] = 0.0

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "electricity_price": self.state["electricity_price"],
            "price_multiplier": self.state["price_multiplier"],
            "target_multiplier": self.state["target_multiplier"],
            "is_negative_price": self.state["is_negative_price"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["net_load_ratio", "overload_pulse", "export_pulse"],
            "outputs": [
                "electricity_price",
                "price_multiplier",
                "target_multiplier",
                "is_negative_price",
            ],
        }
