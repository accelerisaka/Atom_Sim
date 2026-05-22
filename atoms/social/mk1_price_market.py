"""
MK1: PriceMarket — 动态电价市场。
经济规律：基于过载比 ρ 的非线性峰谷电价
        multiplier = clip( 1 + γ · max(0, ρ - ρ_thr)^p ,  1, M_max )
        price = base_price · multiplier
        多步指数平滑以避免抖动
Natural DT = 2.0s

设计原则
--------
本原子是一个"市场机制"型仿真器：将上游的过载信号（标量 [0, +∞)）翻译为
电价标量。不接收任何场/数组类型的输入；标量在 step() 内做平滑/上限处理。

端口契约 (Input Port Schema)
----------------------------
- overload_signal : float — 当前过载比 ρ ∈ [0, +∞)，由 GR1.overload_ratio 经
                            AVG 策略平均后送入。
- overload_pulse  : float — 事件通道送入的"突发过载比"，非零时即时调高目标价。

端口契约 (Output Port Schema)
----------------------------
- electricity_price : float, 单位 yuan/kWh
- price_multiplier  : float, 当前价倍数（相对 base_price）
- target_multiplier : float, 未平滑前的目标倍数（便于 debug/可视化）
"""

from __future__ import annotations

from typing import Any, Dict

from core.base import AtomicSimulator


class PriceMarket(AtomicSimulator):
    """动态电价市场仿真器。"""

    def __init__(
        self,
        sim_id: str = "MK1",
        base_price: float = 0.55,        # yuan/kWh, 平段电价
        max_multiplier: float = 4.0,     # 峰段最高 4x base
        sensitivity: float = 3.0,        # γ
        threshold: float = 0.8,          # ρ_thr：低于此过载比不调价
        nonlinearity: float = 1.4,       # p
        smoothing: float = 0.35,         # 指数平滑因子（每 step 向 target 收敛比例）
        pulse_boost: float = 0.6,        # 事件脉冲对目标倍数的额外注入幅度
    ):
        super().__init__(sim_id, natural_dt=2.0)
        self.base_price = float(base_price)
        self.max_multiplier = float(max_multiplier)
        self.gamma = float(sensitivity)
        self.threshold = float(threshold)
        self.p = float(nonlinearity)
        self.smoothing = float(smoothing)
        self.pulse_boost = float(pulse_boost)

        self.state["price_multiplier"] = 1.0
        self.state["target_multiplier"] = 1.0
        self.state["electricity_price"] = self.base_price

    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def step(self, dt: float) -> None:
        rho = max(0.0, self._to_float(self.inputs.get("overload_signal"), 0.0))
        pulse = max(0.0, self._to_float(self.inputs.get("overload_pulse"), 0.0))

        excess = max(0.0, rho - self.threshold)
        target_mult = 1.0 + self.gamma * (excess ** self.p)

        if pulse > 0.0:
            target_mult += self.pulse_boost * (pulse - self.threshold) if pulse > self.threshold else 0.0

        target_mult = max(1.0, min(target_mult, self.max_multiplier))
        self.state["target_multiplier"] = target_mult

        cur = self.state["price_multiplier"]
        new_mult = cur + (target_mult - cur) * self.smoothing
        new_mult = max(1.0, min(new_mult, self.max_multiplier))

        self.state["price_multiplier"] = new_mult
        self.state["electricity_price"] = self.base_price * new_mult

        # 事件脉冲消费完后清零（避免下次 step 重复使用同一脉冲）
        self.inputs["overload_pulse"] = 0.0

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "electricity_price": self.state["electricity_price"],
            "price_multiplier": self.state["price_multiplier"],
            "target_multiplier": self.state["target_multiplier"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["overload_signal", "overload_pulse"],
            "outputs": [
                "electricity_price",
                "price_multiplier",
                "target_multiplier",
            ],
        }
