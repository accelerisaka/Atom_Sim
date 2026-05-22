"""
PD7: PriceSensitivityAppraisal — 价格-舒适权衡的逐户设定温度调整。
理论依据：行为经济学的"参考点-效用权衡"模型 + 逐户异质性偏好。
核心思想：
    每户感受到电价高于参考价时，按其个人价格敏感度产生"上调设定温度"的意愿，
    从而牺牲一定舒适度换取电费下降，形成对电网过载的需求侧响应。
Natural DT = 2.0s

设计原则
--------
本原子绝对不假设来源/形状：所有数组输入都用 _to_per_house() 兜底成长度 N。
- 价格敏感度与舒适基准来自 HU1（identity 直通），非常稳定；
- 当前电价是 MK1 输出的标量；
- local_temp 仅作"舒适缺口"参考，不强制要求上游必须提供。

端口契约 (Input Port Schema)
----------------------------
- price_sensitivity : ndarray[N] float in [0, 1]
- base_setpoint     : ndarray[N] float, °C
- current_price     : float, yuan/kWh
- local_temp        : ndarray[N] float, °C (可选；用于刹车机制：若已经远高于设定，
                                         不再继续上调)

端口契约 (Output Port Schema)
----------------------------
- setpoint        : ndarray[N] float, °C
- setpoint_uplift : ndarray[N] float, °C  (= setpoint - base_setpoint)
- avg_uplift      : float — 全场平均上调，便于可视化
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class PriceSensitivityAppraisal(AtomicSimulator):
    """逐户价格-舒适权衡仿真器。"""

    def __init__(
        self,
        sim_id: str = "PD7",
        num_households: int = 24,
        reference_price: float = 0.55,    # yuan/kWh, 参考价（无价格压力时）
        max_uplift: float = 4.0,          # °C 最大上调幅度
        elasticity: float = 2.5,          # 价格弹性系数
        inertia: float = 0.55,            # 时间惯性 (越大越慢响应)
        comfort_brake_temp: float = 32.0, # 当 local_temp 高于此值时减弱上调（避免热射病）
    ):
        super().__init__(sim_id, natural_dt=2.0)
        self.num_households = num_households
        self.ref_price = float(reference_price)
        self.max_uplift = float(max_uplift)
        self.elasticity = float(elasticity)
        self.inertia = float(inertia)
        self.brake_temp = float(comfort_brake_temp)

        self.state["setpoint_uplift"] = np.zeros(num_households, dtype=np.float64)
        self.state["setpoint"] = np.full(num_households, 26.0, dtype=np.float64)
        self.state["avg_uplift"] = 0.0

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 主步进
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        sensitivity = np.clip(
            self._to_per_house(self.inputs.get("price_sensitivity"), 0.5),
            0.0,
            1.0,
        )
        base_sp = self._to_per_house(self.inputs.get("base_setpoint"), 26.0)
        local_temp = self._to_per_house(self.inputs.get("local_temp"), 30.0)
        price = self._to_float(self.inputs.get("current_price"), self.ref_price)

        # 价格相对涨幅 (>=0)
        premium = max(0.0, price / max(self.ref_price, 1e-9) - 1.0)

        # 目标上调 = 个体敏感度 × 价格弹性 × 涨幅
        target_uplift = sensitivity * self.elasticity * premium

        # 舒适刹车：若环境温度过高，降低上调意愿（避免极端不舒适）
        excess_heat = np.maximum(local_temp - self.brake_temp, 0.0)
        brake_factor = 1.0 / (1.0 + 0.4 * excess_heat)
        target_uplift = target_uplift * brake_factor

        target_uplift = np.clip(target_uplift, 0.0, self.max_uplift)

        # 时间惯性平滑
        prev = self.state["setpoint_uplift"]
        new_uplift = self.inertia * prev + (1.0 - self.inertia) * target_uplift

        self.state["setpoint_uplift"] = new_uplift
        self.state["setpoint"] = base_sp + new_uplift
        self.state["avg_uplift"] = float(np.mean(new_uplift))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "setpoint": self.state["setpoint"].copy(),
            "setpoint_uplift": self.state["setpoint_uplift"].copy(),
            "avg_uplift": self.state["avg_uplift"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [
                "price_sensitivity",
                "base_setpoint",
                "current_price",
                "local_temp",
            ],
            "outputs": ["setpoint", "setpoint_uplift", "avg_uplift"],
        }
