"""
HU2: HouseholdWinterProfile — 冬季家庭用户档案与位置布局。
物理意义：寒潮场景下的逐户静态档案，包含取暖设定、EV 拥有与需求响应偏好。
Natural DT = 1.0s

端口契约 (Output Port Schema)
----------------------------
- positions          : ndarray[N, 2] float — 世界坐标 (m)
- price_sensitivity  : ndarray[N] float in [0, 1]
- heating_setpoint   : ndarray[N] float, °C — 室内舒适取暖目标温度
- heating_priority   : ndarray[N] float in [0, 1] — 取暖不可削减优先级 (1=绝不降)
- has_ev             : ndarray[N] bool — 是否拥有新能源车
- ev_flexibility     : ndarray[N] float in [0, 1] — EV 充电可中断灵活度
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class HouseholdWinterProfile(AtomicSimulator):
    """冬季家庭档案/位置 静态仿真器。"""

    def __init__(
        self,
        sim_id: str = "HU2",
        num_households: int = 24,
        world_size: Tuple[float, float] = (30.0, 30.0),
        sensitivity_range: Tuple[float, float] = (0.2, 1.0),
        heating_setpoint_range: Tuple[float, float] = (20.0, 24.0),
        heating_priority_range: Tuple[float, float] = (0.55, 1.0),
        ev_ownership_rate: float = 0.65,
        ev_flexibility_range: Tuple[float, float] = (0.3, 1.0),
        seed: int = 2027,
    ):
        super().__init__(sim_id, natural_dt=1.0)
        self.num_households = num_households
        self.world_size = world_size

        rng = np.random.default_rng(seed)
        margin = 1.0
        positions = rng.uniform(
            low=[margin, margin],
            high=[world_size[0] - margin, world_size[1] - margin],
            size=(num_households, 2),
        )
        sensitivity = rng.uniform(
            sensitivity_range[0], sensitivity_range[1], size=num_households
        )
        heating_sp = rng.uniform(
            heating_setpoint_range[0], heating_setpoint_range[1], size=num_households
        )
        priority = rng.uniform(
            heating_priority_range[0], heating_priority_range[1], size=num_households
        )
        has_ev = rng.random(num_households) < ev_ownership_rate
        ev_flex = rng.uniform(
            ev_flexibility_range[0], ev_flexibility_range[1], size=num_households
        )
        ev_flex = np.where(has_ev, ev_flex, 0.0)

        self.state["positions"] = positions
        self.state["price_sensitivity"] = sensitivity
        self.state["heating_setpoint"] = heating_sp
        self.state["heating_priority"] = priority
        self.state["has_ev"] = has_ev
        self.state["ev_flexibility"] = ev_flex

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "positions": self.state["positions"].copy(),
            "price_sensitivity": self.state["price_sensitivity"].copy(),
            "heating_setpoint": self.state["heating_setpoint"].copy(),
            "heating_priority": self.state["heating_priority"].copy(),
            "has_ev": self.state["has_ev"].copy(),
            "ev_flexibility": self.state["ev_flexibility"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [],
            "outputs": [
                "positions",
                "price_sensitivity",
                "heating_setpoint",
                "heating_priority",
                "has_ev",
                "ev_flexibility",
            ],
        }
