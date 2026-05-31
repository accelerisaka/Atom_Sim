"""
HU3: HouseholdVPPProfile — 分布式光伏+储能社区家庭档案。
物理意义：逐户静态档案（PV 装机、电池容量、套利偏好、位置布局）。
Natural DT = 1.0s

端口契约 (Output Port Schema)
----------------------------
- positions              : ndarray[N, 2]
- pv_capacity_kw         : ndarray[N] float — 屋顶光伏装机 (kW)
- battery_capacity_kwh   : ndarray[N] float — 电池容量 (kWh)
- battery_max_power_kw   : ndarray[N] float — 充放电功率上限 (kW)
- arbitrage_sensitivity  : ndarray[N] float in [0, 1] — 电价套利积极性
- base_load_scale        : ndarray[N] float — 基础负荷缩放系数
- has_battery            : ndarray[N] bool — 是否安装储能
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class HouseholdVPPProfile(AtomicSimulator):
    """VPP 社区家庭档案/位置 静态仿真器。"""

    def __init__(
        self,
        sim_id: str = "HU3",
        num_households: int = 24,
        world_size: Tuple[float, float] = (30.0, 30.0),
        pv_capacity_range: Tuple[float, float] = (2.0, 8.0),
        battery_capacity_range: Tuple[float, float] = (4.0, 13.5),
        battery_power_range: Tuple[float, float] = (2.0, 5.0),
        arbitrage_sensitivity_range: Tuple[float, float] = (0.2, 1.0),
        base_load_scale_range: Tuple[float, float] = (0.7, 1.3),
        battery_ownership_rate: float = 0.85,
        seed: int = 2030,
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
        pv_cap = rng.uniform(pv_capacity_range[0], pv_capacity_range[1], num_households)
        bat_cap = rng.uniform(
            battery_capacity_range[0], battery_capacity_range[1], num_households
        )
        bat_pwr = rng.uniform(
            battery_power_range[0], battery_power_range[1], num_households
        )
        sensitivity = rng.uniform(
            arbitrage_sensitivity_range[0],
            arbitrage_sensitivity_range[1],
            num_households,
        )
        load_scale = rng.uniform(
            base_load_scale_range[0], base_load_scale_range[1], num_households
        )
        has_battery = rng.random(num_households) < battery_ownership_rate
        bat_cap = np.where(has_battery, bat_cap, 0.0)
        bat_pwr = np.where(has_battery, bat_pwr, 0.0)

        self.state["positions"] = positions
        self.state["pv_capacity_kw"] = pv_cap
        self.state["battery_capacity_kwh"] = bat_cap
        self.state["battery_max_power_kw"] = bat_pwr
        self.state["arbitrage_sensitivity"] = sensitivity
        self.state["base_load_scale"] = load_scale
        self.state["has_battery"] = has_battery

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "positions": self.state["positions"].copy(),
            "pv_capacity_kw": self.state["pv_capacity_kw"].copy(),
            "battery_capacity_kwh": self.state["battery_capacity_kwh"].copy(),
            "battery_max_power_kw": self.state["battery_max_power_kw"].copy(),
            "arbitrage_sensitivity": self.state["arbitrage_sensitivity"].copy(),
            "base_load_scale": self.state["base_load_scale"].copy(),
            "has_battery": self.state["has_battery"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [],
            "outputs": [
                "positions",
                "pv_capacity_kw",
                "battery_capacity_kwh",
                "battery_max_power_kw",
                "arbitrage_sensitivity",
                "base_load_scale",
                "has_battery",
            ],
        }
