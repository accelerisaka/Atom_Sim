"""
HB1: HomeBatteryStorage — 逐户家用储能充放电与 SOC 动力学。
物理规律：
  白天：光伏盈余优先充电；满电后余电上网（负向净负荷贡献）
  晚间：按放电指令向电网售电或仅供自用
Natural DT = 0.5s

端口契约 (Input Port Schema)
----------------------------
- pv_power              : ndarray[N] float, kW
- base_load             : ndarray[N] float, kW
- battery_capacity_kwh  : ndarray[N] float
- battery_max_power_kw  : ndarray[N] float
- has_battery           : ndarray[N] bool
- discharge_fraction    : ndarray[N] float in [0, 1] — 向电网放电意愿
- self_use_bias         : ndarray[N] float in [0, 1] — 自用优先（抑制上网售电）

端口契约 (Output Port Schema)
----------------------------
- battery_soc       : ndarray[N] float in [0, 1]
- battery_net_kw    : ndarray[N] float — 对电网侧的净贡献 (正=从网取电, 负=向网送电)
- grid_export_kw    : ndarray[N] float — 仅光伏直送上网部分
- grid_discharge_kw : ndarray[N] float — 电池向电网放电
- grid_charge_kw    : ndarray[N] float — 从电网充电（预留，本场景通常为 0）
- total_export      : float, kW
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class HomeBatteryStorage(AtomicSimulator):
    """家用储能 SOC 与充放电仿真器。"""

    def __init__(
        self,
        sim_id: str = "HB1",
        num_households: int = 24,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.93,
        min_soc: float = 0.08,
        default_discharge_fraction: float = 0.0,
        default_self_use_bias: float = 0.7,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_households = num_households
        self.eta_c = float(charge_efficiency)
        self.eta_d = float(discharge_efficiency)
        self.min_soc = float(min_soc)
        self._default_discharge = float(default_discharge_fraction)
        self._default_self_use = float(default_self_use_bias)

        self.state["battery_soc"] = np.full(num_households, 0.35, dtype=np.float64)
        self.state["battery_net_kw"] = np.zeros(num_households, dtype=np.float64)
        self.state["grid_export_kw"] = np.zeros(num_households, dtype=np.float64)
        self.state["grid_discharge_kw"] = np.zeros(num_households, dtype=np.float64)
        self.state["grid_charge_kw"] = np.zeros(num_households, dtype=np.float64)
        self.state["total_export"] = 0.0

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

    def step(self, dt: float) -> None:
        dt_eff = max(float(dt), 1e-9)

        pv = self._to_per_house(self.inputs.get("pv_power"), 0.0)
        base = self._to_per_house(self.inputs.get("base_load"), 0.0)
        cap_kwh = self._to_per_house(self.inputs.get("battery_capacity_kwh"), 0.0)
        max_pwr = self._to_per_house(self.inputs.get("battery_max_power_kw"), 0.0)
        has_bat = self._to_per_house_bool(self.inputs.get("has_battery"), False)
        discharge_frac = np.clip(
            self._to_per_house(
                self.inputs.get("discharge_fraction"), self._default_discharge
            ),
            0.0,
            1.0,
        )
        self_use = np.clip(
            self._to_per_house(self.inputs.get("self_use_bias"), self._default_self_use),
            0.0,
            1.0,
        )

        soc = self.state["battery_soc"].copy()
        net = np.zeros(self.num_households, dtype=np.float64)
        export_pv = np.zeros(self.num_households, dtype=np.float64)
        discharge_grid = np.zeros(self.num_households, dtype=np.float64)
        charge_grid = np.zeros(self.num_households, dtype=np.float64)

        surplus = np.maximum(pv - base, 0.0)
        deficit = np.maximum(base - pv, 0.0)

        for i in range(self.num_households):
            if not has_bat[i] or cap_kwh[i] < 1e-6:
                export_pv[i] = surplus[i]
                net[i] = deficit[i] - surplus[i]
                soc[i] = 0.0
                continue

            headroom_kwh = max(0.0, (1.0 - soc[i]) * cap_kwh[i])
            charge_pwr = min(
                surplus[i],
                max_pwr[i],
                headroom_kwh / dt_eff / max(self.eta_c, 1e-6),
            )
            soc[i] += charge_pwr * dt_eff * self.eta_c / cap_kwh[i]
            remain_surplus = surplus[i] - charge_pwr

            avail_kwh = max(0.0, (soc[i] - self.min_soc) * cap_kwh[i])
            avail_pwr = min(
                max_pwr[i],
                avail_kwh * self.eta_d / dt_eff,
            )

            discharge_self = min(deficit[i], avail_pwr)
            soc[i] -= discharge_self * dt_eff / (cap_kwh[i] * max(self.eta_d, 1e-6))

            remain_pwr = max(0.0, avail_pwr - discharge_self)
            sell_factor = discharge_frac[i] * (1.0 - 0.75 * self_use[i])
            discharge_grid[i] = remain_pwr * sell_factor
            soc[i] -= (
                discharge_grid[i]
                * dt_eff
                / (cap_kwh[i] * max(self.eta_d, 1e-6))
            )

            export_pv[i] = remain_surplus
            # 正=从电网取电，负=向电网送电（内部光伏充电不计入 net）
            net[i] = deficit[i] - discharge_self - discharge_grid[i] - remain_surplus

            soc[i] = float(np.clip(soc[i], 0.0, 1.0))

        self.state["battery_soc"] = soc
        self.state["battery_net_kw"] = net
        self.state["grid_export_kw"] = export_pv + discharge_grid
        self.state["grid_discharge_kw"] = discharge_grid
        self.state["grid_charge_kw"] = charge_grid
        self.state["total_export"] = float(
            np.sum(np.maximum(-net, 0.0)) + np.sum(export_pv)
        )

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "battery_soc": self.state["battery_soc"].copy(),
            "battery_net_kw": self.state["battery_net_kw"].copy(),
            "grid_export_kw": self.state["grid_export_kw"].copy(),
            "grid_discharge_kw": self.state["grid_discharge_kw"].copy(),
            "grid_charge_kw": self.state["grid_charge_kw"].copy(),
            "total_export": self.state["total_export"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [
                "pv_power",
                "base_load",
                "battery_capacity_kwh",
                "battery_max_power_kw",
                "has_battery",
                "discharge_fraction",
                "self_use_bias",
            ],
            "outputs": [
                "battery_soc",
                "battery_net_kw",
                "grid_export_kw",
                "grid_discharge_kw",
                "grid_charge_kw",
                "total_export",
            ],
        }
