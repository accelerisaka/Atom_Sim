"""
AC1: ACLoadDevice — 逐户空调用电负荷模型。
物理规律：温差驱动制冷功率
        P_i = clip( base_idle + k_cool * max(0, T_local_i - T_set_i),  0, P_max )
Natural DT = 0.5s

设计原则
--------
本原子绝对不假设温度/设定温度的来源、形状或单位。
- local_temp / setpoint 都按 ndarray[N] 接收（来自 transforms 在 HU1.positions
  上对 TD1 温度场的采样、以及来自 PD7 的逐户设定温度）；
- 标量/None/不等长输入会被广播兜底成长度 N 的数组。
- 任何"场→逐户"的空间采样均不在此完成，由 core/transforms.py 集中处理。

端口契约 (Input Port Schema)
----------------------------
- local_temp  : ndarray[N] float, 单位 °C
- setpoint    : ndarray[N] float, 单位 °C

端口契约 (Output Port Schema)
----------------------------
- power_load  : ndarray[N] float, 单位 kW
- ac_on       : ndarray[N] bool
- avg_temp_diff : float — 全场用户平均"过冷需求" max(0, T_local - T_set)，便于可视化
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class ACLoadDevice(AtomicSimulator):
    """逐户空调负荷仿真器。"""

    def __init__(
        self,
        sim_id: str = "AC1",
        num_households: int = 24,
        cooling_coeff: float = 0.55,        # kW per °C 温差
        max_load_per_house: float = 3.0,    # kW 单户压缩机峰值
        idle_load: float = 0.15,            # kW 待机/风扇基载
        on_threshold: float = 0.5,          # °C 启动温差阈值
        default_local_temp: float = 30.0,
        default_setpoint: float = 26.0,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.num_households = num_households
        self.k = cooling_coeff
        self.max_load = max_load_per_house
        self.idle_load = idle_load
        self.on_threshold = on_threshold
        self._default_local_temp = default_local_temp
        self._default_setpoint = default_setpoint

        self.state["power_load"] = np.zeros(num_households, dtype=np.float64)
        self.state["ac_on"] = np.zeros(num_households, dtype=bool)
        self.state["avg_temp_diff"] = 0.0

    # ------------------------------------------------------------------
    # 内部工具：把任意形式的输入规整为长度 N 的 float 数组
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

    # ------------------------------------------------------------------
    # 主步进
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        local_temp = self._to_per_house(
            self.inputs.get("local_temp"), self._default_local_temp
        )
        setpoint = self._to_per_house(
            self.inputs.get("setpoint"), self._default_setpoint
        )

        diff = local_temp - setpoint
        on_mask = diff > self.on_threshold

        # 制冷功率：温差线性，叠加待机
        cooling_power = self.k * np.maximum(diff, 0.0)
        load = np.where(on_mask, self.idle_load + cooling_power, 0.0)
        load = np.clip(load, 0.0, self.max_load)

        self.state["power_load"] = load
        self.state["ac_on"] = on_mask
        self.state["avg_temp_diff"] = float(np.mean(np.maximum(diff, 0.0)))

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "power_load": self.state["power_load"].copy(),
            "ac_on": self.state["ac_on"].copy(),
            "avg_temp_diff": self.state["avg_temp_diff"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["local_temp", "setpoint"],
            "outputs": ["power_load", "ac_on", "avg_temp_diff"],
        }
