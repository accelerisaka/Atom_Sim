"""
GR1: GridTransformer — 配电变压器汇总与过载状态机。
物理规律：节点功率守恒  P_total = Σ P_i
        过载比 ρ = P_total / P_capacity
Natural DT = 0.5s

设计原则
--------
- 输入只接受 ndarray[N] 形式的"逐户负荷"，求和与单位换算都在本原子内部完成
  （这是本领域内的纯计算，不是跨原子的格式转换）。
- 不假设上游 N 的取值，按输入实际长度求和，但用初始化时的 num_households 用于
  缺省兜底数组的形状。
- 当过载比阈值首次穿越 1.0 时，记录 overload_event_pending，由外层桥接器拉走
  并以 EVENT 策略发射给 MK1，实现"瞬间事件 + 慢速平均"的双通道。

端口契约 (Input Port Schema)
----------------------------
- household_loads : ndarray[N] float, 单位 kW

端口契约 (Output Port Schema)
----------------------------
- total_load          : float, 单位 kW
- overload_ratio      : float, 无量纲
- overloaded          : bool
- capacity            : float, 单位 kW
- overload_pulse_ratio: float — 仅在过载触发瞬间为当前 ratio，其余为 0.0
                                （供 EVENT 通道载荷使用）
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class GridTransformer(AtomicSimulator):
    """变压器汇总 + 过载状态仿真器。"""

    def __init__(
        self,
        sim_id: str = "GR1",
        capacity_kw: float = 38.0,
        num_households: int = 24,
        overload_event_threshold: float = 1.0,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.capacity = float(capacity_kw)
        self.num_households = num_households
        self.overload_threshold = overload_event_threshold

        self.state["total_load"] = 0.0
        self.state["overload_ratio"] = 0.0
        self.state["overloaded"] = False
        self.state["overload_pulse_ratio"] = 0.0
        # 用于事件桥外部访问，记录"上一步是否处于过载"
        self.state["was_overloaded"] = False

    def step(self, dt: float) -> None:
        loads = self.inputs.get("household_loads")
        if loads is None:
            arr = np.zeros(self.num_households, dtype=np.float64)
        else:
            arr = np.asarray(loads, dtype=np.float64).ravel()

        total = float(np.sum(arr))
        ratio = total / max(self.capacity, 1e-9)
        is_over = ratio > self.overload_threshold

        # 边沿检测：仅在 False -> True 触发瞬间脉冲
        edge_trigger = (not self.state["was_overloaded"]) and is_over
        self.state["overload_pulse_ratio"] = float(ratio if edge_trigger else 0.0)

        self.state["total_load"] = total
        self.state["overload_ratio"] = float(ratio)
        self.state["overloaded"] = bool(is_over)
        self.state["was_overloaded"] = bool(is_over)

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "total_load": self.state["total_load"],
            "overload_ratio": self.state["overload_ratio"],
            "overloaded": self.state["overloaded"],
            "capacity": self.capacity,
            "overload_pulse_ratio": self.state["overload_pulse_ratio"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["household_loads"],
            "outputs": [
                "total_load",
                "overload_ratio",
                "overloaded",
                "capacity",
                "overload_pulse_ratio",
            ],
        }
