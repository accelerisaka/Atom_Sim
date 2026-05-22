"""
HU1: HouseholdLayout — 家庭用户档案与位置布局。
物理意义：表示分布在城市/社区某一供电区域内的若干用户家庭，
        每户拥有静态的位置、价格敏感度（异质偏好）、舒适基准设定温度。
Natural DT = 1.0s   (本原子内部状态静态，dt 仅用于推进 current_time)

设计原则
--------
本原子是一个"档案/Profile"型仿真器，自身不演化任何动力学，仅在 t=0 用
确定性 RNG 生成一组户型档案，并在 step() 中保持不变。其 outputs 用作
其他原子（PD7、空间采样 transforms 等）的稳定参数源。

端口契约 (Output Port Schema)
----------------------------
- positions          : ndarray[N, 2] float
    每户在世界坐标下的位置（行、列两个分量），与 TD1.cell_size 同单位（米）。
    用于 transform 在 TD1 温度场上做空间采样。
- price_sensitivity  : ndarray[N] float in [0, 1]
    每户对电价上涨的心理敏感度。0=完全不敏感，1=极度敏感。
- base_setpoint      : ndarray[N] float, 单位 °C
    每户在零电价情境下的舒适基准设定温度。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class HouseholdLayout(AtomicSimulator):
    """家庭档案/位置 静态仿真器。"""

    def __init__(
        self,
        sim_id: str = "HU1",
        num_households: int = 24,
        world_size: Tuple[float, float] = (30.0, 30.0),
        sensitivity_range: Tuple[float, float] = (0.15, 1.0),
        setpoint_range: Tuple[float, float] = (23.5, 27.5),
        seed: int = 2026,
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

        base_sp = rng.uniform(
            setpoint_range[0], setpoint_range[1], size=num_households
        )

        self.state["positions"] = positions
        self.state["price_sensitivity"] = sensitivity
        self.state["base_setpoint"] = base_sp

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "positions": self.state["positions"].copy(),
            "price_sensitivity": self.state["price_sensitivity"].copy(),
            "base_setpoint": self.state["base_setpoint"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [],
            "outputs": ["positions", "price_sensitivity", "base_setpoint"],
        }
