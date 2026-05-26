"""
CF1: CrushStressField — 人群挤压应力场仿真。
物理规律：高密度区域产生超过人体承受阈值的接触挤压应力 (Crush Force)。
实现：由宏观密度场经非线性本构映射为二维挤压应力场。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class CrushStressField(AtomicSimulator):
    """由人群密度计算局部挤压应力场的仿真器。"""

    def __init__(
        self,
        sim_id: str = "CF1",
        grid_size: Tuple[int, int] = (50, 50),
        safe_density: float = 3.0,
        critical_density: float = 6.0,
        max_stress: float = 1.0,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.grid_size = grid_size
        self.safe_density = safe_density
        self.critical_density = critical_density
        self.max_stress = max_stress

        self.state["crush_stress_field"] = np.zeros(grid_size, dtype=np.float64)
        self.state["peak_stress"] = 0.0

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        density = self.inputs.get("density_map")
        if density is None:
            self.current_time += dt
            return

        rho = np.asarray(density, dtype=np.float64)
        if rho.shape != self.grid_size:
            self.current_time += dt
            return

        # 超过安全密度后应力急剧上升（指数型拥挤本构）
        excess = np.maximum(rho - self.safe_density, 0.0)
        span = max(self.critical_density - self.safe_density, 0.1)
        normalized = excess / span
        stress = self.max_stress * (1.0 - np.exp(-2.5 * normalized ** 2))

        # 极高密度区（> critical）额外加成
        super_critical = rho > self.critical_density
        stress[super_critical] = np.minimum(
            stress[super_critical] + 0.25 * (rho[super_critical] - self.critical_density),
            self.max_stress,
        )

        self.state["crush_stress_field"] = np.clip(stress, 0.0, self.max_stress)
        self.state["peak_stress"] = float(np.max(stress))
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "crush_stress_field": self.state["crush_stress_field"].copy(),
            "peak_stress": self.state["peak_stress"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["density_map"],
            "outputs": ["crush_stress_field", "peak_stress"],
        }
