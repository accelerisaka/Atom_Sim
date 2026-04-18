"""
TD1: HeatConduction — 热传导与火焰仿真。
物理规律：傅里叶导热定律  ∂T/∂t = α ∇²T
实现：二维有限差分网格 (Finite Difference)。
Natural DT = 0.5s
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from core.base import AtomicSimulator


class HeatConduction(AtomicSimulator):
    """二维热传导有限差分仿真器。"""

    def __init__(
        self,
        sim_id: str = "TD1",
        grid_size: Tuple[int, int] = (50, 50),
        cell_size: float = 0.5,
        alpha: float = 0.02,
        ambient_temp: float = 20.0,
    ):
        super().__init__(sim_id, natural_dt=0.5)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.alpha = alpha  # 热扩散系数 m²/s
        self.ambient_temp = ambient_temp

        self.state["temperature_field"] = np.full(grid_size, ambient_temp, dtype=np.float64)
        self.state["heat_flux"] = np.zeros(grid_size, dtype=np.float64)
        self.state["fire_sources"] = []  # [(row, col, intensity), ...]

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        T = self.state["temperature_field"]
        nx, ny = self.grid_size

        ignition = self.inputs.get("ignition_points", [])
        for pt in ignition:
            r, c = int(pt[0]), int(pt[1])
            intensity = pt[2] if len(pt) > 2 else 800.0
            if 0 <= r < nx and 0 <= c < ny:
                T[r, c] = max(T[r, c], intensity)
                if (r, c, intensity) not in self.state["fire_sources"]:
                    self.state["fire_sources"].append((r, c, intensity))

        for r, c, intensity in self.state["fire_sources"]:
            T[r, c] = max(T[r, c], intensity)

        # 有限差分 (显式欧拉)
        r_coeff = self.alpha * dt / (self.cell_size ** 2)
        r_coeff = min(r_coeff, 0.24)  # CFL 稳定性限制

        laplacian = np.zeros_like(T)
        laplacian[1:-1, 1:-1] = (
            T[2:, 1:-1] + T[:-2, 1:-1] +
            T[1:-1, 2:] + T[1:-1, :-2] -
            4.0 * T[1:-1, 1:-1]
        )

        T_new = T + r_coeff * laplacian

        # 边界：Neumann (绝热)
        T_new[0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]

        # 热通量 = |∇T|
        grad_x = np.gradient(T_new, self.cell_size, axis=0)
        grad_y = np.gradient(T_new, self.cell_size, axis=1)
        self.state["heat_flux"] = np.sqrt(grad_x**2 + grad_y**2)

        self.state["temperature_field"] = T_new
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "temperature_field": self.state["temperature_field"].copy(),
            "heat_flux": self.state["heat_flux"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["ignition_points"],
            "outputs": ["temperature_field", "heat_flux"],
        }
