"""
FL4: CrowdFluid — 宏观人流密度仿真。
物理规律：连续性方程  ∂ρ/∂t + ∇·(ρv) = 0
实现：将个体位置映射到密度网格。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class CrowdFluid(AtomicSimulator):
    """宏观人流密度计算器。"""

    def __init__(
        self,
        sim_id: str = "FL4",
        grid_size: Tuple[int, int] = (50, 50),
        cell_size: float = 0.5,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.grid_size = grid_size
        self.cell_size = cell_size

        self.state["density_map"] = np.zeros(grid_size, dtype=np.float64)
        # 每个格子的面积
        self._cell_area = cell_size ** 2

    def step(self, dt: float) -> None:
        positions = self.inputs.get("agent_positions")
        if positions is None:
            self.current_time += dt
            return

        density = np.zeros(self.grid_size, dtype=np.float64)
        positions = np.asarray(positions)

        if positions.ndim == 2 and positions.shape[0] > 0:
            # 将世界坐标映射到网格坐标
            grid_coords = (positions / self.cell_size).astype(int)
            nr, nc = self.grid_size

            valid = (
                (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < nr) &
                (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < nc)
            )
            gc = grid_coords[valid]

            # 统计每格人数
            np.add.at(density, (gc[:, 0], gc[:, 1]), 1.0)

            # 转换为密度 (人/m²)
            density /= self._cell_area

        # 平滑（高斯核近似：3x3 均值滤波）
        smoothed = np.zeros_like(density)
        smoothed[1:-1, 1:-1] = (
            density[:-2, :-2] + density[:-2, 1:-1] + density[:-2, 2:] +
            density[1:-1, :-2] + density[1:-1, 1:-1] + density[1:-1, 2:] +
            density[2:, :-2] + density[2:, 1:-1] + density[2:, 2:]
        ) / 9.0

        self.state["density_map"] = smoothed
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "density_map": self.state["density_map"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["agent_positions"],
            "outputs": ["density_map"],
        }
