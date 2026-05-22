"""
FL3: Diffusion2D — 烟雾/CO 扩散仿真。
物理规律：菲克扩散定律  ∂C/∂t = D ∇²C + S
实现：二维有限差分，受障碍物（墙壁）影响。
Natural DT = 0.2s

端口契约 (Input Port Schema)
----------------------------
- smoke_source_rate  : ndarray[H, W] float, 单位 1/s
    每格的产烟源强度；**由 S2S 总线上的 transform 从温度场换算而来**。
    本原子不再接受原始温度场，不做任何阈值判断。
- ventilation_status : float — 通风强度（0=关闭, 越大排烟越强）
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class Diffusion2D(AtomicSimulator):
    """二维烟雾扩散仿真器。"""

    def __init__(
        self,
        sim_id: str = "FL3",
        grid_size: Tuple[int, int] = (50, 50),
        cell_size: float = 0.5,
        diffusion_coeff: float = 0.05,
        decay_rate: float = 0.001,
    ):
        super().__init__(sim_id, natural_dt=0.2)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.D = diffusion_coeff
        self.decay_rate = decay_rate

        self.state["smoke_density_field"] = np.zeros(grid_size, dtype=np.float64)
        self.state["visibility_field"] = np.ones(grid_size, dtype=np.float64)
        # 障碍物掩码: True = 可通行
        self.state["passable_mask"] = np.ones(grid_size, dtype=bool)

    def set_obstacles(self, mask: np.ndarray) -> None:
        """设置障碍物掩码 (True=可通行, False=墙壁)。"""
        self.state["passable_mask"] = mask.astype(bool)

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        C = self.state["smoke_density_field"]
        mask = self.state["passable_mask"]

        # source rate 已经是"产烟速率(1/s)"，温度→速率的换算在 transform 中完成
        source_rate = self.inputs.get("smoke_source_rate")
        if source_rate is not None:
            rate = np.asarray(source_rate, dtype=np.float64)
            if rate.shape == C.shape:
                C = C + rate * dt
            elif rate.ndim == 0:
                C = C + float(rate) * dt

        # 有限差分扩散
        r = self.D * dt / (self.cell_size ** 2)
        r = min(r, 0.24)

        laplacian = np.zeros_like(C)
        laplacian[1:-1, 1:-1] = (
            C[2:, 1:-1] + C[:-2, 1:-1] +
            C[1:-1, 2:] + C[1:-1, :-2] -
            4.0 * C[1:-1, 1:-1]
        )
        laplacian *= mask  # 墙壁不扩散

        C_new = C + r * laplacian

        # 通风排烟
        ventilation = self.inputs.get("ventilation_status", 0.0)
        if ventilation > 0:
            C_new *= (1.0 - ventilation * 0.01 * dt)

        # 自然衰减
        C_new *= (1.0 - self.decay_rate * dt)
        C_new = np.clip(C_new, 0.0, 1.0)

        # 墙壁位置浓度清零
        C_new[~mask] = 0.0

        self.state["smoke_density_field"] = C_new

        # 能见度 = 1 - smoke (简化 Beer-Lambert 定律)
        self.state["visibility_field"] = np.clip(1.0 - C_new * 3.0, 0.0, 1.0)

        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "smoke_density_field": self.state["smoke_density_field"].copy(),
            "visibility_field": self.state["visibility_field"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["smoke_source_rate", "ventilation_status"],
            "outputs": ["smoke_density_field", "visibility_field"],
        }
