"""
FL5: HeavyGasDiffusion2D — 比空气重的有毒气体地面蔓延仿真。
物理规律：菲克扩散 + 重力沉降（向下富集）+ 局部排风抽取。
实现：二维有限差分，重气体沿地面累积并向低处蔓延。
Natural DT = 0.1s

端口契约 (Input Port Schema)
----------------------------
- gas_source_rate   : ndarray[H, W] float, 单位 1/s — 每格泄漏源强度
- ventilation_field : ndarray[H, W] float, 单位 1/s — 每格排风抽取速率（由 VT1 提供）

端口契约 (Output Port Schema)
-----------------------------
- gas_concentration_field : ndarray[H, W] float in [0, 1]
- gas_hazard_field        : ndarray[H, W] float in [0, 1] — 毒性危害指数
- visibility_field        : ndarray[H, W] float in [0, 1] — 1=完全可见
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class HeavyGasDiffusion2D(AtomicSimulator):
    """比空气重的有毒气体二维扩散与地面富集仿真器。"""

    def __init__(
        self,
        sim_id: str = "FL5",
        grid_size: Tuple[int, int] = (50, 50),
        cell_size: float = 0.5,
        diffusion_coeff: float = 0.04,
        sedimentation_coeff: float = 0.35,
        floor_enhance_rows: int = 6,
        floor_diffusion_boost: float = 1.8,
        decay_rate: float = 0.0005,
        hazard_gain: float = 2.2,
        visibility_gain: float = 2.8,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.D = diffusion_coeff
        self.sedimentation_coeff = sedimentation_coeff
        self.floor_enhance_rows = floor_enhance_rows
        self.floor_diffusion_boost = floor_diffusion_boost
        self.decay_rate = decay_rate
        self.hazard_gain = hazard_gain
        self.visibility_gain = visibility_gain

        H, W = grid_size
        self.state["gas_concentration_field"] = np.zeros((H, W), dtype=np.float64)
        self.state["gas_hazard_field"] = np.zeros((H, W), dtype=np.float64)
        self.state["visibility_field"] = np.ones((H, W), dtype=np.float64)
        self.state["passable_mask"] = np.ones((H, W), dtype=bool)

    def set_obstacles(self, mask: np.ndarray) -> None:
        """设置障碍物掩码 (True=可通行, False=墙壁)。"""
        self.state["passable_mask"] = mask.astype(bool)

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        C = self.state["gas_concentration_field"].copy()
        mask = self.state["passable_mask"]
        H, W = C.shape

        source_rate = self.inputs.get("gas_source_rate")
        if source_rate is not None:
            rate = np.asarray(source_rate, dtype=np.float64)
            if rate.shape == C.shape:
                C = C + rate * dt
            elif rate.ndim == 0:
                C = C + float(rate) * dt

        # 有限差分扩散（近地面层增强水平蔓延）
        D_local = np.full_like(C, self.D)
        if self.floor_enhance_rows > 0:
            floor_start = max(0, H - self.floor_enhance_rows)
            D_local[floor_start:, :] *= self.floor_diffusion_boost

        r_max = np.max(D_local * dt / (self.cell_size ** 2))
        r_eff = min(float(r_max), 0.24) if r_max > 0 else 0.0
        scale = r_eff / r_max if r_max > 1e-12 else 0.0

        laplacian = np.zeros_like(C)
        laplacian[1:-1, 1:-1] = (
            C[2:, 1:-1] + C[:-2, 1:-1]
            + C[1:-1, 2:] + C[1:-1, :-2]
            - 4.0 * C[1:-1, 1:-1]
        )
        laplacian *= mask
        C = C + scale * laplacian

        # 重力沉降：气体从高处（小 row）向低处（大 row）转移
        sed = self.sedimentation_coeff * C * mask
        C = C - sed * dt
        C[1:, :] = C[1:, :] + sed[:-1, :] * dt

        # 局部智能排风抽取
        vent_field = self.inputs.get("ventilation_field")
        if vent_field is not None:
            vf = np.asarray(vent_field, dtype=np.float64)
            if vf.shape == C.shape:
                C = C * np.clip(1.0 - vf * dt, 0.0, 1.0)
            elif vf.ndim == 0 and float(vf) > 0:
                C = C * (1.0 - float(vf) * dt)

        C = C * (1.0 - self.decay_rate * dt)
        C = np.clip(C, 0.0, 1.0)
        C[~mask] = 0.0

        hazard = np.clip(1.0 - np.exp(-self.hazard_gain * C), 0.0, 1.0)
        visibility = np.clip(1.0 - self.visibility_gain * C, 0.0, 1.0)

        self.state["gas_concentration_field"] = C
        self.state["gas_hazard_field"] = hazard
        self.state["visibility_field"] = visibility
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "gas_concentration_field": self.state["gas_concentration_field"].copy(),
            "gas_hazard_field": self.state["gas_hazard_field"].copy(),
            "visibility_field": self.state["visibility_field"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["gas_source_rate", "ventilation_field"],
            "outputs": [
                "gas_concentration_field",
                "gas_hazard_field",
                "visibility_field",
            ],
        }
