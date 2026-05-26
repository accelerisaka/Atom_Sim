"""
EX1: ExplosionBlastWave — 体育场爆炸冲击波/恐慌诱因场仿真。
物理规律：点源冲击以球面波形式向外传播并随距离与时间衰减。
实现：二维网格上的径向扩散 + 指数衰减，输出危害场与烟尘能见度场。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class ExplosionBlastWave(AtomicSimulator):
    """爆炸冲击波与烟尘危害场传播仿真器。"""

    def __init__(
        self,
        sim_id: str = "EX1",
        grid_size: Tuple[int, int] = (50, 50),
        cell_size: float = 0.5,
        wave_speed: float = 8.0,
        decay_rate: float = 0.35,
        peak_intensity: float = 1.0,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.wave_speed = wave_speed
        self.decay_rate = decay_rate
        self.peak_intensity = peak_intensity

        H, W = grid_size
        self.state["blast_hazard_field"] = np.zeros((H, W), dtype=np.float64)
        self.state["dust_visibility_field"] = np.ones((H, W), dtype=np.float64)
        self.state["wave_front_radius"] = 0.0
        self._epicenter_grid: Tuple[int, int] = (H // 2, W // 2)
        self._initial_pulse: float = 0.0

    def set_epicenter_grid(self, row: int, col: int, initial_pulse: float = 1.0) -> None:
        self._epicenter_grid = (int(row), int(col))
        self._initial_pulse = float(initial_pulse)

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        H, W = self.grid_size
        er, ec = self._epicenter_grid
        hazard = np.zeros((H, W), dtype=np.float64)

        # 冲击波前缘半径随时间扩展
        self.state["wave_front_radius"] += self.wave_speed * dt
        radius_m = self.state["wave_front_radius"]

        yy, xx = np.mgrid[0:H, 0:W]
        dist_cells = np.sqrt((yy - er) ** 2 + (xx - ec) ** 2)
        dist_m = dist_cells * self.cell_size

        # 环带冲击：前缘附近能量最高
        band_width = max(self.cell_size * 2.0, 0.5)
        in_band = np.abs(dist_m - radius_m) < band_width
        hazard[in_band] = self.peak_intensity * self._initial_pulse

        # 已波及的区域内残留危害（指数衰减）
        behind = dist_m < radius_m
        if np.any(behind):
            age_factor = np.exp(-self.decay_rate * (radius_m - dist_m[behind]) / max(self.wave_speed, 0.1))
            hazard[behind] = np.maximum(hazard[behind], self.peak_intensity * age_factor * 0.6)

        # 爆心持续高热区（前 3s）
        if self.current_time < 3.0:
            core = dist_cells * self.cell_size < 3.0
            hazard[core] = np.maximum(hazard[core], self.peak_intensity * self._initial_pulse)

        # 外部激励（场景注入的初始脉冲场）
        ext = self.inputs.get("blast_seed_field")
        if ext is not None:
            ext_arr = np.asarray(ext, dtype=np.float64)
            if ext_arr.shape == (H, W):
                hazard = np.maximum(hazard, ext_arr)

        hazard = np.clip(hazard, 0.0, 1.0)
        self.state["blast_hazard_field"] = hazard

        # 烟尘降低能见度
        dust = np.clip(hazard * 0.85 + 0.05, 0.0, 1.0)
        self.state["dust_visibility_field"] = np.clip(1.0 - dust, 0.05, 1.0)

        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "blast_hazard_field": self.state["blast_hazard_field"].copy(),
            "dust_visibility_field": self.state["dust_visibility_field"].copy(),
            "wave_front_radius": float(self.state["wave_front_radius"]),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["blast_seed_field"],
            "outputs": ["blast_hazard_field", "dust_visibility_field", "wave_front_radius"],
        }
