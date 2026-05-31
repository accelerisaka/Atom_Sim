"""
SR1: SolarIrradiance — 社区尺度太阳辐照度曲线。
物理规律：高斯型日照曲线，峰值出现在 day_peak_time，夜间为零。
Natural DT = 1.0s

端口契约 (Output Port Schema)
----------------------------
- irradiance_factor : float in [0, 1] — 归一化辐照度
- is_daylight       : bool
"""

from __future__ import annotations

import math
from typing import Any, Dict

from core.base import AtomicSimulator


class SolarIrradiance(AtomicSimulator):
    """太阳辐照度（标量曲线）仿真器。"""

    def __init__(
        self,
        sim_id: str = "SR1",
        day_start: float = 6.0,
        day_end: float = 78.0,
        day_peak_time: float = 42.0,
        peak_width: float = 18.0,
        cloud_noise_amp: float = 0.08,
        seed: int = 2031,
    ):
        super().__init__(sim_id, natural_dt=1.0)
        self.day_start = float(day_start)
        self.day_end = float(day_end)
        self.day_peak = float(day_peak_time)
        self.peak_width = max(float(peak_width), 1e-6)
        self.cloud_amp = float(cloud_noise_amp)
        self._rng_phase = (seed % 1000) * 0.017

        self.state["irradiance_factor"] = 0.0
        self.state["is_daylight"] = False

    def _diurnal_factor(self, t: float) -> float:
        if t < self.day_start or t >= self.day_end:
            return 0.0
        x = (t - self.day_peak) / self.peak_width
        base = math.exp(-0.5 * x * x)
        ripple = 1.0 + self.cloud_amp * math.sin(
            0.35 * t + self._rng_phase
        )
        return max(0.0, min(1.0, base * ripple))

    def step(self, dt: float) -> None:
        factor = self._diurnal_factor(self.current_time)
        self.state["irradiance_factor"] = factor
        self.state["is_daylight"] = factor > 0.02

        if dt > 0:
            self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "irradiance_factor": self.state["irradiance_factor"],
            "is_daylight": self.state["is_daylight"],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [],
            "outputs": ["irradiance_factor", "is_daylight"],
        }
