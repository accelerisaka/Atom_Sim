"""
VT1: IntelligentVentilation — 地铁站智能排风系统。
物理规律：分区浓度监测 → 阈值触发 → 局部抽风加速气体消散。
实现：将场划分为若干控制区，超阈值区域激活对应排风单元。
Natural DT = 0.1s

端口契约 (Input Port Schema)
----------------------------
- gas_concentration_field : ndarray[H, W] float in [0, 1]

端口契约 (Output Port Schema)
-----------------------------
- ventilation_field     : ndarray[H, W] float — 每格排风抽取速率 (1/s)
- active_fan_count      : int — 当前激活的排风分区数
- total_extraction_rate : float — 全场总抽取速率积分
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class IntelligentVentilation(AtomicSimulator):
    """基于局部浓度阈值的智能排风控制仿真器。"""

    def __init__(
        self,
        sim_id: str = "VT1",
        grid_size: Tuple[int, int] = (50, 50),
        zone_rows: int = 5,
        zone_cols: int = 5,
        concentration_threshold: float = 0.12,
        release_threshold_ratio: float = 0.65,
        max_extraction_rate: float = 0.45,
        response_gain: float = 1.2,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.grid_size = grid_size
        self.zone_rows = zone_rows
        self.zone_cols = zone_cols
        self.concentration_threshold = concentration_threshold
        self.release_threshold = concentration_threshold * release_threshold_ratio
        self.max_extraction_rate = max_extraction_rate
        self.response_gain = response_gain

        H, W = grid_size
        n_zones = zone_rows * zone_cols
        self.state["ventilation_field"] = np.zeros((H, W), dtype=np.float64)
        self.state["zone_active"] = np.zeros(n_zones, dtype=bool)
        self.state["active_fan_count"] = 0
        self.state["total_extraction_rate"] = 0.0

        self._zone_row_edges = np.linspace(0, H, zone_rows + 1, dtype=int)
        self._zone_col_edges = np.linspace(0, W, zone_cols + 1, dtype=int)

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        conc = self.inputs.get("gas_concentration_field")
        H, W = self.grid_size
        vent = np.zeros((H, W), dtype=np.float64)
        zone_active = self.state["zone_active"].copy()

        if conc is not None:
            field = np.asarray(conc, dtype=np.float64)
            if field.shape == (H, W):
                zi = 0
                for zr in range(self.zone_rows):
                    r0, r1 = self._zone_row_edges[zr], self._zone_row_edges[zr + 1]
                    for zc in range(self.zone_cols):
                        c0, c1 = self._zone_col_edges[zc], self._zone_col_edges[zc + 1]
                        patch = field[r0:r1, c0:c1]
                        if patch.size == 0:
                            zi += 1
                            continue

                        peak = float(np.max(patch))
                        if peak >= self.concentration_threshold:
                            zone_active[zi] = True
                        elif peak <= self.release_threshold:
                            zone_active[zi] = False

                        if zone_active[zi]:
                            excess = (peak - self.concentration_threshold) / max(
                                1.0 - self.concentration_threshold, 1e-6
                            )
                            rate = self.response_gain * self.max_extraction_rate * np.clip(
                                excess, 0.0, 1.0
                            )
                            vent[r0:r1, c0:c1] = rate
                        zi += 1

        self.state["ventilation_field"] = vent
        self.state["zone_active"] = zone_active
        self.state["active_fan_count"] = int(np.sum(zone_active))
        self.state["total_extraction_rate"] = float(np.sum(vent))
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "ventilation_field": self.state["ventilation_field"].copy(),
            "active_fan_count": int(self.state["active_fan_count"]),
            "total_extraction_rate": float(self.state["total_extraction_rate"]),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["gas_concentration_field"],
            "outputs": [
                "ventilation_field",
                "active_fan_count",
                "total_extraction_rate",
            ],
        }
