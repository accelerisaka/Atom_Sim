"""
P6: BottleneckGate — 瓶颈通行约束仿真。
物理规律：水流模型/容量约束  Q = min(Q_in, C)
实现：当区域密度过大时，强制限制出口通过率。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class BottleneckGate(AtomicSimulator):
    """瓶颈口通行限速仿真器。"""

    def __init__(
        self,
        sim_id: str = "P6",
        capacity: float = 1.5,
        gate_width: float = 1.2,
        grid_size: Tuple[int, int] = (50, 50),
        critical_density: float = 5.0,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.capacity = capacity
        self.gate_width = gate_width
        self.grid_size = grid_size
        self.critical_density = critical_density

        self.state["flow_constraint"] = 1.0

    def step(self, dt: float) -> None:
        density_map = self.inputs.get("density")
        if density_map is None:
            self.current_time += dt
            return

        density_map = np.asarray(density_map)
        peak_density = float(np.max(density_map))

        if peak_density <= self.critical_density:
            constraint = 1.0
        else:
            constraint = self.critical_density / peak_density

        max_flow = self.capacity * self.gate_width
        actual_demand = peak_density * 1.3
        if actual_demand > max_flow:
            constraint = min(constraint, max_flow / actual_demand)

        self.state["flow_constraint"] = float(np.clip(constraint, 0.05, 1.0))
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {"flow_constraint": self.state["flow_constraint"]}

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["density"],
            "outputs": ["flow_constraint"],
        }
