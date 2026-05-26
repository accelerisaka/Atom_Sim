"""
BC1: EmergencyBroadcast — 体育场应急广播系统仿真。
社会/设施：灾害发生后场馆开启 PA，向观众播报备用疏散路线。
实现：广播激活状态 + 引导强度（与播报持续时间相关）。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from core.base import AtomicSimulator


class EmergencyBroadcast(AtomicSimulator):
    """应急广播激活与引导强度仿真器。"""

    def __init__(
        self,
        sim_id: str = "BC1",
        broadcast_delay: float = 5.0,
        ramp_duration: float = 8.0,
        max_guidance: float = 1.0,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.broadcast_delay = broadcast_delay
        self.ramp_duration = ramp_duration
        self.max_guidance = max_guidance

        self.state["broadcast_active"] = 0.0
        self.state["guidance_intensity"] = 0.0
        self.state["announced_exit_index"] = 0
        self._exit_positions: List[np.ndarray] = []

    def set_announced_exit(self, exit_index: int, exits: List[Tuple[float, float]]) -> None:
        self.state["announced_exit_index"] = int(exit_index)
        self._exit_positions = [np.array(e, dtype=np.float64) for e in exits]

    def get_announced_exit_position(self) -> np.ndarray:
        idx = int(self.state["announced_exit_index"])
        if not self._exit_positions:
            return np.zeros(2, dtype=np.float64)
        idx = min(max(idx, 0), len(self._exit_positions) - 1)
        return self._exit_positions[idx].copy()

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        override = self.inputs.get("force_broadcast")
        if override is not None and float(override) > 0.5:
            active = 1.0
            intensity = self.max_guidance
        else:
            t = self.current_time
            if t < self.broadcast_delay:
                active = 0.0
                intensity = 0.0
            else:
                active = 1.0
                ramp_t = t - self.broadcast_delay
                intensity = self.max_guidance * min(ramp_t / max(self.ramp_duration, 0.1), 1.0)

        self.state["broadcast_active"] = float(active)
        self.state["guidance_intensity"] = float(intensity)
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "broadcast_active": self.state["broadcast_active"],
            "guidance_intensity": self.state["guidance_intensity"],
            "announced_exit_index": int(self.state["announced_exit_index"]),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["force_broadcast"],
            "outputs": ["broadcast_active", "guidance_intensity", "announced_exit_index"],
        }
