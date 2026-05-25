"""
TL1: TrafficLightController — 路口交通信号灯控制器。
物理规律：周期性相位切换（行人/车辆互补信号）。
Natural DT = 0.1s
"""

from __future__ import annotations

from typing import Any, Dict

from core.base import AtomicSimulator


class TrafficLightController(AtomicSimulator):
    """双相位交通灯：行人绿灯时车辆红灯，反之亦然。"""

    PHASE_PED_GREEN = "PED_GREEN"
    PHASE_PED_RED = "PED_RED"

    def __init__(
        self,
        sim_id: str = "TL1",
        ped_green_duration: float = 15.0,
        ped_red_duration: float = 25.0,
        initial_phase: str = PHASE_PED_RED,
    ):
        super().__init__(sim_id, natural_dt=0.1)
        self.ped_green_duration = ped_green_duration
        self.ped_red_duration = ped_red_duration

        self.state["phase"] = initial_phase
        self.state["phase_elapsed"] = 0.0
        self.state["pedestrian_signal"] = 1.0 if initial_phase == self.PHASE_PED_GREEN else 0.0
        self.state["vehicle_signal"] = 0.0 if initial_phase == self.PHASE_PED_GREEN else 1.0
        self.state["time_remaining"] = (
            ped_green_duration if initial_phase == self.PHASE_PED_GREEN else ped_red_duration
        )

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        phase = self.state["phase"]
        elapsed = self.state["phase_elapsed"] + dt

        if phase == self.PHASE_PED_GREEN:
            duration = self.ped_green_duration
            if elapsed >= duration:
                phase = self.PHASE_PED_RED
                elapsed = 0.0
        else:
            duration = self.ped_red_duration
            if elapsed >= duration:
                phase = self.PHASE_PED_GREEN
                elapsed = 0.0

        self.state["phase"] = phase
        self.state["phase_elapsed"] = elapsed

        ped_sig = 1.0 if phase == self.PHASE_PED_GREEN else 0.0
        self.state["pedestrian_signal"] = ped_sig
        self.state["vehicle_signal"] = 1.0 - ped_sig
        self.state["time_remaining"] = (
            self.ped_green_duration - elapsed
            if phase == self.PHASE_PED_GREEN
            else self.ped_red_duration - elapsed
        )

        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "pedestrian_signal": float(self.state["pedestrian_signal"]),
            "vehicle_signal": float(self.state["vehicle_signal"]),
            "phase": self.state["phase"],
            "time_remaining": float(self.state["time_remaining"]),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [],
            "outputs": ["pedestrian_signal", "vehicle_signal", "phase", "time_remaining"],
        }
