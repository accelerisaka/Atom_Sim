"""
VEH1: LaneVehicleTraffic — 车道车辆运动仿真。
物理规律：一维匀速运动，遇红灯在停止线前制动。
Natural DT = 0.02s
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from core.base import AtomicSimulator


class LaneVehicleTraffic(AtomicSimulator):
    """沿水平车道行驶的车辆队列，遵守交通信号。"""

    def __init__(
        self,
        sim_id: str = "VEH1",
        num_vehicles: int = 6,
        lane_y: float = 10.0,
        cruise_speed: float = 8.0,
        vehicle_length: float = 4.0,
        stop_line_x: float = 12.0,
        spawn_x: float = 0.5,
        world_size: Tuple[float, float] = (30.0, 20.0),
        respawn_delay: float = 3.0,
    ):
        super().__init__(sim_id, natural_dt=0.02)
        self.num_vehicles = num_vehicles
        self.lane_y = lane_y
        self.cruise_speed = cruise_speed
        self.vehicle_length = vehicle_length
        self.stop_line_x = stop_line_x
        self.spawn_x = spawn_x
        self.world_size = world_size
        self.respawn_delay = respawn_delay

        spacing = vehicle_length * 1.8
        positions = np.zeros((num_vehicles, 2), dtype=np.float64)
        for i in range(num_vehicles):
            positions[i] = [spawn_x - i * spacing, lane_y]

        self.state["positions"] = positions
        self.state["velocities"] = np.zeros((num_vehicles, 2), dtype=np.float64)
        self.state["active"] = np.ones(num_vehicles, dtype=bool)
        self.state["respawn_timer"] = np.zeros(num_vehicles, dtype=np.float64)

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        signal = self.inputs.get("vehicle_signal", 1.0)
        if isinstance(signal, np.ndarray):
            signal = float(signal.ravel()[0])
        else:
            signal = float(signal)

        pos = self.state["positions"]
        vel = self.state["velocities"]
        active = self.state["active"]
        timers = self.state["respawn_timer"]

        for i in range(self.num_vehicles):
            if not active[i]:
                timers[i] += dt
                if timers[i] >= self.respawn_delay:
                    pos[i] = [self.spawn_x, self.lane_y]
                    vel[i] = [0.0, 0.0]
                    active[i] = True
                    timers[i] = 0.0
                continue

            x = pos[i, 0]
            can_go = signal > 0.5

            min_gap = float("inf")
            for j in range(self.num_vehicles):
                if j == i or not active[j]:
                    continue
                if pos[j, 0] > x:
                    min_gap = min(min_gap, pos[j, 0] - x)

            target_speed = 0.0
            if can_go and min_gap >= self.vehicle_length * 1.4:
                target_speed = self.cruise_speed
            elif not can_go:
                if x < self.stop_line_x - 2.0:
                    target_speed = self.cruise_speed
                elif x >= self.stop_line_x - self.vehicle_length * 0.5:
                    target_speed = 0.0
                else:
                    target_speed = max(0.0, (self.stop_line_x - x) * 2.0)

            vel[i, 0] += (target_speed - vel[i, 0]) * min(1.0, dt * 6.0)
            vel[i, 1] = 0.0
            pos[i, 0] += vel[i, 0] * dt

            if pos[i, 0] > self.world_size[0] + 2.0:
                active[i] = False
                timers[i] = 0.0

        self.state["positions"] = pos
        self.state["velocities"] = vel
        self.state["active"] = active
        self.state["respawn_timer"] = timers
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "positions": self.state["positions"].copy(),
            "velocities": self.state["velocities"].copy(),
            "active": self.state["active"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["vehicle_signal"],
            "outputs": ["positions", "velocities", "active"],
        }
