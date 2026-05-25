"""
PED1: PedestrianCrossingBody — 行人过马路微观动力学。
继承 RigidBody2D 社会力模型，针对过街场景调参。
Natural DT = 0.02s
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from atoms.physical.cm1_rigid_body import RigidBody2D


class PedestrianCrossingBody(RigidBody2D):
    """行人过街专用刚体动力学（慢速、小半径、高从众权重）。"""

    AGENT_RADIUS = 0.2
    MAX_SPEED = 1.4
    MASS = 70.0
    HERD_WEIGHT = 0.45
    EXIT_REACH_DIST = 0.8

    def __init__(
        self,
        sim_id: str = "PED1",
        num_agents: int = 40,
        world_size: Tuple[float, float] = (30.0, 20.0),
        default_speed: float = 1.2,
        seed: int = 42,
    ):
        super().__init__(sim_id=sim_id, num_agents=num_agents, world_size=world_size)
        self.natural_dt = 0.02
        rng = np.random.default_rng(seed)
        self.state["desired_speed"] = np.full(num_agents, default_speed, dtype=np.float64)
        self.state["crossing"] = np.zeros(num_agents, dtype=bool)
        self.state["waiting_time"] = np.zeros(num_agents, dtype=np.float64)

        # 默认在南侧人行道排队
        self.state["positions"] = rng.uniform(
            [8.0, 5.5], [22.0, 7.0], size=(num_agents, 2)
        )
        self.state["velocities"] = np.zeros((num_agents, 2), dtype=np.float64)
        self.state["active"] = np.ones(num_agents, dtype=bool)
        self.state["fallen"] = np.zeros(num_agents, dtype=bool)

        self._crosswalk_x_range: Tuple[float, float] = (12.0, 18.0)
        self._north_target_y: float = 14.0

    def set_crosswalk(self, x_range: Tuple[float, float], north_y: float) -> None:
        self._crosswalk_x_range = x_range
        self._north_target_y = north_y

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        super().step(dt)

        pos = self.state["positions"]
        vel = self.state["velocities"]
        waiting = self.state["waiting_time"]

        speed = np.linalg.norm(vel, axis=1)
        in_crosswalk_x = (pos[:, 0] >= self._crosswalk_x_range[0]) & (
            pos[:, 0] <= self._crosswalk_x_range[1]
        )
        moving_north = vel[:, 1] > 0.15
        self.state["crossing"] = in_crosswalk_x & (moving_north | (pos[:, 1] > 9.0))

        nearly_still = speed < 0.1
        self.state["waiting_time"] = np.where(
            nearly_still & (pos[:, 1] < 10.0), waiting + dt, waiting * 0.95
        )

    def _compute_desired_velocity(
        self, pos: np.ndarray, active: np.ndarray, efficiency: Any
    ) -> np.ndarray:
        """无外部 desired_velocity 时，朝北侧目标移动。"""
        desired_input = self.inputs.get("desired_velocity")
        if desired_input is not None and isinstance(desired_input, np.ndarray):
            return desired_input.astype(np.float64).copy()
        return super()._compute_desired_velocity(pos, active, efficiency)

    def set_exits(self, exits: List[Tuple[float, float]]) -> None:
        super().set_exits(exits)

    def get_outputs(self) -> Dict[str, Any]:
        out = super().get_outputs()
        out["crossing"] = self.state["crossing"].copy()
        out["waiting_time"] = self.state["waiting_time"].copy()
        return out

    def schema(self) -> Dict[str, Any]:
        base = super().schema()
        base["outputs"] = list(base["outputs"]) + ["crossing", "waiting_time"]
        return base
