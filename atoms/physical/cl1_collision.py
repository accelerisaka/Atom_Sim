"""
CL1: CollisionDetector — 车-行人碰撞检测。
物理规律：圆形碰撞体重叠判定 + 冲量输出。
Natural DT = 0.02s
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class CollisionDetector(AtomicSimulator):
    """检测车辆与行人之间的碰撞，输出命中掩码与冲量。"""

    def __init__(
        self,
        sim_id: str = "CL1",
        num_pedestrians: int = 40,
        num_vehicles: int = 6,
        ped_radius: float = 0.25,
        veh_radius: float = 1.2,
        impulse_magnitude: float = 25.0,
    ):
        super().__init__(sim_id, natural_dt=0.02)
        self.num_pedestrians = num_pedestrians
        self.num_vehicles = num_vehicles
        self.ped_radius = ped_radius
        self.veh_radius = veh_radius
        self.impulse_magnitude = impulse_magnitude
        self.collision_dist = ped_radius + veh_radius

        self.state["hit_mask"] = np.zeros(num_pedestrians, dtype=bool)
        self.state["hit_impulses"] = np.zeros((num_pedestrians, 2), dtype=np.float64)
        self.state["collision_count"] = 0
        self.state["new_collisions"] = np.zeros(num_pedestrians, dtype=bool)

    def step(self, dt: float) -> None:
        if dt <= 0:
            return

        ped_pos = self.inputs.get("ped_positions")
        veh_pos = self.inputs.get("veh_positions")
        ped_active = self.inputs.get("ped_active")
        veh_active = self.inputs.get("veh_active")

        hit_mask = np.zeros(self.num_pedestrians, dtype=bool)
        new_hits = np.zeros(self.num_pedestrians, dtype=bool)
        impulses = np.zeros((self.num_pedestrians, 2), dtype=np.float64)
        prev_hits = self.state["hit_mask"].copy()
        collision_count = int(self.state["collision_count"])

        if ped_pos is None or veh_pos is None:
            self.state["hit_mask"] = hit_mask
            self.state["hit_impulses"] = impulses
            self.state["new_collisions"] = new_hits
            self.current_time += dt
            return

        pp = np.asarray(ped_pos, dtype=np.float64)
        vp = np.asarray(veh_pos, dtype=np.float64)
        pa = np.asarray(ped_active, dtype=bool) if ped_active is not None else np.ones(pp.shape[0], dtype=bool)
        va = np.asarray(veh_active, dtype=bool) if veh_active is not None else np.ones(vp.shape[0], dtype=bool)

        n_ped = min(pp.shape[0], self.num_pedestrians)
        n_veh = min(vp.shape[0], self.num_vehicles)

        for i in range(n_ped):
            if not pa[i]:
                continue
            for j in range(n_veh):
                if not va[j]:
                    continue
                diff = pp[i] - vp[j]
                dist = np.linalg.norm(diff)
                if dist < self.collision_dist:
                    hit_mask[i] = True
                    if not prev_hits[i]:
                        new_hits[i] = True
                        collision_count += 1
                    if dist > 1e-6:
                        direction = diff / dist
                    else:
                        direction = np.array([0.0, 1.0])
                    impulses[i] += direction * self.impulse_magnitude

        self.state["hit_mask"] = hit_mask
        self.state["new_collisions"] = new_hits
        self.state["hit_impulses"] = impulses
        self.state["collision_count"] = collision_count
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "hit_mask": self.state["hit_mask"].copy(),
            "new_collisions": self.state["new_collisions"].copy(),
            "hit_impulses": self.state["hit_impulses"].copy(),
            "collision_count": int(self.state["collision_count"]),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": ["ped_positions", "veh_positions", "ped_active", "veh_active"],
            "outputs": ["hit_mask", "new_collisions", "hit_impulses", "collision_count"],
        }
