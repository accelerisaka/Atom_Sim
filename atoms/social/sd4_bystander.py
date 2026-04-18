"""
SD4: BystanderEffect — 旁观者效应与救助。
社会博弈：当有人摔倒时，基于周围人数计算"责任扩散"。
P(help) ∝ 1/√N ，受人格参数"宜人性"调制。
Natural DT = Event-driven
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.base import AtomicSimulator


class BystanderEffect(AtomicSimulator):
    """事件驱动的旁观者效应仿真器。"""

    def __init__(
        self,
        sim_id: str = "SD4",
        num_agents: int = 50,
        base_help_probability: float = 0.6,
    ):
        super().__init__(sim_id, natural_dt=0.0)  # event-driven
        self.num_agents = num_agents
        self.base_help_prob = base_help_probability

        rng = np.random.default_rng(123)
        self.state["agreeableness"] = rng.uniform(0.2, 1.0, size=num_agents)
        self.state["help_action_triggered"] = np.zeros(num_agents, dtype=bool)
        self.state["helper_ids"] = []

    def step(self, dt: float) -> None:
        downed = self.inputs.get("downed_agents_count", 0)
        bystanders = self.inputs.get("nearby_bystanders_count", 0)
        fallen_mask = self.inputs.get("fallen_mask")
        active_mask = self.inputs.get("active_mask")
        positions = self.inputs.get("agent_positions")

        self.state["help_action_triggered"][:] = False
        self.state["helper_ids"] = []

        if downed == 0 and (fallen_mask is None or not np.any(fallen_mask)):
            self.current_time += dt
            return

        n = self.num_agents

        if fallen_mask is not None and positions is not None:
            fallen_mask = np.asarray(fallen_mask, dtype=bool)[:n]
            pos = np.asarray(positions, dtype=np.float64)[:n]
            active = np.asarray(active_mask, dtype=bool)[:n] if active_mask is not None else np.ones(n, dtype=bool)

            fallen_indices = np.where(fallen_mask)[0]

            for fi in fallen_indices:
                diffs = pos - pos[fi]
                dists = np.linalg.norm(diffs, axis=1)
                nearby = (dists < 5.0) & active & ~fallen_mask
                nearby_count = int(np.sum(nearby))
                if nearby_count == 0:
                    continue

                # P(help) = base * agreeableness / sqrt(N)
                nearby_ids = np.where(nearby)[0]
                for nid in nearby_ids:
                    p = (self.base_help_prob
                         * self.state["agreeableness"][nid]
                         / np.sqrt(nearby_count))
                    if np.random.random() < p:
                        self.state["help_action_triggered"][nid] = True
                        self.state["helper_ids"].append(int(nid))
                        break  # 一人施救即可
        else:
            # 简化路径：仅用聚合计数
            if bystanders > 0:
                p = self.base_help_prob / np.sqrt(max(bystanders, 1))
                if np.random.random() < p:
                    self.state["help_action_triggered"][0] = True

        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "help_action_triggered": self.state["help_action_triggered"].copy(),
            "helper_ids": list(self.state["helper_ids"]),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [
                "downed_agents_count", "nearby_bystanders_count",
                "fallen_mask", "active_mask", "agent_positions",
            ],
            "outputs": ["help_action_triggered", "helper_ids"],
        }
