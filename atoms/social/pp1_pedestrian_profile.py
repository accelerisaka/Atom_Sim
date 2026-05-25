"""
PP1: PedestrianProfile — 行人性格档案（静态）。
社会意义：异质性格参数，驱动闯红灯倾向与规则遵守度。
Natural DT = 1.0s
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.base import AtomicSimulator


class PedestrianProfile(AtomicSimulator):
    """行人物种档案：闯红灯倾向、耐心、规则遵守度。"""

    def __init__(
        self,
        sim_id: str = "PP1",
        num_agents: int = 40,
        jaywalk_range: Tuple[float, float] = (0.05, 0.95),
        patience_range: Tuple[float, float] = (0.1, 0.9),
        rule_compliance_range: Tuple[float, float] = (0.1, 0.95),
        seed: int = 77,
    ):
        super().__init__(sim_id, natural_dt=1.0)
        self.num_agents = num_agents
        rng = np.random.default_rng(seed)

        # Beta 分布制造性格两极分化：多数守规矩，少数极端闯红灯
        jaywalk = rng.beta(2.0, 5.0, size=num_agents)
        jaywalk = np.clip(jaywalk * (jaywalk_range[1] - jaywalk_range[0]) + jaywalk_range[0], 0, 1)

        # 强制生成几个极端闯红灯者
        extreme_count = max(2, num_agents // 8)
        extreme_idx = rng.choice(num_agents, size=extreme_count, replace=False)
        jaywalk[extreme_idx] = rng.uniform(0.75, 0.98, size=extreme_count)

        patience = rng.uniform(patience_range[0], patience_range[1], size=num_agents)
        patience[extreme_idx] = rng.uniform(0.05, 0.25, size=extreme_count)

        rule = 1.0 - jaywalk * 0.7 + rng.normal(0, 0.08, size=num_agents)
        rule = np.clip(rule, rule_compliance_range[0], rule_compliance_range[1])

        self.state["jaywalk_tendency"] = jaywalk
        self.state["patience"] = patience
        self.state["rule_compliance"] = rule

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        self.current_time += dt

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "jaywalk_tendency": self.state["jaywalk_tendency"].copy(),
            "patience": self.state["patience"].copy(),
            "rule_compliance": self.state["rule_compliance"].copy(),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "inputs": [],
            "outputs": ["jaywalk_tendency", "patience", "rule_compliance"],
        }
