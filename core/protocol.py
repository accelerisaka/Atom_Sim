"""
S2S 连接协议 —— 定义数据在原子间的流转规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class ExchangeStrategy(Enum):
    """跨速率交换策略枚举。"""
    ZOH = "ZOH"        # 零阶保持
    AVG = "AVG"        # 时间平均
    INTERP = "INTERP"  # 线性插值
    EVENT = "EVENT"    # 事件触发脉冲


@dataclass
class PortAddress:
    """仿真器端口地址。"""
    sim_id: str
    port: str

    def __repr__(self) -> str:
        return f"{self.sim_id}.{self.port}"


@dataclass
class S2SConnection:
    """
    一条 S2S 连接，描述了:
    - 源端口 -> 目标端口
    - 跨速率交换策略
    - 可选的数据变换函数
    - 最小延迟（用于打破代数环）
    """
    connection_id: str
    source: PortAddress
    target: PortAddress
    strategy: ExchangeStrategy = ExchangeStrategy.ZOH
    transform: Optional[Callable[[Any], Any]] = None
    min_delay: float = 0.0

    def apply_transform(self, value: Any) -> Any:
        if self.transform is not None:
            return self.transform(value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "source": repr(self.source),
            "target": repr(self.target),
            "strategy": self.strategy.value,
            "min_delay": self.min_delay,
        }

    def __repr__(self) -> str:
        return (
            f"S2S[{self.connection_id}] "
            f"{self.source} --({self.strategy.value})--> {self.target}"
        )


@dataclass
class EventMessage:
    """事件触发脉冲消息。"""
    event_type: str
    source_sim_id: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 越小越优先

    def __lt__(self, other: "EventMessage") -> bool:
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp
