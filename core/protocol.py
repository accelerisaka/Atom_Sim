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
    - 数据变换函数 (transform) 及其注册名 (transform_name)
    - 最小延迟（用于打破代数环）
    - 描述文本 (description) 便于 coding agent 读取

    注意：`transform` 必须来自 `core.transforms.TRANSFORM_REGISTRY`。
    原则上每一条连接都应显式声明 transform_name，即使是 `identity` 也要形式化声明，
    以保证系统内不存在"隐性耦合"。
    """
    connection_id: str
    source: PortAddress
    target: PortAddress
    strategy: ExchangeStrategy = ExchangeStrategy.ZOH
    # 注：所有通过 core.transforms.register_transform 注册的 transform，
    # 其可调用对象都被统一包裹为二参形式 `fn(value, ctx)`。
    # 即使原始是一参纯函数，包裹层也会忽略 ctx，保持调用点契约统一。
    transform: Optional[Callable[..., Any]] = None
    transform_name: Optional[str] = None
    min_delay: float = 0.0
    description: str = ""

    def apply_transform(self, value: Any, ctx: Any = None) -> Any:
        """
        应用 transform。`ctx` 为 `core.transforms.TransformContext`（可选），
        用于需要跨仿真器多源数据的变换（如按个体位置采样场）。
        """
        if self.transform is None:
            return value
        try:
            return self.transform(value, ctx)
        except TypeError:
            # 兜底：旧式一参 transform（未经 register_transform 包裹时的直接注入）
            return self.transform(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "source": repr(self.source),
            "target": repr(self.target),
            "strategy": self.strategy.value,
            "transform": self.transform_name,
            "min_delay": self.min_delay,
            "description": self.description,
        }

    def __repr__(self) -> str:
        tf = f" via {self.transform_name}" if self.transform_name else ""
        return (
            f"S2S[{self.connection_id}] "
            f"{self.source} --({self.strategy.value}{tf})--> {self.target}"
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
