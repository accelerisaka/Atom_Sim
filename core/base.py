"""
AtomicSimulator 基类 —— 所有仿真原子的统一契约。
设计原则：内部自洽，外部解耦。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AtomicSimulator(ABC):
    """所有仿真器的抽象基类。"""

    def __init__(self, sim_id: str, natural_dt: float):
        self.sim_id = sim_id
        self.natural_dt = natural_dt
        self.current_time: float = 0.0
        self.state: Dict[str, Any] = {}
        self.inputs: Dict[str, Any] = {}
        self._output_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, dt: float) -> None:
        """
        基于当前 self.inputs 和 self.state，推演 dt 时间后的新 state。
        仅修改 self.state 和 self.current_time。
        """

    @abstractmethod
    def get_outputs(self) -> Dict[str, Any]:
        """暴露给 S2S 总线的数据快照。"""

    # ------------------------------------------------------------------
    # 默认实现
    # ------------------------------------------------------------------

    def schema(self) -> Dict[str, Any]:
        """自描述接口：定义所需 Inputs 字段和生成的 Outputs 字段。"""
        return {"inputs": [], "outputs": []}

    def record_output(self) -> None:
        """将当前输出快照压入历史队列，供 AVG / INTERP 策略使用。"""
        self._output_history.append({
            "time": self.current_time,
            "data": self.get_outputs(),
        })

    def get_output_history(self, since: float = 0.0) -> List[Dict[str, Any]]:
        """获取 since 时刻之后的所有历史输出。"""
        return [h for h in self._output_history if h["time"] >= since]

    def trim_history(self, before: float) -> None:
        """丢弃过早的历史以控制内存。"""
        self._output_history = [
            h for h in self._output_history if h["time"] >= before
        ]

    def reset(self) -> None:
        """重置仿真器到初始状态。"""
        self.current_time = 0.0
        self.state.clear()
        self.inputs.clear()
        self._output_history.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.sim_id} t={self.current_time:.4f}>"
