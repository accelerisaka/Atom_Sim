"""
Orchestrator —— 全局执行引擎。
采用 PEQ（优先事件队列）+ 多速率时间步进的混合架构。
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import AtomicSimulator
from .causality import CausalityGuard, TimeGroup
from .protocol import EventMessage, ExchangeStrategy, S2SConnection
from .strategies import StrategyEngine

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    多速率联合仿真编排器。

    执行管道:
    1. 寻址下一个同步点
    2. 异步推进各仿真器本地时间
    3. 中断与事件队列消化
    4. 状态拉取与缓冲
    5. 策略应用与注入
    """

    def __init__(self) -> None:
        self.simulators: Dict[str, AtomicSimulator] = {}
        self.connections: List[S2SConnection] = []
        self.time_groups: Dict[str, TimeGroup] = {}
        self.global_time: float = 0.0

        # 优先事件队列 (PEQ)
        self._event_queue: List[EventMessage] = []
        # 事件回调: event_type -> handler
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        # 因果守卫
        self.causality_guard = CausalityGuard()

        # 全局状态总线 —— 每次同步后的快照
        self.global_bus: Dict[str, Dict[str, Any]] = {}
        # 历史记录（供可视化）
        self.history: List[Dict[str, Any]] = []

        # 事件连接索引: 目标 sim_id -> [connections with EVENT strategy]
        self._event_connections: Dict[str, List[S2SConnection]] = defaultdict(list)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register_simulator(self, sim: AtomicSimulator, group_name: str) -> None:
        self.simulators[sim.sim_id] = sim
        if group_name not in self.time_groups:
            raise KeyError(f"TimeGroup '{group_name}' not registered.")
        self.time_groups[group_name].sim_ids.append(sim.sim_id)

    def register_time_group(self, name: str, dt: float) -> None:
        group = TimeGroup(name, dt, [])
        self.time_groups[name] = group
        self.causality_guard.register_group(group)

    def add_connection(self, conn: S2SConnection) -> None:
        warnings = self.causality_guard.validate_connection(conn)
        for w in warnings:
            logger.warning(w)
        self.connections.append(conn)
        if conn.strategy == ExchangeStrategy.EVENT:
            self._event_connections[conn.target.sim_id].append(conn)

    def register_event_handler(
        self, event_type: str, handler: Callable[[EventMessage], None]
    ) -> None:
        self._event_handlers[event_type].append(handler)

    # ------------------------------------------------------------------
    # 事件注入
    # ------------------------------------------------------------------

    def inject_event(self, event: EventMessage) -> None:
        heapq.heappush(self._event_queue, event)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, duration: float, snapshot_interval: float = 0.1) -> None:
        """执行仿真 duration 秒。"""
        end_time = self.global_time + duration
        next_snapshot = self.global_time + snapshot_interval

        while self.global_time < end_time - 1e-9:
            # Phase 1: 找到下一个同步点
            next_sync = self._next_sync_point(end_time)

            # Phase 2: 处理到达同步点之前的所有事件
            self._process_events_until(next_sync)

            # Phase 3: 各组追赶至同步点
            self._advance_groups_to(next_sync)

            # Phase 4: 拉取输出到全局总线
            self._pull_outputs()

            # Phase 5: 策略应用与注入
            self._apply_strategies()

            self.global_time = next_sync

            # 定期快照
            if self.global_time >= next_snapshot - 1e-9:
                self._take_snapshot()
                next_snapshot += snapshot_interval

        logger.info(f"Simulation complete. t={self.global_time:.4f}s")

    # ------------------------------------------------------------------
    # Phase 1: 寻址下一个同步点
    # ------------------------------------------------------------------

    def _next_sync_point(self, end_time: float) -> float:
        candidates = []
        for group in self.time_groups.values():
            if group.dt > 0:
                next_t = group.local_time + group.dt
                if next_t <= end_time + 1e-9:
                    candidates.append(next_t)
        if self._event_queue:
            candidates.append(self._event_queue[0].timestamp)
        if not candidates:
            return end_time
        return min(min(candidates), end_time)

    # ------------------------------------------------------------------
    # Phase 2: 事件处理
    # ------------------------------------------------------------------

    def _process_events_until(self, until_time: float) -> None:
        while self._event_queue and self._event_queue[0].timestamp <= until_time + 1e-9:
            event = heapq.heappop(self._event_queue)
            logger.debug(f"Processing event: {event.event_type} @ t={event.timestamp}")

            for handler in self._event_handlers.get(event.event_type, []):
                handler(event)

            # EVENT 策略连接的直接注入
            for conn in self.connections:
                if (
                    conn.strategy == ExchangeStrategy.EVENT
                    and conn.source.sim_id == event.source_sim_id
                ):
                    target_sim = self.simulators.get(conn.target.sim_id)
                    if target_sim is not None:
                        value = event.payload.get(conn.source.port)
                        if value is not None:
                            target_sim.inputs[conn.target.port] = conn.apply_transform(value)
                            target_sim.step(0.0)
                            target_sim.record_output()

    # ------------------------------------------------------------------
    # Phase 3: 各组追赶
    # ------------------------------------------------------------------

    def _advance_groups_to(self, target_time: float) -> None:
        for group in self.time_groups.values():
            if group.dt <= 0:
                continue
            while group.local_time < target_time - 1e-9:
                step_end = min(group.local_time + group.dt, target_time)
                actual_dt = step_end - group.local_time
                for sim_id in group.sim_ids:
                    sim = self.simulators[sim_id]
                    sim.step(actual_dt)
                    sim.record_output()
                group.local_time = step_end
                group.last_output_time = step_end

    # ------------------------------------------------------------------
    # Phase 4: 拉取输出
    # ------------------------------------------------------------------

    def _pull_outputs(self) -> None:
        for sim_id, sim in self.simulators.items():
            self.global_bus[sim_id] = sim.get_outputs()

    # ------------------------------------------------------------------
    # Phase 5: 策略应用与注入
    # ------------------------------------------------------------------

    def _apply_strategies(self) -> None:
        for conn in self.connections:
            if conn.strategy == ExchangeStrategy.EVENT:
                continue

            src_sim = self.simulators.get(conn.source.sim_id)
            tgt_sim = self.simulators.get(conn.target.sim_id)
            if src_sim is None or tgt_sim is None:
                continue

            history = src_sim.get_output_history(
                since=self.global_time - self._get_lookback(conn)
            )
            value = StrategyEngine.resolve(conn, history, self.global_time)
            if value is not None:
                tgt_sim.inputs[conn.target.port] = value

    def _get_lookback(self, conn: S2SConnection) -> float:
        """根据策略决定需要回溯多少历史。"""
        tgt_sim = self.simulators.get(conn.target.sim_id)
        if tgt_sim is None:
            return 1.0
        if conn.strategy == ExchangeStrategy.AVG:
            return tgt_sim.natural_dt * 2
        if conn.strategy == ExchangeStrategy.INTERP:
            src_sim = self.simulators.get(conn.source.sim_id)
            return (src_sim.natural_dt if src_sim else 1.0) * 3
        return 1.0

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> None:
        snapshot: Dict[str, Any] = {"time": round(self.global_time, 6)}
        for sim_id, sim in self.simulators.items():
            snapshot[sim_id] = _safe_serialize(sim.get_outputs())
        self.history.append(snapshot)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def get_sim_group(self, sim_id: str) -> Optional[str]:
        for gname, group in self.time_groups.items():
            if sim_id in group.sim_ids:
                return gname
        return None

    def summary(self) -> str:
        lines = [f"=== Orchestrator  t={self.global_time:.4f}s ==="]
        for gname, group in self.time_groups.items():
            lines.append(f"  [{gname}] dt={group.dt}s  sims={group.sim_ids}")
        lines.append(f"  Connections: {len(self.connections)}")
        lines.append(f"  Pending events: {len(self._event_queue)}")
        return "\n".join(lines)


def _safe_serialize(obj: Any) -> Any:
    """把 numpy array 等转成可 JSON 序列化的形式。"""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
