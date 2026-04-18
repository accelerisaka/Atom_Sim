"""
因果一致性守卫 —— 确保跨组交互不违反因果律。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .protocol import S2SConnection


class CausalityGuard:
    """检测并防止因果性违规：代数环、未来数据引用。"""

    def __init__(self) -> None:
        self._sim_groups: Dict[str, "TimeGroup"] = {}
        self._adj: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register_group(self, group: "TimeGroup") -> None:
        for sim_id in group.sim_ids:
            self._sim_groups[sim_id] = group

    def get_group(self, sim_id: str) -> "TimeGroup":
        return self._sim_groups[sim_id]

    # ------------------------------------------------------------------
    # 连接验证
    # ------------------------------------------------------------------

    def validate_connection(self, conn: S2SConnection) -> List[str]:
        """返回违规警告列表（空 = 通过）。"""
        warnings: List[str] = []
        src_id = conn.source.sim_id
        tgt_id = conn.target.sim_id

        self._adj[src_id].add(tgt_id)

        if src_id in self._sim_groups and tgt_id in self._sim_groups:
            src_g = self._sim_groups[src_id]
            tgt_g = self._sim_groups[tgt_id]
            if tgt_g.local_time < src_g.last_output_time:
                warnings.append(
                    f"[CAUSALITY] {tgt_id} (t={tgt_g.local_time:.4f}) "
                    f"would read future data from {src_id} "
                    f"(last_output={src_g.last_output_time:.4f})"
                )

        if self._has_cycle(src_id, tgt_id):
            conn.min_delay = max(
                self._sim_groups.get(src_id, _DummyGroup()).dt,
                self._sim_groups.get(tgt_id, _DummyGroup()).dt,
            )
            warnings.append(
                f"[CYCLE] {src_id} <-> {tgt_id} detected. "
                f"Injected min_delay={conn.min_delay:.4f}s"
            )
        return warnings

    def validate_all(self, connections: List[S2SConnection]) -> List[str]:
        all_warnings: List[str] = []
        for conn in connections:
            all_warnings.extend(self.validate_connection(conn))
        return all_warnings

    # ------------------------------------------------------------------
    # 代数环检测 (DFS)
    # ------------------------------------------------------------------

    def _has_cycle(self, src: str, tgt: str) -> bool:
        """添加 src->tgt 边后是否出现 tgt->...->src 的环。"""
        visited: Set[str] = set()
        stack: List[str] = [tgt]
        while stack:
            node = stack.pop()
            if node == src:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._adj.get(node, set()))
        return False

    def detect_algebraic_loops(self) -> List[Tuple[str, ...]]:
        """返回所有强连通分量中包含多个节点的环。"""
        index_counter = [0]
        stack: List[str] = []
        lowlink: Dict[str, int] = {}
        index: Dict[str, int] = {}
        on_stack: Set[str] = set()
        sccs: List[Tuple[str, ...]] = []

        def strongconnect(v: str) -> None:
            index[v] = lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in self._adj.get(v, set()):
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    sccs.append(tuple(scc))

        for v in list(self._adj.keys()):
            if v not in index:
                strongconnect(v)
        return sccs


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------

class TimeGroup:
    """Orchestrator 中的时间分组信息容器。"""

    def __init__(self, name: str, dt: float, sim_ids: List[str]):
        self.name = name
        self.dt = dt
        self.sim_ids = sim_ids
        self.local_time: float = 0.0
        self.last_output_time: float = 0.0


class _DummyGroup:
    dt = 0.0
