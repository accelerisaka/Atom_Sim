"""
联合仿真环境（topology）的"先后顺序"与"基线原子集"计算。

为什么需要它？
----------------
原子仿真器在某个联合仿真环境里的标识 —— 复用 / 继承 / 新增 ——
天然依赖"在这个环境之前都有哪些原子已经被引入"。
该模块给出这个"之前"的定义：

1. 内置场景按固定顺序排列：
       fire (第 1 位) → grid (第 2 位)
   这是仓库出厂自带的两个项目，火灾场景先于电网场景。

2. 其它（由 cursor_agent 生成的）场景按 manifest 中
   ``generated_at`` 字段排序，紧跟内置场景之后；
   找不到 manifest 时退化为 yaml 文件的 mtime。

3. 任一场景 X 的"基线"定义为：在 X 之前出现过的所有场景
   引用到的原子文件相对路径的并集。
   classifier 用此基线判定 X 中每个原子是 reused / inherited / new。

这样，原子的"复用/继承/新增"标识就被严格绑定到了所在的联合仿真环境，
而不是依赖一份与场景无关的"atoms/ 目录快照"。
"""

from __future__ import annotations

from typing import List, Set

from .atom_scanner import build_class_to_file_index
from .manifest import load_manifest
from .topology_parser import find_topology_path, list_topologies, load_topology


# 出厂内置场景的固定顺序。fire 先、grid 后。
BUILTIN_ORDER: List[str] = ["fire", "grid"]


def topology_order() -> List[str]:
    """返回当前仓库中所有 topology 名字的有序列表。

    排序规则
    --------
    - BUILTIN_ORDER 中存在的场景按其固定先后顺序排在最前。
    - 其它场景按下列优先级排序：
        1. manifest 的 ``generated_at``（cursor_agent 生成时落盘）
        2. topology yaml 的文件 mtime
        3. 名字字母序兜底
    """
    topos = list_topologies()
    name_to_topo = {t["name"]: t for t in topos}

    ordered: List[str] = []
    for n in BUILTIN_ORDER:
        if n in name_to_topo:
            ordered.append(n)

    remaining = [t for t in topos if t["name"] not in ordered]

    def _sort_key(t: dict) -> tuple:
        manifest = load_manifest(t["name"])
        if manifest and manifest.get("generated_at"):
            return (0, str(manifest["generated_at"]), t["name"])
        path = find_topology_path(t["name"])
        if path is not None:
            try:
                return (1, path.stat().st_mtime, t["name"])
            except OSError:
                pass
        return (2, 0.0, t["name"])

    remaining.sort(key=_sort_key)
    ordered.extend(t["name"] for t in remaining)
    return ordered


def atom_files_used_by(name: str) -> Set[str]:
    """返回 ``name`` 这个 topology 直接引用到的所有原子文件相对路径集合。"""
    topo = load_topology(name)
    if topo is None:
        return set()

    class_index = build_class_to_file_index()
    out: Set[str] = set()
    for sim in topo["simulators"]:
        cls_ref = sim.get("class", "")
        file_rel = class_index.get(cls_ref)
        if file_rel is None and "." in cls_ref:
            tail = cls_ref.rsplit(".", 1)[-1]
            file_rel = class_index.get(tail)
        if file_rel:
            out.add(file_rel)
    return out


def baseline_for(name: str) -> Set[str]:
    """返回 ``name`` 这个 topology 的基线 —— 它之前所有 topology
    引用到的原子文件相对路径并集。

    name 是序列里的第一位时返回空集合（其中所有原子都将被判为新增 / 继承）。
    name 不在当前仓库中时也返回空集合。
    """
    order = topology_order()
    if name not in order:
        return set()
    idx = order.index(name)

    baseline: Set[str] = set()
    for prev in order[:idx]:
        baseline |= atom_files_used_by(prev)
    return baseline
