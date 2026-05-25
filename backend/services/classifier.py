"""
原子仿真器在某个联合仿真环境里的"复用 / 继承 / 新增"分类逻辑。

判定准则（按优先级）：
1. 若 class 来源文件已在 pre-snapshot 中（agent 跑前就存在） → reused
2. 否则 ast 解析该新文件，若其顶层 class 的父类 ∈ 已存在的仿真器类名集合 → inherited
3. 其余 → new

snapshot 只在 cursor_agent 触发的 generate 任务里有意义；
首屏（GET /api/scenarios/{name}）若没有 snapshot 上下文，则统一标 reused（已存在的就是"已有"）。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set

from .atom_scanner import build_class_to_file_index, scan_all_atoms


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_classes_of(file_rel: str) -> List[str]:
    p = _project_root() / file_rel
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out: List[str] = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    out.append(b.id)
                elif isinstance(b, ast.Attribute):
                    parts: List[str] = []
                    cur = b
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                    out.append(".".join(reversed(parts)))
            return out
    return []


def classify_simulators(
    topology_simulators: List[Dict],
    pre_snapshot: Optional[Set[str]] = None,
) -> Dict[str, Dict]:
    """
    针对一个 topology 的 simulators 列表，返回 {sim_id: {classification, file, ...}}。

    pre_snapshot=None：把所有能在当前 atoms/ 找到的都视为 reused。
    pre_snapshot=set：基于快照差异给出 reused/inherited/new。
    """
    class_index = build_class_to_file_index()
    existing_class_names = {info["class_name"] for info in scan_all_atoms()}

    out: Dict[str, Dict] = {}
    for sim in topology_simulators:
        sim_id = sim["sim_id"]
        cls_ref = sim["class"]
        file_rel = class_index.get(cls_ref)

        # 若 cls_ref 形如 "atoms.physical.HeatConduction"，提取尾段类名再查
        if file_rel is None and "." in cls_ref:
            tail = cls_ref.rsplit(".", 1)[-1]
            file_rel = class_index.get(tail)

        classification = "unknown"
        if file_rel is None:
            classification = "unknown"
        elif pre_snapshot is None:
            classification = "reused"
        else:
            if file_rel in pre_snapshot:
                classification = "reused"
            else:
                # 新文件 → 看 class 继承
                bases = _base_classes_of(file_rel)
                inherited = False
                for b in bases:
                    bn = b.rsplit(".", 1)[-1]
                    # 父类是 AtomicSimulator 视为 new；父类是其它已存在的仿真器类视为 inherited
                    if bn == "AtomicSimulator":
                        continue
                    if bn in existing_class_names:
                        inherited = True
                        break
                classification = "inherited" if inherited else "new"

        out[sim_id] = {
            "classification": classification,
            "file": file_rel,
        }
    return out
