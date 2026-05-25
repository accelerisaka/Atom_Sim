"""
atoms/ 目录文件快照——用于在 cursor_agent 运行前后 diff，识别新增文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_SCAN_DIRS = ("atoms/physical", "atoms/social")


def snapshot_atoms() -> Set[str]:
    """返回相对仓库根的 .py 文件相对路径集合。"""
    root = _project_root()
    out: Set[str] = set()
    for sub in _SCAN_DIRS:
        d = root / sub
        if not d.exists():
            continue
        for py in d.glob("*.py"):
            if py.name == "__init__.py":
                continue
            out.add(py.relative_to(root).as_posix())
    return out


def diff_atoms(before: Set[str]) -> List[str]:
    """返回快照之后新增的文件相对路径列表。"""
    after = snapshot_atoms()
    new_files = sorted(after - before)
    return new_files
