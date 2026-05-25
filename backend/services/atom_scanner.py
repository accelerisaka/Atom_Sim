"""
扫描 atoms/{physical,social}/*.py，提取每个原子仿真器的元数据。

每个 .py 文件被视为一个原子仿真器：
- 模块顶部 docstring → 详情面板要显示的"开头注释"
- 顶层 class 定义（继承自 AtomicSimulator 或其子类）→ class_name + base_class
- sim_id：优先取 class.__init__ 的默认 sim_id（通过 ast 解析默认参数），
  否则回落到 docstring 首行 "Xxx: ClassName — ..." 的前缀
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_ATOM_DIRS = [
    Path("atoms/physical"),
    Path("atoms/social"),
]


def _project_root() -> Path:
    """backend/services/atom_scanner.py → repo root."""
    return Path(__file__).resolve().parents[2]


def _extract_default_sim_id(cls: ast.ClassDef) -> Optional[str]:
    """从 class.__init__ 的默认参数中提取 sim_id 的默认值。"""
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        args = node.args
        defaults = args.defaults or []
        positional = args.args[1:]  # 跳过 self
        if not defaults:
            return None
        offset = len(positional) - len(defaults)
        for i, arg in enumerate(positional):
            if arg.arg != "sim_id":
                continue
            di = i - offset
            if di < 0 or di >= len(defaults):
                return None
            default_node = defaults[di]
            if isinstance(default_node, ast.Constant) and isinstance(default_node.value, str):
                return default_node.value
        return None
    return None


def _split_docstring_first_line(doc: str) -> str:
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _guess_sim_id_from_docstring(doc: str) -> Optional[str]:
    """docstring 首行常为 'TD1: HeatConduction — 热传导...'，截取前缀。"""
    first = _split_docstring_first_line(doc)
    if not first:
        return None
    m = re.match(r"^([A-Z]{1,4}\d{0,3})\s*[:：]", first)
    if m:
        return m.group(1)
    return None


def _guess_name_from_docstring(doc: str) -> Optional[str]:
    """docstring 首行的 ':' 之后取作 display name。"""
    first = _split_docstring_first_line(doc)
    if not first:
        return None
    # 形如 "TD1: HeatConduction — 热传导与火焰仿真。"
    m = re.match(r"^[A-Z]{1,4}\d{0,3}\s*[:：]\s*([^—\-—]+?)(?:\s*[—\-—]\s*(.+))?$", first)
    if m:
        cn = m.group(1).strip()
        cn_desc = m.group(2)
        return cn if not cn_desc else f"{cn} — {cn_desc.strip()}"
    return first


def scan_atom_file(path: Path) -> Optional[Dict[str, Any]]:
    """解析单个 .py 文件。返回 None 表示该文件不是仿真器（无合适 class）。"""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    module_doc = ast.get_docstring(tree) or ""

    # 找第一个看起来像仿真器的 class（继承自 AtomicSimulator 或任意类）
    candidate: Optional[ast.ClassDef] = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            candidate = node
            break

    if candidate is None:
        return None

    base_classes: List[str] = []
    for b in candidate.bases:
        if isinstance(b, ast.Name):
            base_classes.append(b.id)
        elif isinstance(b, ast.Attribute):
            parts: List[str] = []
            cur: Any = b
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            base_classes.append(".".join(reversed(parts)))

    class_name = candidate.name
    default_sim_id = _extract_default_sim_id(candidate)
    sim_id = (
        default_sim_id
        or _guess_sim_id_from_docstring(module_doc)
        or class_name.upper()[:4]
    )
    display_name = _guess_name_from_docstring(module_doc) or class_name

    rel = path.relative_to(_project_root()).as_posix()
    category = "physical" if "physical" in rel else "social" if "social" in rel else "other"
    module_path = ".".join(rel[:-3].split("/"))  # atoms/physical/td1_heat.py -> atoms.physical.td1_heat

    return {
        "sim_id": sim_id,
        "class_name": class_name,
        "name": display_name,
        "docstring": module_doc.strip(),
        "file": rel,
        "module": module_path,
        "category": category,
        "base_classes": base_classes,
    }


def scan_all_atoms() -> List[Dict[str, Any]]:
    """扫描 atoms/{physical,social}/ 下的全部 .py 文件。"""
    root = _project_root()
    out: List[Dict[str, Any]] = []
    seen_files: set = set()
    for sub in _ATOM_DIRS:
        d = root / sub
        if not d.exists():
            continue
        for py in sorted(d.glob("*.py")):
            if py.name == "__init__.py":
                continue
            if py in seen_files:
                continue
            seen_files.add(py)
            info = scan_atom_file(py)
            if info is not None:
                out.append(info)
    return out


def build_class_to_file_index() -> Dict[str, str]:
    """class_name -> 文件相对路径，便于 topology -> 文件定位。"""
    idx: Dict[str, str] = {}
    for info in scan_all_atoms():
        idx[info["class_name"]] = info["file"]
        # 同时给出 atoms.physical.ClassName 这种短形式，对应 topology yaml 的 class 字段
        cat = info["category"]
        if cat in ("physical", "social"):
            short = f"atoms.{cat}.{info['class_name']}"
            idx[short] = info["file"]
    return idx
