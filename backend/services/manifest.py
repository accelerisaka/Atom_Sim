"""
持久化每次 generate 任务的 atoms 快照基线，使得新生成场景的
"复用 / 继承 / 新增"分类在后续请求里仍然可用。

存储位置：cursor_agent/_scenario_manifests/<name>.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_dir() -> Path:
    d = _project_root() / "cursor_agent" / "_scenario_manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_generation(
    name: str,
    baseline_atom_files: List[str],
    new_atom_files: List[str],
    extras: Optional[Dict[str, Any]] = None,
) -> Path:
    payload = {
        "name": name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_atom_files": sorted(baseline_atom_files),
        "new_atom_files": sorted(new_atom_files),
    }
    if extras:
        payload.update(extras)
    path = _manifest_dir() / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(name: str) -> Optional[Dict[str, Any]]:
    path = _manifest_dir() / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
