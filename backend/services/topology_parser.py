"""
解析 config/topology_*.yaml，转成前端友好的结构。

- list_topologies(): 返回 [{name, file, has_run_script}]，name 是 fire/grid/xxx
- load_topology(name): 返回 {name, time_groups, simulators[], connections[], scenario}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


CONFIG_DIR = "config"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _topology_files() -> List[Path]:
    root = _project_root() / CONFIG_DIR
    if not root.exists():
        return []
    out: List[Path] = []
    for p in sorted(root.glob("topology_*.y*ml")):
        out.append(p)
    # 兼容根目录直接的 topology.yaml
    bare = _project_root() / CONFIG_DIR / "topology.yaml"
    if bare.exists() and bare not in out:
        out.append(bare)
    return out


def _name_from_file(path: Path) -> str:
    stem = path.stem  # topology_fire
    if stem.startswith("topology_"):
        return stem[len("topology_"):]
    return stem


def list_topologies() -> List[Dict[str, Any]]:
    root = _project_root()
    out: List[Dict[str, Any]] = []
    for p in _topology_files():
        name = _name_from_file(p)
        run_script = root / f"run_{name}.py"
        out.append({
            "name": name,
            "file": p.relative_to(root).as_posix(),
            "has_run_script": run_script.exists(),
            "run_script": f"run_{name}.py" if run_script.exists() else None,
        })
    return out


def find_topology_path(name: str) -> Optional[Path]:
    for p in _topology_files():
        if _name_from_file(p) == name:
            return p
    return None


def _normalize_class_ref(cls: str) -> str:
    """topology yaml 中 'atoms.physical.HeatConduction' 等就原样返回。"""
    return cls


def load_topology(name: str) -> Optional[Dict[str, Any]]:
    path = find_topology_path(name)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    time_groups = data.get("time_groups", {}) or {}
    sim_block = data.get("simulators", {}) or {}
    conn_block = data.get("connections", []) or []
    scenario = data.get("scenario", {}) or {}

    simulators: List[Dict[str, Any]] = []
    for sim_id, cfg in sim_block.items():
        simulators.append({
            "sim_id": sim_id,
            "class": _normalize_class_ref(cfg.get("class", "")),
            "group": cfg.get("group", ""),
            "params": cfg.get("params", {}) or {},
        })

    connections: List[Dict[str, Any]] = []
    for c in conn_block:
        connections.append({
            "connection_id": c.get("connection_id", ""),
            "source_sim": (c.get("source") or {}).get("sim_id", ""),
            "source_port": (c.get("source") or {}).get("port", ""),
            "target_sim": (c.get("target") or {}).get("sim_id", ""),
            "target_port": (c.get("target") or {}).get("port", ""),
            "strategy": c.get("strategy", ""),
            "transform": c.get("transform", ""),
            "description": c.get("description", ""),
        })

    return {
        "name": name,
        "file": path.relative_to(_project_root()).as_posix(),
        "time_groups": time_groups,
        "simulators": simulators,
        "connections": connections,
        "scenario": scenario,
    }


def write_temp_topology(name: str, scenario_override: Dict[str, Any]) -> Optional[Path]:
    """以 name 对应的 yaml 为底，覆盖 scenario 字段，写到 config/_runtime_<name>.yaml。"""
    path = find_topology_path(name)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if scenario_override:
        # 浅合并：override 顶层覆盖原 scenario 顶层 key
        data["scenario"] = {**(data.get("scenario") or {}), **scenario_override}
    out_path = _project_root() / CONFIG_DIR / f"_runtime_{name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return out_path
