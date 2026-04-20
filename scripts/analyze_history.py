#!/usr/bin/env python3
"""
对 output/history.json（Orchestrator 导出快照列表）做统计分析。

用法:
  python scripts/analyze_history.py
  python scripts/analyze_history.py -i output/history.json --csv output/history_cm1_stats.csv
"""

from __future__ import annotations

import argparse
import io
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _as_bool_arr(x: Any, n: int) -> np.ndarray:
    if x is None:
        return np.ones(n, dtype=bool)
    a = np.asarray(x, dtype=bool).ravel()
    if a.size < n:
        return np.pad(a, (0, n - a.size), constant_values=True)
    return a[:n]


def _stats_cm1(cm1: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pos = cm1.get("positions")
    if pos is None:
        return None
    positions = np.asarray(pos, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        return None
    n = positions.shape[0]
    vel = cm1.get("velocities")
    if vel is not None:
        v = np.asarray(vel, dtype=np.float64)
        if v.shape == (n, 2):
            speed = np.linalg.norm(v, axis=1)
        else:
            speed = np.zeros(n)
    else:
        speed = np.zeros(n)

    active = _as_bool_arr(cm1.get("active"), n)
    fallen = _as_bool_arr(cm1.get("fallen"), n)

    walking = active & ~fallen
    escaped = ~active & ~fallen
    down = fallen

    return {
        "n_agents": int(n),
        "n_walking": int(np.sum(walking)),
        "n_escaped": int(np.sum(escaped)),
        "n_fallen": int(np.sum(down)),
        "n_active_true": int(np.sum(active)),
        "speed_mean": float(np.mean(speed)),
        "speed_max": float(np.max(speed)),
        "speed_mean_walking": float(np.mean(speed[walking])) if np.any(walking) else 0.0,
        "pos_x_min": float(np.min(positions[:, 0])),
        "pos_x_max": float(np.max(positions[:, 0])),
        "pos_y_min": float(np.min(positions[:, 1])),
        "pos_y_max": float(np.max(positions[:, 1])),
    }


def _scalar_field_stats(field: Any, name: str) -> Dict[str, float]:
    if field is None:
        return {}
    arr = np.asarray(field, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        f"{name}_min": float(np.min(arr)),
        f"{name}_max": float(np.max(arr)),
        f"{name}_mean": float(np.mean(arr)),
    }


def analyze_history(
    history: List[Dict[str, Any]],
    world_size: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    missing_cm1_times: List[float] = []
    cm1_count = 0

    for snap in history:
        t = float(snap.get("time", 0.0))
        row: Dict[str, Any] = {"time": t, "has_cm1": False}

        cm1 = snap.get("CM1")
        if isinstance(cm1, dict):
            st = _stats_cm1(cm1)
            if st:
                row["has_cm1"] = True
                cm1_count += 1
                row.update(st)
                if world_size:
                    wx, wy = world_size
                    row["out_of_bounds"] = bool(
                        st["pos_x_min"] < 0
                        or st["pos_x_max"] > wx
                        or st["pos_y_min"] < 0
                        or st["pos_y_max"] > wy
                    )
        if not row.get("has_cm1"):
            missing_cm1_times.append(t)

        td1 = snap.get("TD1")
        if isinstance(td1, dict) and td1.get("temperature_field") is not None:
            row.update(_scalar_field_stats(td1["temperature_field"], "td1_T"))

        fl3 = snap.get("FL3")
        if isinstance(fl3, dict) and fl3.get("smoke_density") is not None:
            row.update(_scalar_field_stats(fl3["smoke_density"], "fl3_smoke"))

        pd3 = snap.get("PD3")
        if isinstance(pd3, dict) and pd3.get("panic_level") is not None:
            p = np.asarray(pd3["panic_level"], dtype=np.float64).ravel()
            if p.size:
                row["pd3_panic_mean"] = float(np.mean(p))
                row["pd3_panic_max"] = float(np.max(p))

        rows.append(row)

    summary = {
        "n_snapshots": len(history),
        "time_first": float(history[0]["time"]) if history else None,
        "time_last": float(history[-1]["time"]) if history else None,
        "cm1_snapshots": cm1_count,
        "missing_cm1_snapshots": len(missing_cm1_times),
        "missing_cm1_times": missing_cm1_times,
    }
    return rows, summary


def print_report(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print("=== history.json 统计 ===\n")
    print(f"快照总数:        {summary['n_snapshots']}")
    if summary["time_first"] is not None:
        print(f"时间范围:        {summary['time_first']} — {summary['time_last']} s")
    print(f"含 CM1 的快照:   {summary['cm1_snapshots']}")
    print(f"缺失 CM1 的快照: {summary['missing_cm1_snapshots']}")

    if summary["missing_cm1_times"]:
        mt = summary["missing_cm1_times"]
        print(f"  缺失时刻 (前10个): {mt[:10]}{' ...' if len(mt) > 10 else ''}")

    cm1_rows = [r for r in rows if r.get("has_cm1")]
    if not cm1_rows:
        print("\n未找到任何 CM1 有效数据。")
        return

    print("\n--- CM1 终态（最后一条含 CM1 的快照）---")
    last = cm1_rows[-1]
    print(f"  t = {last['time']} s")
    print(f"  人数 n_agents     = {last.get('n_agents')}")
    print(f"  仍在走 walking   = {last.get('n_walking')}")
    print(f"  已撤离 escaped   = {last.get('n_escaped')}")
    print(f"  跌倒 fallen      = {last.get('n_fallen')}")
    print(f"  速度均值(全体)   = {last.get('speed_mean', 0):.6f}")
    print(f"  速度均值(行走)   = {last.get('speed_mean_walking', 0):.6f}")

    print("\n--- CM1 随时间范围 ---")
    walks = [r["n_walking"] for r in cm1_rows]
    esc = [r["n_escaped"] for r in cm1_rows]
    fal = [r["n_fallen"] for r in cm1_rows]
    print(f"  walking:   min={min(walks)} max={max(walks)}")
    print(f"  escaped:   min={min(esc)} max={max(esc)}")
    print(f"  fallen:    min={min(fal)} max={max(fal)}")

    if any(r.get("out_of_bounds") for r in cm1_rows):
        n_oob = sum(1 for r in cm1_rows if r.get("out_of_bounds"))
        print(f"\n警告: {n_oob} 条 CM1 快照出现 out_of_bounds（相对给定 world_size）")

    # 环境场（若有）
    tmax = [r.get("td1_T_max") for r in rows if r.get("td1_T_max") is not None]
    if tmax:
        print("\n--- TD1 温度场 ---")
        print(f"  全局最大 T_max: {max(tmax):.4f}")

    smax = [r.get("fl3_smoke_max") for r in rows if r.get("fl3_smoke_max") is not None]
    if smax:
        print("\n--- FL3 烟雾 ---")
        print(f"  全局 smoke max: {max(smax):.6f}")


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    cm1_rows = [r for r in rows if r.get("has_cm1")]
    if not cm1_rows:
        print("无 CM1 数据，跳过 CSV。", file=sys.stderr)
        return
    keys = sorted({k for r in cm1_rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(cm1_rows)
    print(f"已写入 CM1 时间序列 CSV: {path}")


def main() -> int:
    # Windows 控制台尽量避免中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (io.UnsupportedOperation, AttributeError):
            pass

    parser = argparse.ArgumentParser(description="统计分析联合仿真 history.json")
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("output/history.json"),
        help="JSON 路径（默认 output/history.json）",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="可选：导出含 CM1 的快照为 CSV",
    )
    parser.add_argument(
        "--world-size",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="可选：场地尺寸，用于检测越界，例如 25 25",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"文件不存在: {args.input}", file=sys.stderr)
        return 1

    print(f"加载 {args.input} ...")
    with args.input.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list):
        print("根节点应为 JSON 数组（快照列表）。", file=sys.stderr)
        return 1

    ws: Optional[Tuple[float, float]] = None
    if args.world_size is not None:
        ws = (args.world_size[0], args.world_size[1])

    rows, summary = analyze_history(history, world_size=ws)
    print_report(rows, summary)

    if args.csv:
        write_csv(rows, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
