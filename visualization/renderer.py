"""
可视化渲染器 — 将仿真快照渲染为实时/离线图表。
支持：温度场、烟雾场、人群位置与密度、恐慌热力图。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# 自定义配色
_FIRE_CMAP = LinearSegmentedColormap.from_list(
    "fire", ["#000000", "#ff4500", "#ff8c00", "#ffff00", "#ffffff"]
) if HAS_MPL else None

_SMOKE_CMAP = LinearSegmentedColormap.from_list(
    "smoke", ["#ffffff", "#888888", "#333333", "#000000"]
) if HAS_MPL else None


class SimulationRenderer:
    """仿真可视化渲染器。"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 单帧渲染
    # ------------------------------------------------------------------

    def render_frame(
        self,
        snapshot: Dict[str, Any],
        frame_idx: int,
        world_size: tuple = (25.0, 25.0),
        walls: Optional[List] = None,
        exits: Optional[List] = None,
    ) -> Optional[str]:
        """渲染单帧四面板图。返回保存路径。"""
        if not HAS_MPL:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Building Fire Evacuation  t = {snapshot.get('time', 0):.2f}s",
                     fontsize=14, fontweight="bold")

        self._draw_temperature(axes[0, 0], snapshot)
        self._draw_smoke(axes[0, 1], snapshot)
        self._draw_agents(axes[1, 0], snapshot, world_size, walls, exits)
        self._draw_panic(axes[1, 1], snapshot)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = self.output_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return str(path)

    # ------------------------------------------------------------------
    # 面板绘制
    # ------------------------------------------------------------------

    def _draw_temperature(self, ax: Any, snap: Dict) -> None:
        td1 = snap.get("TD1", {})
        field = td1.get("temperature_field")
        if field is not None:
            field = np.asarray(field)
            im = ax.imshow(field, cmap=_FIRE_CMAP, origin="lower",
                           vmin=20, vmax=800, aspect="equal")
            plt.colorbar(im, ax=ax, label="Temperature (°C)")
        ax.set_title("TD1: Temperature Field")

    def _draw_smoke(self, ax: Any, snap: Dict) -> None:
        fl3 = snap.get("FL3", {})
        field = fl3.get("smoke_density_field")
        if field is not None:
            field = np.asarray(field)
            im = ax.imshow(field, cmap=_SMOKE_CMAP, origin="lower",
                           vmin=0, vmax=1, aspect="equal")
            plt.colorbar(im, ax=ax, label="Smoke Density")
        ax.set_title("FL3: Smoke Density")

    def _draw_agents(
        self, ax: Any, snap: Dict,
        world_size: tuple,
        walls: Optional[List],
        exits: Optional[List],
    ) -> None:
        ax.set_xlim(0, world_size[0])
        ax.set_ylim(0, world_size[1])
        ax.set_aspect("equal")
        ax.set_title("CM1: Agent Positions")

        if walls:
            for w in walls:
                a, b = w
                ax.plot([a[0], b[0]], [a[1], b[1]], "k-", linewidth=2)

        if exits:
            for ex in exits:
                ax.plot(ex[0], ex[1], "g^", markersize=12, zorder=5)

        cm1 = snap.get("CM1", {})
        positions = cm1.get("positions")
        active = cm1.get("active")
        fallen = cm1.get("fallen")

        if positions is not None:
            pos = np.asarray(positions)
            act = np.asarray(active) if active is not None else np.ones(pos.shape[0], dtype=bool)
            fall = np.asarray(fallen) if fallen is not None else np.zeros(pos.shape[0], dtype=bool)

            escaped = ~act & ~fall
            walking = act & ~fall
            on_ground = fall

            if np.any(walking):
                ax.scatter(pos[walking, 0], pos[walking, 1],
                           c="dodgerblue", s=20, alpha=0.8, label="Walking")
            if np.any(on_ground):
                ax.scatter(pos[on_ground, 0], pos[on_ground, 1],
                           c="red", s=40, marker="x", label="Fallen")
            if np.any(escaped):
                ax.scatter(pos[escaped, 0], pos[escaped, 1],
                           c="green", s=15, alpha=0.3, label="Escaped")
            ax.legend(loc="upper right", fontsize=8)

    def _draw_panic(self, ax: Any, snap: Dict) -> None:
        pd3 = snap.get("PD3", {})
        panic = pd3.get("panic_level")
        if panic is not None:
            panic = np.asarray(panic, dtype=np.float64).ravel()
            x = np.arange(len(panic))
            colors = [plt.cm.RdYlGn_r(float(v)) for v in panic]
            ax.bar(x, panic, color=colors, width=1.0)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Agent ID")
            ax.set_ylabel("Panic Level")
        ax.set_title("PD3: Panic Levels")

    # ------------------------------------------------------------------
    # 批量渲染
    # ------------------------------------------------------------------

    def render_all(
        self,
        history: List[Dict[str, Any]],
        world_size: tuple = (25.0, 25.0),
        walls: Optional[List] = None,
        exits: Optional[List] = None,
    ) -> List[str]:
        """渲染全部历史帧。"""
        paths = []
        for idx, snap in enumerate(history):
            p = self.render_frame(snap, idx, world_size, walls, exits)
            if p:
                paths.append(p)
        return paths

    # ------------------------------------------------------------------
    # JSON 导出（给前端/外部工具用）
    # ------------------------------------------------------------------

    @staticmethod
    def export_history_json(
        history: List[Dict[str, Any]], filepath: str = "output/history.json"
    ) -> str:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=_json_default)
        return filepath


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
