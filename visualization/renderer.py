"""
可视化渲染器 — 将仿真快照渲染为实时/离线图表。
支持：温度场、烟雾场、人群位置与密度、恐慌热力图。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

FrameSource = Union[str, np.ndarray]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


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

    def _build_frame_figure(
        self,
        snapshot: Dict[str, Any],
        frame_idx: int,
        world_size: tuple,
        walls: Optional[List],
        exits: Optional[List],
    ) -> Any:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Building Fire Evacuation  t = {snapshot.get('time', 0):.2f}s",
                     fontsize=14, fontweight="bold")

        self._draw_temperature(axes[0, 0], snapshot)
        self._draw_smoke(axes[0, 1], snapshot)
        self._draw_agents(axes[1, 0], snapshot, world_size, walls, exits)
        self._draw_panic(axes[1, 1], snapshot)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def render_frame_to_array(
        self,
        snapshot: Dict[str, Any],
        frame_idx: int,
        world_size: tuple = (25.0, 25.0),
        walls: Optional[List] = None,
        exits: Optional[List] = None,
    ) -> Optional[np.ndarray]:
        """渲染单帧为 RGB 数组 (H, W, 3) uint8，不写磁盘。"""
        if not HAS_MPL:
            return None

        fig = self._build_frame_figure(
            snapshot, frame_idx, world_size, walls, exits
        )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[..., :3].copy()

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

        fig = self._build_frame_figure(
            snapshot, frame_idx, world_size, walls, exits
        )
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
        """渲染全部历史帧到 PNG 文件。"""
        paths = []
        for idx, snap in enumerate(history):
            p = self.render_frame(snap, idx, world_size, walls, exits)
            if p:
                paths.append(p)
        return paths

    def render_all_to_arrays(
        self,
        history: List[Dict[str, Any]],
        world_size: tuple = (25.0, 25.0),
        walls: Optional[List] = None,
        exits: Optional[List] = None,
    ) -> List[np.ndarray]:
        """将全部历史帧渲染为内存中的 RGB 数组列表（不写 PNG）。"""
        frames: List[np.ndarray] = []
        for idx, snap in enumerate(history):
            arr = self.render_frame_to_array(snap, idx, world_size, walls, exits)
            if arr is not None:
                frames.append(arr)
        return frames

    # ------------------------------------------------------------------
    # 动图 / 视频导出
    # ------------------------------------------------------------------

    def export_gif(
        self,
        frames: List[FrameSource],
        output_name: str = "simulation.gif",
        fps: int = 4,
        scale: float = 0.5,
    ) -> Optional[str]:
        """将帧列表合成为 GIF 动图。

        Args:
            frames: PNG 路径或 RGB 数组 (H,W,3) uint8 的有序列表。
            output_name: 输出文件名。
            fps: 每秒帧数，控制播放速度。
            scale: 缩放比例（0~1），降低 GIF 文件体积。

        Returns:
            GIF 文件路径，失败返回 None。
        """
        if not HAS_PIL:
            print("[renderer] 缺少 Pillow，无法导出 GIF。请运行: pip install pillow")
            return None
        if not frames:
            return None

        duration_ms = int(1000 / max(fps, 1))
        pil_frames: List[Any] = []
        for item in frames:
            if isinstance(item, (str, Path)):
                img = Image.open(item).convert("RGBA")
            else:
                arr = np.asarray(item)
                img = Image.fromarray(arr).convert("RGBA")
            if scale != 1.0:
                new_w = max(1, int(img.width * scale))
                new_h = max(1, int(img.height * scale))
                img = img.resize((new_w, new_h), Image.LANCZOS)
            pil_frames.append(
                img.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
            )

        out_path = self.output_dir / output_name
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        return str(out_path)

    def export_video(
        self,
        frames: List[FrameSource],
        output_name: str = "simulation.mp4",
        fps: int = 8,
    ) -> Optional[str]:
        """将帧列表合成为 MP4 视频（需要 imageio[ffmpeg]）。

        Args:
            frames: PNG 路径或 RGB 数组 (H,W,3) uint8 的有序列表。
            output_name: 输出文件名。
            fps: 帧率。

        Returns:
            视频文件路径，失败返回 None。
        """
        if not HAS_IMAGEIO:
            print("[renderer] 缺少 imageio，无法导出视频。请运行: pip install imageio[ffmpeg]")
            return None
        if not frames:
            return None

        out_path = self.output_dir / output_name
        try:
            writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                        pixelformat="yuv420p", macro_block_size=8)
            for item in frames:
                if isinstance(item, (str, Path)):
                    frame = imageio.imread(item)
                else:
                    frame = np.asarray(item)
                # yuv420p 要求宽高均为偶数
                h, w = frame.shape[:2]
                if h % 2 != 0 or w % 2 != 0:
                    frame = frame[: h - h % 2, : w - w % 2]
                writer.append_data(frame)
            writer.close()
        except Exception as exc:
            print(f"[renderer] 视频导出失败: {exc}")
            print("  提示：请确认已安装 ffmpeg 并可被 imageio 调用。")
            return None
        return str(out_path)

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
