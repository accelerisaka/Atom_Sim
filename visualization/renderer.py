"""
可视化渲染器 — 把 **全部 10 个原子仿真器** 的运行情况压缩进单张图。

面板布局 (3 行 x 4 列)
------------------------------------------------------------
行 1  物理场 (2D imshow, 世界坐标对齐):
  (0,0) TD1  温度场
  (0,1) FL3  烟雾浓度场
  (0,2) FL3  能见度场
  (0,3) FL4  宏观人群密度

行 2  个体空间散点 (walls/exits 叠加):
  (1,0) CM1  主场景 (走/倒/逃/救助 + 速度箭头)
  (1,1) CM1 位置按 PD3 恐慌值着色
  (1,2) CM1 位置按 BP6 认知效率着色
  (1,3) CM1 位置按 BP5 感知能力着色 + S2 从众偏置箭头

行 3  时间序列 / 条形图 / 汇总:
  (2,0) P6  flow_constraint 时序 (含 max 密度对照)
  (2,1) SD4 救助触发计数时序 (逐快照)
  (2,2) PD3 逐个体恐慌条形图 (颜色=级别)
  (2,3) 全局统计文本面板 (逃出/在场/跌倒/救助/最大温度/最大烟雾/平均恐慌)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

FrameSource = Union[str, np.ndarray]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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


_FIRE_CMAP = LinearSegmentedColormap.from_list(
    "fire", ["#000000", "#ff4500", "#ff8c00", "#ffff00", "#ffffff"]
) if HAS_MPL else None

_SMOKE_CMAP = LinearSegmentedColormap.from_list(
    "smoke", ["#ffffff", "#888888", "#333333", "#000000"]
) if HAS_MPL else None

_VIS_CMAP = LinearSegmentedColormap.from_list(
    "vis", ["#0a0a0a", "#446688", "#bbd0e8", "#ffffff"]
) if HAS_MPL else None

_DENS_CMAP = LinearSegmentedColormap.from_list(
    "dens", ["#f7fbff", "#6baed6", "#2171b5", "#08306b", "#330033"]
) if HAS_MPL else None


class SimulationRenderer:
    """把全部原子仿真器状态压到一张图的渲染器。"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 单帧绘制
    # ==================================================================

    def _build_frame_figure(
        self,
        snapshot: Dict[str, Any],
        frame_idx: int,
        world_size: tuple,
        walls: Optional[List],
        exits: Optional[List],
        history_window: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Any:
        fig, axes = plt.subplots(3, 4, figsize=(22, 14))
        t_now = snapshot.get("time", 0.0)
        fig.suptitle(
            f"Atom Co-Simulation Dashboard   t = {t_now:.2f}s   frame #{frame_idx}",
            fontsize=15, fontweight="bold",
        )

        # Row 1 — 物理场
        self._draw_temperature(axes[0, 0], snapshot, world_size)
        self._draw_smoke(axes[0, 1], snapshot, world_size)
        self._draw_visibility(axes[0, 2], snapshot, world_size)
        self._draw_crowd_density(axes[0, 3], snapshot, world_size)

        # Row 2 — 个体散点
        self._draw_agents_main(axes[1, 0], snapshot, world_size, walls, exits)
        self._draw_agents_colored(
            axes[1, 1], snapshot, world_size, walls, exits,
            sim_id="PD3", field="panic_level",
            title="PD3: Panic per agent", cmap="RdYlGn_r",
            vmin=0.0, vmax=1.0, cbar_label="Panic",
        )
        self._draw_agents_colored(
            axes[1, 2], snapshot, world_size, walls, exits,
            sim_id="BP6", field="cognitive_efficiency_multiplier",
            title="BP6: Cognitive efficiency",
            cmap="viridis", vmin=0.0, vmax=1.0, cbar_label="Efficiency",
        )
        self._draw_agents_colored(
            axes[1, 3], snapshot, world_size, walls, exits,
            sim_id="BP5", field="perception_mask",
            title="BP5: Perception mask + S2 herd bias",
            cmap="plasma", vmin=0.0, vmax=1.0, cbar_label="Perception",
            quiver_sim="S2", quiver_field="herd_velocity_bias",
        )

        # Row 3 — 时序 / 条形 / 文本
        self._draw_p6_timeseries(axes[2, 0], history_window, t_now)
        self._draw_sd4_timeseries(axes[2, 1], history_window, t_now)
        self._draw_panic_bars(axes[2, 2], snapshot)
        self._draw_stats_panel(axes[2, 3], snapshot, history_window)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def render_frame_to_array(
        self,
        snapshot: Dict[str, Any],
        frame_idx: int,
        world_size: tuple = (25.0, 25.0),
        walls: Optional[List] = None,
        exits: Optional[List] = None,
        history_window: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[np.ndarray]:
        if not HAS_MPL:
            return None
        fig = self._build_frame_figure(
            snapshot, frame_idx, world_size, walls, exits, history_window
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
        history_window: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        if not HAS_MPL:
            return None
        fig = self._build_frame_figure(
            snapshot, frame_idx, world_size, walls, exits, history_window
        )
        path = self.output_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return str(path)

    # ==================================================================
    # 行 1：物理场
    # ==================================================================
    #
    # 场网格约定：field[row, col] 中 row 对应世界 x (水平)，col 对应世界 y (垂直)。
    # 为与 agent 散点坐标系（x 水平、y 垂直）一致，统一 imshow 时转置 + extent。

    def _imshow_world(
        self, ax: Any, field: np.ndarray, world_size: tuple,
        cmap: Any, vmin: float, vmax: float, cbar_label: str,
    ) -> None:
        extent = [0.0, world_size[0], 0.0, world_size[1]]
        im = ax.imshow(
            field.T, cmap=cmap, origin="lower",
            extent=extent, vmin=vmin, vmax=vmax, aspect="equal",
        )
        ax.set_xlim(0, world_size[0])
        ax.set_ylim(0, world_size[1])
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    def _draw_temperature(self, ax: Any, snap: Dict, world_size: tuple) -> None:
        ax.set_title("TD1: Temperature (°C)")
        td1 = snap.get("TD1", {})
        field = td1.get("temperature_field")
        if field is None:
            ax.text(0.5, 0.5, "no TD1 data", ha="center", va="center", transform=ax.transAxes)
            return
        field = np.asarray(field, dtype=np.float64)
        self._imshow_world(ax, field, world_size, _FIRE_CMAP, 20.0, 800.0, "T (°C)")

    def _draw_smoke(self, ax: Any, snap: Dict, world_size: tuple) -> None:
        ax.set_title("FL3: Smoke density")
        fl3 = snap.get("FL3", {})
        field = fl3.get("smoke_density_field")
        if field is None:
            ax.text(0.5, 0.5, "no FL3 data", ha="center", va="center", transform=ax.transAxes)
            return
        field = np.asarray(field, dtype=np.float64)
        self._imshow_world(ax, field, world_size, _SMOKE_CMAP, 0.0, 1.0, "Density")

    def _draw_visibility(self, ax: Any, snap: Dict, world_size: tuple) -> None:
        ax.set_title("FL3: Visibility field")
        fl3 = snap.get("FL3", {})
        field = fl3.get("visibility_field")
        if field is None:
            ax.text(0.5, 0.5, "no FL3 visibility", ha="center", va="center", transform=ax.transAxes)
            return
        field = np.asarray(field, dtype=np.float64)
        self._imshow_world(ax, field, world_size, _VIS_CMAP, 0.0, 1.0, "Visibility")

    def _draw_crowd_density(self, ax: Any, snap: Dict, world_size: tuple) -> None:
        ax.set_title("FL4: Crowd density (people/m²)")
        fl4 = snap.get("FL4", {})
        field = fl4.get("density_map")
        if field is None:
            ax.text(0.5, 0.5, "no FL4 data", ha="center", va="center", transform=ax.transAxes)
            return
        field = np.asarray(field, dtype=np.float64)
        vmax = max(float(np.max(field)) * 1.1, 1e-3)
        self._imshow_world(ax, field, world_size, _DENS_CMAP, 0.0, vmax, "ρ (1/m²)")

    # ==================================================================
    # 行 2：个体散点
    # ==================================================================

    def _overlay_walls_exits(
        self, ax: Any,
        world_size: tuple,
        walls: Optional[List], exits: Optional[List],
    ) -> None:
        ax.set_xlim(0, world_size[0])
        ax.set_ylim(0, world_size[1])
        ax.set_aspect("equal")
        if walls:
            for w in walls:
                a, b = w
                ax.plot([a[0], b[0]], [a[1], b[1]], "k-", linewidth=2)
        if exits:
            for ex in exits:
                ax.plot(ex[0], ex[1], "g^", markersize=11, zorder=5)

    def _draw_agents_main(
        self, ax: Any, snap: Dict,
        world_size: tuple, walls: Optional[List], exits: Optional[List],
    ) -> None:
        ax.set_title("CM1: Agents + walls/exits + velocity")
        self._overlay_walls_exits(ax, world_size, walls, exits)

        cm1 = snap.get("CM1", {})
        positions = cm1.get("positions")
        velocities = cm1.get("velocities")
        active = cm1.get("active")
        fallen = cm1.get("fallen")
        if positions is None:
            return

        pos = np.asarray(positions)
        vel = np.asarray(velocities) if velocities is not None else np.zeros_like(pos)
        act = np.asarray(active, dtype=bool) if active is not None else np.ones(pos.shape[0], dtype=bool)
        fall = np.asarray(fallen, dtype=bool) if fallen is not None else np.zeros(pos.shape[0], dtype=bool)

        escaped = ~act & ~fall
        walking = act & ~fall
        on_ground = fall

        # 救助者高亮
        sd4 = snap.get("SD4", {})
        helper_ids = sd4.get("helper_ids", []) or []
        helpers = np.zeros(pos.shape[0], dtype=bool)
        for hid in helper_ids:
            if 0 <= int(hid) < helpers.size:
                helpers[int(hid)] = True

        if np.any(walking):
            ax.scatter(pos[walking, 0], pos[walking, 1],
                       c="dodgerblue", s=22, alpha=0.85, label="Walking",
                       edgecolors="none", zorder=3)
            # 速度箭头
            vw = vel[walking]
            ax.quiver(
                pos[walking, 0], pos[walking, 1], vw[:, 0], vw[:, 1],
                angles="xy", scale_units="xy", scale=2.0,
                color="black", alpha=0.4, width=0.003, zorder=4,
            )
        if np.any(on_ground):
            ax.scatter(pos[on_ground, 0], pos[on_ground, 1],
                       c="red", s=55, marker="x", label="Fallen", zorder=4)
        if np.any(escaped):
            ax.scatter(pos[escaped, 0], pos[escaped, 1],
                       c="green", s=18, alpha=0.35, label="Escaped", zorder=2)
        if np.any(helpers & act):
            ax.scatter(pos[helpers & act, 0], pos[helpers & act, 1],
                       facecolors="none", edgecolors="orange", s=140,
                       linewidths=2, label="Helper", zorder=6)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    def _draw_agents_colored(
        self, ax: Any, snap: Dict,
        world_size: tuple, walls: Optional[List], exits: Optional[List],
        sim_id: str, field: str,
        title: str, cmap: str, vmin: float, vmax: float, cbar_label: str,
        quiver_sim: Optional[str] = None,
        quiver_field: Optional[str] = None,
    ) -> None:
        ax.set_title(title)
        self._overlay_walls_exits(ax, world_size, walls, exits)

        cm1 = snap.get("CM1", {})
        positions = cm1.get("positions")
        active = cm1.get("active")
        fallen = cm1.get("fallen")
        values = snap.get(sim_id, {}).get(field)
        if positions is None or values is None:
            ax.text(0.5, 0.5, f"no {sim_id}.{field}", ha="center", va="center", transform=ax.transAxes)
            return

        pos = np.asarray(positions)
        act = np.asarray(active, dtype=bool) if active is not None else np.ones(pos.shape[0], dtype=bool)
        fall = np.asarray(fallen, dtype=bool) if fallen is not None else np.zeros(pos.shape[0], dtype=bool)
        vals = np.asarray(values, dtype=np.float64).ravel()

        if vals.size < pos.shape[0]:
            vals = np.concatenate([vals, np.full(pos.shape[0] - vals.size, np.nan)])
        vals = vals[: pos.shape[0]]

        in_building = act & ~fall
        if np.any(in_building):
            sc = ax.scatter(
                pos[in_building, 0], pos[in_building, 1],
                c=vals[in_building], cmap=cmap, vmin=vmin, vmax=vmax,
                s=35, edgecolors="black", linewidths=0.3, zorder=3,
            )
            plt.colorbar(sc, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
        if np.any(fall):
            ax.scatter(pos[fall, 0], pos[fall, 1],
                       c="red", marker="x", s=50, zorder=4)
        escaped = ~act & ~fall
        if np.any(escaped):
            ax.scatter(pos[escaped, 0], pos[escaped, 1],
                       c="gray", s=14, alpha=0.3, zorder=2)

        # 可选：叠加矢量场
        if quiver_sim and quiver_field:
            vec = snap.get(quiver_sim, {}).get(quiver_field)
            if vec is not None:
                vec = np.asarray(vec, dtype=np.float64)
                if vec.ndim == 2 and vec.shape[0] >= pos.shape[0] and vec.shape[1] == 2:
                    vec = vec[: pos.shape[0]]
                    mask_v = in_building & (np.linalg.norm(vec, axis=1) > 1e-3)
                    if np.any(mask_v):
                        ax.quiver(
                            pos[mask_v, 0], pos[mask_v, 1],
                            vec[mask_v, 0], vec[mask_v, 1],
                            angles="xy", scale_units="xy", scale=1.5,
                            color="white", alpha=0.9, width=0.004, zorder=5,
                            edgecolor="black", linewidth=0.3,
                        )

    # ==================================================================
    # 行 3：时序 / 条形 / 文本
    # ==================================================================

    def _draw_p6_timeseries(
        self, ax: Any,
        history_window: Optional[Sequence[Dict[str, Any]]],
        t_now: float,
    ) -> None:
        ax.set_title("P6: flow_constraint + max crowd density")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("flow_constraint", color="#1f77b4")
        ax.set_ylim(0.0, 1.05)

        ax2 = ax.twinx()
        ax2.set_ylabel("peak ρ (1/m²)", color="#d62728")

        if not history_window:
            return

        times = []
        flows = []
        peaks = []
        for snap in history_window:
            times.append(float(snap.get("time", 0.0)))
            flows.append(float(snap.get("P6", {}).get("flow_constraint", 1.0)))
            density_map = snap.get("FL4", {}).get("density_map")
            if density_map is not None:
                peaks.append(float(np.max(np.asarray(density_map))))
            else:
                peaks.append(0.0)

        if times:
            ax.plot(times, flows, color="#1f77b4", linewidth=1.8, label="flow_constraint")
            ax.fill_between(times, 0.0, flows, color="#1f77b4", alpha=0.15)
            ax2.plot(times, peaks, color="#d62728", linewidth=1.2, linestyle="--", label="peak ρ")
            ax.axvline(t_now, color="gray", linewidth=0.6, linestyle=":")
            x_lo = times[0]
            x_hi = max(times[-1], t_now, x_lo + 1e-3)
            ax.set_xlim(x_lo, x_hi)

    def _draw_sd4_timeseries(
        self, ax: Any,
        history_window: Optional[Sequence[Dict[str, Any]]],
        t_now: float,
    ) -> None:
        ax.set_title("SD4: help triggers + fallen count")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("count")

        if not history_window:
            return

        times: List[float] = []
        fallen_counts: List[int] = []
        help_cum: List[int] = []
        cumulative_helpers: set = set()

        for snap in history_window:
            times.append(float(snap.get("time", 0.0)))
            fallen = snap.get("CM1", {}).get("fallen")
            fallen_counts.append(int(np.sum(np.asarray(fallen, dtype=bool))) if fallen is not None else 0)
            helpers = snap.get("SD4", {}).get("helper_ids", []) or []
            for h in helpers:
                cumulative_helpers.add(int(h))
            help_cum.append(len(cumulative_helpers))

        if times:
            ax.plot(times, fallen_counts, color="crimson", linewidth=1.6, label="fallen (current)")
            ax.fill_between(times, 0, fallen_counts, color="crimson", alpha=0.15)
            ax.plot(times, help_cum, color="darkorange", linewidth=1.6, linestyle="--",
                    label="helpers (cumulative)")
            ax.axvline(t_now, color="gray", linewidth=0.6, linestyle=":")
            x_lo = times[0]
            x_hi = max(times[-1], t_now, x_lo + 1e-3)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(bottom=0)
            ax.legend(loc="upper left", fontsize=8)

    def _draw_panic_bars(self, ax: Any, snap: Dict) -> None:
        ax.set_title("PD3: Panic level bars")
        pd3 = snap.get("PD3", {})
        panic = pd3.get("panic_level")
        if panic is None:
            ax.text(0.5, 0.5, "no PD3", ha="center", va="center", transform=ax.transAxes)
            return
        panic = np.asarray(panic, dtype=np.float64).ravel()
        x = np.arange(len(panic))
        colors = [plt.cm.RdYlGn_r(float(np.clip(v, 0, 1))) for v in panic]
        ax.bar(x, panic, color=colors, width=1.0)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Agent ID")
        ax.set_ylabel("Panic")

    def _draw_stats_panel(
        self, ax: Any,
        snap: Dict,
        history_window: Optional[Sequence[Dict[str, Any]]],
    ) -> None:
        ax.set_title("Global stats")
        ax.axis("off")

        t_now = snap.get("time", 0.0)
        cm1 = snap.get("CM1", {})
        active = np.asarray(cm1.get("active", []), dtype=bool)
        fallen = np.asarray(cm1.get("fallen", []), dtype=bool)
        n_total = int(active.size) if active.size else 0
        n_fallen = int(np.sum(fallen)) if fallen.size else 0
        n_walking = int(np.sum(active & ~fallen)) if active.size else 0
        n_escaped = int(np.sum((~active) & (~fallen))) if active.size else 0

        td1_field = snap.get("TD1", {}).get("temperature_field")
        max_T = float(np.max(np.asarray(td1_field))) if td1_field is not None else 0.0

        fl3_field = snap.get("FL3", {}).get("smoke_density_field")
        max_smoke = float(np.max(np.asarray(fl3_field))) if fl3_field is not None else 0.0
        mean_smoke = float(np.mean(np.asarray(fl3_field))) if fl3_field is not None else 0.0

        panic = snap.get("PD3", {}).get("panic_level")
        mean_panic = float(np.mean(np.asarray(panic))) if panic is not None else 0.0
        max_panic = float(np.max(np.asarray(panic))) if panic is not None else 0.0

        eff = snap.get("BP6", {}).get("cognitive_efficiency_multiplier")
        mean_eff = float(np.mean(np.asarray(eff))) if eff is not None else 1.0

        perc = snap.get("BP5", {}).get("perception_mask")
        mean_perc = float(np.mean(np.asarray(perc))) if perc is not None else 1.0

        flow = snap.get("P6", {}).get("flow_constraint", 1.0)
        herd = snap.get("S2", {}).get("herd_velocity_bias")
        herd_mag = 0.0
        if herd is not None:
            herd_arr = np.asarray(herd)
            if herd_arr.ndim == 2 and herd_arr.shape[1] == 2:
                herd_mag = float(np.mean(np.linalg.norm(herd_arr, axis=1)))

        helpers_now = snap.get("SD4", {}).get("helper_ids", []) or []
        cumulative_helpers: set = set()
        if history_window:
            for s in history_window:
                for h in s.get("SD4", {}).get("helper_ids", []) or []:
                    cumulative_helpers.add(int(h))

        lines = [
            f"time            = {t_now:6.2f} s",
            f"agents total    = {n_total}",
            f"  walking       = {n_walking}",
            f"  fallen        = {n_fallen}",
            f"  escaped       = {n_escaped}",
            "-" * 28,
            f"TD1 max T       = {max_T:6.1f} °C",
            f"FL3 max smoke   = {max_smoke:5.3f}",
            f"FL3 mean smoke  = {mean_smoke:5.3f}",
            "-" * 28,
            f"PD3 mean panic  = {mean_panic:5.3f}",
            f"PD3 max panic   = {max_panic:5.3f}",
            f"BP6 mean eff    = {mean_eff:5.3f}",
            f"BP5 mean perc   = {mean_perc:5.3f}",
            f"S2  |herd|      = {herd_mag:5.3f} m/s",
            "-" * 28,
            f"P6  flow_constr = {float(flow):5.3f}",
            f"SD4 helpers now = {len(helpers_now)}",
            f"SD4 helpers tot = {len(cumulative_helpers)}",
        ]
        ax.text(
            0.02, 0.98, "\n".join(lines),
            family="monospace", fontsize=10,
            va="top", ha="left", transform=ax.transAxes,
        )

    # ==================================================================
    # 批量渲染
    # ==================================================================

    def render_all(
        self,
        history: List[Dict[str, Any]],
        world_size: tuple = (25.0, 25.0),
        walls: Optional[List] = None,
        exits: Optional[List] = None,
    ) -> List[str]:
        paths = []
        for idx, snap in enumerate(history):
            p = self.render_frame(
                snap, idx, world_size, walls, exits,
                history_window=history[: idx + 1],
            )
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
        frames: List[np.ndarray] = []
        for idx, snap in enumerate(history):
            arr = self.render_frame_to_array(
                snap, idx, world_size, walls, exits,
                history_window=history[: idx + 1],
            )
            if arr is not None:
                frames.append(arr)
        return frames

    # ==================================================================
    # GIF / MP4 导出
    # ==================================================================

    def export_gif(
        self,
        frames: List[FrameSource],
        output_name: str = "simulation.gif",
        fps: int = 4,
        scale: float = 0.5,
    ) -> Optional[str]:
        if not HAS_PIL:
            print("[renderer] 缺少 Pillow，无法导出 GIF。")
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
        if not HAS_IMAGEIO:
            print("[renderer] 缺少 imageio，无法导出视频。")
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
                h, w = frame.shape[:2]
                if h % 2 != 0 or w % 2 != 0:
                    frame = frame[: h - h % 2, : w - w % 2]
                writer.append_data(frame)
            writer.close()
        except Exception as exc:
            print(f"[renderer] 视频导出失败: {exc}")
            return None
        return str(out_path)

    # ==================================================================
    # JSON 导出
    # ==================================================================

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
