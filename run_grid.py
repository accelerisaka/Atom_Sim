"""
智能电网负载均衡 联合仿真主入口
================================
加载 config/topology_grid.yaml，注册新原子（HU1/AC1/GR1/MK1/PD7）+ 复用 TD1
作为 TD1，运行 Orchestrator，并基于 orch.history 自定义动态可视化面板。

可视化面板 (2x2)：
  ┌────────────────────────┬────────────────────────┐
  │ 户外温度场 + 家庭分布   │ 逐户负荷 / 设定温度     │
  │ (颜色=温度, 散点=家庭)  │ (柱状=load, 标记=set)   │
  ├────────────────────────┼────────────────────────┤
  │ 总负荷 vs 容量 (时序)   │ 电价 & 平均上调 (时序)  │
  └────────────────────────┴────────────────────────┘
导出 PNG / GIF / MP4 至 output_grid/。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from core.base import AtomicSimulator
from core.orchestrator import Orchestrator
from core.protocol import (
    EventMessage,
    ExchangeStrategy,
    PortAddress,
    S2SConnection,
)
from core.transforms import TRANSFORM_REGISTRY, get_transform

from atoms.physical.td1_heat import HeatConduction
from atoms.physical.hu1_households import HouseholdLayout
from atoms.physical.ac1_ac_load import ACLoadDevice
from atoms.physical.gr1_transformer import GridTransformer
from atoms.social.mk1_price_market import PriceMarket
from atoms.social.pd7_price_appraisal import PriceSensitivityAppraisal


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_grid")


# ------------------------------------------------------------------
# 仿真器工厂
# ------------------------------------------------------------------

_SIM_REGISTRY: Dict[str, type] = {
    "atoms.physical.HeatConduction": HeatConduction,
    "atoms.physical.HouseholdLayout": HouseholdLayout,
    "atoms.physical.ACLoadDevice": ACLoadDevice,
    "atoms.physical.GridTransformer": GridTransformer,
    "atoms.social.PriceMarket": PriceMarket,
    "atoms.social.PriceSensitivityAppraisal": PriceSensitivityAppraisal,
}


def _build_simulator(sim_id: str, cfg: Dict[str, Any]) -> AtomicSimulator:
    cls_name = cfg["class"]
    cls = _SIM_REGISTRY.get(cls_name)
    if cls is None:
        raise ValueError(
            f"Unknown simulator class: {cls_name}. "
            f"Available: {list(_SIM_REGISTRY.keys())}"
        )
    params = cfg.get("params", {})
    for key in ("grid_size", "world_size", "sensitivity_range", "setpoint_range"):
        if key in params and isinstance(params[key], list):
            params[key] = tuple(params[key])
    return cls(sim_id=sim_id, **params)


def _build_connection(cfg: Dict[str, Any]) -> S2SConnection:
    conn_id = cfg["connection_id"]
    transform_name = cfg.get("transform")
    if transform_name is None:
        raise ValueError(
            f"Connection '{conn_id}' missing required field 'transform'."
        )
    transform_fn = get_transform(transform_name)
    return S2SConnection(
        connection_id=conn_id,
        source=PortAddress(cfg["source"]["sim_id"], cfg["source"]["port"]),
        target=PortAddress(cfg["target"]["sim_id"], cfg["target"]["port"]),
        strategy=ExchangeStrategy(cfg["strategy"]),
        transform=transform_fn,
        transform_name=transform_name,
        description=cfg.get("description", ""),
    )


# ------------------------------------------------------------------
# 场景：注入热岛 + 热浪事件
# ------------------------------------------------------------------

def _apply_scenario(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    td1: HeatConduction = orch.simulators.get("TD1")  # type: ignore
    if td1 is None:
        return

    hot_zones = scenario.get("hot_zones", [])
    if hot_zones:
        td1.inputs["ignition_points"] = list(hot_zones)


def _setup_heat_wave_event(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    """在指定时刻注入额外的高温热岛，模拟热浪到来。"""
    hw = scenario.get("heat_wave", {}) or {}
    if not hw.get("enabled", False):
        return
    trigger_time = float(hw.get("trigger_time", 30.0))
    extra_zones = hw.get("extra_zones", [])
    if not extra_zones:
        return

    triggered = {"flag": False}
    td1: HeatConduction = orch.simulators.get("TD1")  # type: ignore
    if td1 is None:
        return

    original_step = td1.step

    def _patched_step(dt: float) -> None:
        original_step(dt)
        if (not triggered["flag"]) and td1.current_time >= trigger_time:
            triggered["flag"] = True
            current = list(td1.inputs.get("ignition_points", []))
            current.extend(list(extra_zones))
            td1.inputs["ignition_points"] = current
            event = EventMessage(
                event_type="HEAT_WAVE",
                source_sim_id="TD1",
                timestamp=td1.current_time,
                payload={"extra_zones": list(extra_zones)},
            )
            orch.inject_event(event)
            logger.info(
                f"  HEAT_WAVE @ t={td1.current_time:.2f}s, "
                f"injected {len(extra_zones)} extra hot zones"
            )

    td1.step = _patched_step  # type: ignore

    def _log_handler(_event: EventMessage) -> None:
        pass  # 事件已在 patched_step 内打印

    orch.register_event_handler("HEAT_WAVE", _log_handler)


# ------------------------------------------------------------------
# 事件桥：GR1 过载边沿 → 注入 OVERLOAD 事件 → MK1 即时反应
# ------------------------------------------------------------------

def _setup_overload_event_bridge(orch: Orchestrator) -> None:
    gr1: GridTransformer = orch.simulators.get("GR1")  # type: ignore
    if gr1 is None:
        return

    original_step = gr1.step

    def _patched_step(dt: float) -> None:
        original_step(dt)
        pulse = gr1.state.get("overload_pulse_ratio", 0.0)
        if pulse > 0.0:
            event = EventMessage(
                event_type="OVERLOAD",
                source_sim_id="GR1",
                timestamp=gr1.current_time,
                payload={"overload_pulse_ratio": float(pulse)},
            )
            orch.inject_event(event)
            logger.info(
                f"  OVERLOAD @ t={gr1.current_time:.2f}s  ratio={pulse:.3f}  "
                f"total_load={gr1.state['total_load']:.2f}kW / cap={gr1.capacity:.1f}kW"
            )

    gr1.step = _patched_step  # type: ignore

    def _handle_overload(event: EventMessage) -> None:
        # MK1 即时消费此脉冲。EVENT 策略的连接已经把 overload_pulse_ratio 写到
        # MK1.inputs["overload_pulse"]；这里仅做一次额外 step 以保证立即响应。
        mk1: PriceMarket = orch.simulators.get("MK1")  # type: ignore
        if mk1 is None:
            return
        mk1.step(0.0)
        mk1.record_output()

    orch.register_event_handler("OVERLOAD", _handle_overload)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main(config_path: str = "config/topology_grid.yaml") -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 1. Orchestrator + 时间分组 ----
    orch = Orchestrator()
    for group_name, group_cfg in config["time_groups"].items():
        orch.register_time_group(group_name, group_cfg["dt"])

    # ---- 2. 仿真器实例化 ----
    for sim_id, sim_cfg in config["simulators"].items():
        sim = _build_simulator(sim_id, sim_cfg)
        orch.register_simulator(sim, sim_cfg["group"])
        logger.info(f"  Registered {sim_id} -> group={sim_cfg['group']}")

    # ---- 3. 连接 ----
    for conn_cfg in config["connections"]:
        conn = _build_connection(conn_cfg)
        orch.add_connection(conn)
    logger.info(f"S2S connections: {len(orch.connections)}")

    loops = orch.causality_guard.detect_algebraic_loops()
    if loops:
        for loop in loops:
            logger.info(f"Detected (intentional) feedback loop: {loop}")

    # ---- 4. 场景 + 事件桥 ----
    scenario = config.get("scenario", {})
    _apply_scenario(orch, scenario)
    _setup_heat_wave_event(orch, scenario)
    _setup_overload_event_bridge(orch)

    logger.info("\n" + orch.summary())

    # ---- 5. 运行 ----
    duration = float(scenario.get("duration", 120.0))
    snap_interval = float(scenario.get("snapshot_interval", 1.0))

    t0 = time.perf_counter()
    orch.run(duration=duration, snapshot_interval=snap_interval)
    elapsed = time.perf_counter() - t0
    logger.info(f"Simulation finished in {elapsed:.2f}s wall-clock, "
                f"{len(orch.history)} snapshots collected.")

    # ---- 6. 可视化 ----
    output_dir = Path("output_grid")
    output_dir.mkdir(exist_ok=True)

    hu1: HouseholdLayout = orch.simulators.get("HU1")  # type: ignore
    td1: HeatConduction = orch.simulators.get("TD1")  # type: ignore
    gr1: GridTransformer = orch.simulators.get("GR1")  # type: ignore

    if hu1 is None or td1 is None or gr1 is None:
        logger.error("Missing core simulators for visualization.")
        return

    static_meta = {
        "positions": hu1.state["positions"],
        "world_size": hu1.world_size,
        "cell_size": td1.cell_size,
        "grid_size": td1.grid_size,
        "capacity": gr1.capacity,
        "num_households": hu1.num_households,
        "base_setpoint": hu1.state["base_setpoint"],
        "price_sensitivity": hu1.state["price_sensitivity"],
    }

    snapshots = orch.history
    if not snapshots:
        logger.warning("No snapshots, skipping visualization.")
        return

    series = _build_time_series(snapshots)
    _render_summary_png(snapshots, series, static_meta, output_dir / "grid_summary.png")
    _render_animation(
        snapshots, series, static_meta,
        gif_path=output_dir / "grid_dynamic.gif",
        mp4_path=output_dir / "grid_dynamic.mp4",
        fps=10,
    )

    # ---- 7. 简短文字摘要 ----
    final = snapshots[-1]
    mk1_out = final.get("MK1", {})
    pd7_out = final.get("PD7", {})
    gr1_out = final.get("GR1", {})
    logger.info("=" * 60)
    logger.info(f"Final t={final['time']:.2f}s")
    logger.info(f"  total_load     = {gr1_out.get('total_load', 0):.2f} kW "
                f"(cap={gr1_out.get('capacity', 0):.1f} kW, "
                f"ρ={gr1_out.get('overload_ratio', 0):.3f})")
    logger.info(f"  electricity_price = {mk1_out.get('electricity_price', 0):.3f} yuan/kWh "
                f"(x{mk1_out.get('price_multiplier', 1):.2f})")
    logger.info(f"  avg setpoint uplift = {pd7_out.get('avg_uplift', 0):.2f} °C")


# ==================================================================
# 可视化辅助
# ==================================================================

def _build_time_series(snapshots: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    times = [s["time"] for s in snapshots]
    total_load, ratio, price, avg_uplift, avg_ac_diff, n_on = [], [], [], [], [], []
    for s in snapshots:
        gr1 = s.get("GR1", {}) or {}
        mk1 = s.get("MK1", {}) or {}
        pd7 = s.get("PD7", {}) or {}
        ac1 = s.get("AC1", {}) or {}
        total_load.append(float(gr1.get("total_load", 0.0)))
        ratio.append(float(gr1.get("overload_ratio", 0.0)))
        price.append(float(mk1.get("electricity_price", 0.0)))
        avg_uplift.append(float(pd7.get("avg_uplift", 0.0)))
        avg_ac_diff.append(float(ac1.get("avg_temp_diff", 0.0)))
        ac_on = ac1.get("ac_on", [])
        n_on.append(int(np.sum(np.asarray(ac_on, dtype=bool))) if len(ac_on) else 0)
    return {
        "time": np.asarray(times, dtype=np.float64),
        "total_load": np.asarray(total_load),
        "overload_ratio": np.asarray(ratio),
        "price": np.asarray(price),
        "avg_uplift": np.asarray(avg_uplift),
        "avg_ac_diff": np.asarray(avg_ac_diff),
        "n_on": np.asarray(n_on),
    }


def _setup_axes(fig, static: Dict[str, Any], series: Dict[str, np.ndarray]):
    """构造 2x2 面板，返回所需的 artists / axes 字典。"""
    axes = fig.subplots(2, 2)
    ax_field, ax_bars, ax_load, ax_price = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    H, W = static["grid_size"]
    cell = static["cell_size"]
    extent = (0, W * cell, H * cell, 0)  # (xmin, xmax, ymax, ymin) ：和 row=y 一致

    # —— 子图 1：温度场 + 家庭散点 ——
    ax_field.set_title("户外温度场 + 家庭分布", pad=22, fontsize=10)
    ax_field.set_xlabel("x (m)")
    ax_field.set_ylabel("y (m)")
    init_field = np.full((H, W), 30.0)
    im = ax_field.imshow(
        init_field, origin="upper", cmap="hot", vmin=28.0, vmax=55.0,
        extent=extent, aspect="equal", interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax_field, fraction=0.046, pad=0.04, label="°C")

    pos = static["positions"]
    sc = ax_field.scatter(
        pos[:, 1], pos[:, 0],
        s=25, c="cyan", edgecolors="black", linewidths=0.7,
        zorder=5, label="家庭",
    )
    title_field = ax_field.text(
        0.02, 1.04, "", transform=ax_field.transAxes,
        fontsize=9, fontweight="bold", color="#333333",
    )

    # —— 子图 2：逐户负荷柱状图 + 设定温度 ——
    n = static["num_households"]
    base_sp = static["base_setpoint"]
    ax_bars.set_title("逐户瞬时负荷 (kW) + 设定温度 (右轴)")
    ax_bars.set_xlabel("Household ID")
    ax_bars.set_ylabel("Load (kW)")
    bar_x = np.arange(n)
    bars = ax_bars.bar(
        bar_x, np.zeros(n), color="#4C9AFF", edgecolor="#1F4E79", width=0.8
    )
    ax_bars.set_ylim(0, 3.5)
    ax_bars.set_xlim(-0.6, n - 0.4)
    ax_bars.axhline(0, color="gray", linewidth=0.5)

    ax_bars2 = ax_bars.twinx()
    ax_bars2.set_ylabel("Setpoint (°C)")
    sp_line, = ax_bars2.plot(
        bar_x, base_sp, "o-", color="#E07B00", markersize=5, linewidth=1.2,
        label="setpoint"
    )
    ax_bars2.set_ylim(22.0, 32.0)
    bp_line, = ax_bars2.plot(
        bar_x, base_sp, "x", color="#999999", markersize=5,
        label="base_setpoint"
    )
    ax_bars2.legend(loc="upper right", fontsize=8)

    # —— 子图 3：总负荷时序 ——
    cap = static["capacity"]
    ax_load.set_title("总负荷 vs 变压器容量")
    ax_load.set_xlabel("time (s)")
    ax_load.set_ylabel("Total load (kW)")
    ax_load.set_xlim(series["time"][0], series["time"][-1])
    y_max = max(series["total_load"].max() * 1.1, cap * 1.3, 1.0)
    ax_load.set_ylim(0, y_max)
    ax_load.axhline(cap, color="red", linestyle="--", linewidth=1.2,
                    label=f"capacity={cap:.1f}kW")
    ax_load.fill_between(
        series["time"], cap, y_max,
        where=series["total_load"] > cap,
        color="red", alpha=0.10, step="post",
    )
    (load_line,) = ax_load.plot([], [], color="#1F4E79", linewidth=1.6, label="total load")
    cursor_load = ax_load.axvline(series["time"][0], color="black", linewidth=0.5, alpha=0.6)
    ax_load.legend(loc="upper left", fontsize=8)

    # —— 子图 4：电价 + 平均上调 ——
    ax_price.set_title("电价 (主轴) & 平均上调温度 (副轴)")
    ax_price.set_xlabel("time (s)")
    ax_price.set_ylabel("Price (yuan/kWh)", color="#B63A2A")
    ax_price.set_xlim(series["time"][0], series["time"][-1])
    p_max = max(series["price"].max() * 1.15, 1.0)
    ax_price.set_ylim(0, p_max)
    (price_line,) = ax_price.plot([], [], color="#B63A2A", linewidth=1.7, label="price")
    ax_price.tick_params(axis="y", labelcolor="#B63A2A")
    cursor_price = ax_price.axvline(series["time"][0], color="black", linewidth=0.5, alpha=0.6)

    ax_price2 = ax_price.twinx()
    ax_price2.set_ylabel("Avg uplift (°C)", color="#2F7A48")
    u_max = max(series["avg_uplift"].max() * 1.3, 0.5)
    ax_price2.set_ylim(0, u_max)
    (uplift_line,) = ax_price2.plot(
        [], [], color="#2F7A48", linewidth=1.4, linestyle="-.", label="avg uplift"
    )
    ax_price2.tick_params(axis="y", labelcolor="#2F7A48")

    legend_elems = [
        Patch(facecolor="#B63A2A", label="electricity_price"),
        Patch(facecolor="#2F7A48", label="avg setpoint uplift"),
    ]
    ax_price.legend(handles=legend_elems, loc="upper left", fontsize=8)

    return {
        "im": im,
        "sc": sc,
        "title_field": title_field,
        "bars": bars,
        "sp_line": sp_line,
        "bp_line": bp_line,
        "load_line": load_line,
        "price_line": price_line,
        "uplift_line": uplift_line,
        "cursor_load": cursor_load,
        "cursor_price": cursor_price,
        "axes": axes,
    }


def _update_frame(idx: int, snapshots, series, static, artists):
    s = snapshots[idx]
    t = s["time"]
    td1 = s.get("TD1", {}) or {}
    ac1 = s.get("AC1", {}) or {}
    pd7 = s.get("PD7", {}) or {}
    gr1 = s.get("GR1", {}) or {}
    mk1 = s.get("MK1", {}) or {}

    # —— 1. 温度场 + 家庭散点 ——
    field = td1.get("temperature_field")
    if field is not None:
        artists["im"].set_data(np.asarray(field))

    pos = static["positions"]  # [N, 2]
    loads = np.asarray(ac1.get("power_load", np.zeros(static["num_households"])))
    on = np.asarray(ac1.get("ac_on", np.zeros(static["num_households"], dtype=bool)))

    sizes = 25 + 80 * np.clip(loads / 3.0, 0.0, 1.0)
    colors = np.where(on, "#FFD24A", "#7AC9FF")  # 黄=AC运行, 蓝=待机
    artists["sc"].set_offsets(np.column_stack([pos[:, 1], pos[:, 0]]))
    artists["sc"].set_sizes(sizes)
    artists["sc"].set_color(colors)

    n_on = int(np.sum(on))
    avg_T = float(np.asarray(field).mean()) if field is not None else 0.0
    artists["title_field"].set_text(
        f"t={t:6.2f}s   T_avg={avg_T:5.2f}°C   AC_on={n_on}/{static['num_households']}"
    )

    # —— 2. 柱状图 + setpoint ——
    for bar, h in zip(artists["bars"], loads):
        bar.set_height(h)
        bar.set_color("#FF8A4C" if h > 1.5 else ("#4C9AFF" if h > 0 else "#CFCFCF"))
    sp = np.asarray(pd7.get("setpoint", static["base_setpoint"]))
    artists["sp_line"].set_ydata(sp)
    artists["bp_line"].set_ydata(static["base_setpoint"])

    # —— 3. 总负荷时序 ——
    times = series["time"][: idx + 1]
    artists["load_line"].set_data(times, series["total_load"][: idx + 1])
    artists["cursor_load"].set_xdata([t, t])

    # —— 4. 电价 + 平均上调 ——
    artists["price_line"].set_data(times, series["price"][: idx + 1])
    artists["uplift_line"].set_data(times, series["avg_uplift"][: idx + 1])
    artists["cursor_price"].set_xdata([t, t])

    return (
        artists["im"],
        artists["sc"],
        artists["title_field"],
        *artists["bars"],
        artists["sp_line"],
        artists["bp_line"],
        artists["load_line"],
        artists["price_line"],
        artists["uplift_line"],
        artists["cursor_load"],
        artists["cursor_price"],
    )


def _render_summary_png(snapshots, series, static, out_path: Path) -> None:
    """渲染最后一帧静态图，作为 summary 预览。"""
    fig = plt.figure(figsize=(14, 9))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS",
                                        "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig.suptitle(
        "Smart Grid Load Balancing — 联合仿真总览（最后一帧）",
        fontsize=14, fontweight="bold",
    )
    artists = _setup_axes(fig, static, series)
    _update_frame(len(snapshots) - 1, snapshots, series, static, artists)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Summary PNG saved: {out_path}")


def _render_animation(
    snapshots,
    series,
    static,
    gif_path: Path,
    mp4_path: Path,
    fps: int = 10,
) -> None:
    """生成动画并导出 GIF + MP4。"""
    fig = plt.figure(figsize=(14, 9))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS",
                                        "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig.suptitle(
        "Smart Grid Load Balancing — 联合仿真动态过程",
        fontsize=14, fontweight="bold",
    )
    artists = _setup_axes(fig, static, series)

    n_frames = len(snapshots)

    def _frame(i: int):
        return _update_frame(i, snapshots, series, static, artists)

    anim = FuncAnimation(
        fig, _frame, frames=n_frames, interval=1000 / fps, blit=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # GIF 导出
    try:
        writer_gif = PillowWriter(fps=fps)
        anim.save(str(gif_path), writer=writer_gif, dpi=90)
        logger.info(f"GIF saved: {gif_path}")
    except Exception as e:
        logger.warning(f"GIF export failed: {e}")

    # MP4 导出 (优先 ffmpeg, 失败则尝试 imageio)
    mp4_ok = False
    try:
        from matplotlib.animation import FFMpegWriter
        writer_mp4 = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400)
        anim.save(str(mp4_path), writer=writer_mp4, dpi=110)
        mp4_ok = True
        logger.info(f"MP4 saved (ffmpeg): {mp4_path}")
    except Exception as e:
        logger.warning(f"FFMpegWriter failed ({e}), trying imageio fallback...")

    if not mp4_ok:
        try:
            import imageio.v2 as imageio  # type: ignore
            frames = []
            for i in range(n_frames):
                _frame(i)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                frames.append(buf)
            imageio.mimsave(str(mp4_path), frames, fps=fps, codec="libx264")
            logger.info(f"MP4 saved (imageio): {mp4_path}")
        except Exception as e:
            logger.warning(f"imageio MP4 fallback also failed: {e}")

    plt.close(fig)


# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Grid Load Balancing Co-Simulation")
    parser.add_argument(
        "-c", "--config", default="config/topology_grid.yaml",
        help="Path to topology YAML config",
    )
    args = parser.parse_args()
    main(args.config)
