"""
冬季寒潮 + 新能源车晚高峰充电 联合仿真主入口
================================================
加载 config/topology_coldwave_ev.yaml，运行 Orchestrator，并动态可视化。

可视化面板 (2x2)：
  ┌────────────────────────┬────────────────────────┐
  │ 户外温度场 + 家庭分布   │ 逐户负荷 (取暖/EV 堆叠) │
  ├────────────────────────┼────────────────────────┤
  │ 总负荷 vs 容量 (时序)   │ 电价 & EV 削减比例      │
  └────────────────────────┴────────────────────────┘
导出 PNG / GIF / MP4 至 output_coldwave_ev/。
"""

from __future__ import annotations

import argparse
import logging
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
from core.transforms import get_transform

from atoms.physical.td1_heat import HeatConduction
from atoms.physical.hu2_winter_households import HouseholdWinterProfile
from atoms.physical.eh1_heating_load import ElectricHeatingLoad
from atoms.physical.ev1_charging_load import EVChargingLoad
from atoms.physical.la1_load_aggregator import LoadAggregator
from atoms.physical.gr1_transformer import GridTransformer
from atoms.social.mk1_price_market import PriceMarket
from atoms.social.pd8_demand_response import PeakLoadDemandResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_coldwave_ev")


_SIM_REGISTRY: Dict[str, type] = {
    "atoms.physical.HeatConduction": HeatConduction,
    "atoms.physical.HouseholdWinterProfile": HouseholdWinterProfile,
    "atoms.physical.ElectricHeatingLoad": ElectricHeatingLoad,
    "atoms.physical.EVChargingLoad": EVChargingLoad,
    "atoms.physical.LoadAggregator": LoadAggregator,
    "atoms.physical.GridTransformer": GridTransformer,
    "atoms.social.PriceMarket": PriceMarket,
    "atoms.social.PeakLoadDemandResponse": PeakLoadDemandResponse,
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
    for key in ("grid_size", "world_size", "sensitivity_range", "heating_setpoint_range",
                "heating_priority_range", "ev_flexibility_range"):
        if key in params and isinstance(params[key], list):
            params[key] = tuple(params[key])
    return cls(sim_id=sim_id, **params)


def _build_connection(cfg: Dict[str, Any]) -> S2SConnection:
    conn_id = cfg["connection_id"]
    transform_name = cfg.get("transform")
    if transform_name is None:
        raise ValueError(f"Connection '{conn_id}' missing required field 'transform'.")
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


def _apply_cold_zones(td1: HeatConduction, zones: List) -> None:
    """在 TD1 上登记持续冷源锚点（由 patched step 以 min 方式施加）。"""
    if zones:
        td1.state["cold_zones"] = list(zones)


def _apply_scenario(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    td1: HeatConduction = orch.simulators.get("TD1")  # type: ignore
    if td1 is None:
        return
    cold_zones = scenario.get("cold_zones", [])
    _apply_cold_zones(td1, cold_zones)


def _setup_td1_cold_physics(orch: Orchestrator) -> None:
    """补丁 TD1.step：在扩散后施加冷源 min(T, T_cold)，不修改 atoms 源码。"""
    td1: HeatConduction = orch.simulators.get("TD1")  # type: ignore
    if td1 is None:
        return

    original_step = td1.step

    def _patched_step(dt: float) -> None:
        original_step(dt)
        T = td1.state["temperature_field"]
        for pt in td1.state.get("cold_zones", []):
            r, c = int(pt[0]), int(pt[1])
            t_cold = float(pt[2]) if len(pt) > 2 else -12.0
            if 0 <= r < T.shape[0] and 0 <= c < T.shape[1]:
                T[r, c] = min(T[r, c], t_cold)

    td1.step = _patched_step  # type: ignore


def _setup_cold_snap_event(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    snap = scenario.get("cold_snap", {}) or {}
    if not snap.get("enabled", False):
        return
    trigger_time = float(snap.get("trigger_time", 15.0))
    extra_zones = snap.get("extra_zones", [])
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
            current = list(td1.state.get("cold_zones", []))
            current.extend(list(extra_zones))
            td1.state["cold_zones"] = current
            event = EventMessage(
                event_type="COLD_SNAP",
                source_sim_id="TD1",
                timestamp=td1.current_time,
                payload={"extra_zones": list(extra_zones)},
            )
            orch.inject_event(event)
            logger.info(
                f"  COLD_SNAP @ t={td1.current_time:.2f}s, "
                f"injected {len(extra_zones)} extra cold zones"
            )

    td1.step = _patched_step  # type: ignore
    orch.register_event_handler("COLD_SNAP", lambda _e: None)


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
        mk1: PriceMarket = orch.simulators.get("MK1")  # type: ignore
        if mk1 is None:
            return
        mk1.step(0.0)
        mk1.record_output()

    orch.register_event_handler("OVERLOAD", _handle_overload)


def main(config_path: str = "config/topology_coldwave_ev.yaml") -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    orch = Orchestrator()
    for group_name, group_cfg in config["time_groups"].items():
        orch.register_time_group(group_name, group_cfg["dt"])

    for sim_id, sim_cfg in config["simulators"].items():
        sim = _build_simulator(sim_id, sim_cfg)
        orch.register_simulator(sim, sim_cfg["group"])
        logger.info(f"  Registered {sim_id} -> group={sim_cfg['group']}")

    for conn_cfg in config["connections"]:
        conn = _build_connection(conn_cfg)
        orch.add_connection(conn)
    logger.info(f"S2S connections: {len(orch.connections)}")

    loops = orch.causality_guard.detect_algebraic_loops()
    if loops:
        for loop in loops:
            logger.info(f"Detected (intentional) feedback loop: {loop}")

    scenario = config.get("scenario", {})
    _apply_scenario(orch, scenario)
    _setup_td1_cold_physics(orch)
    _setup_cold_snap_event(orch, scenario)
    _setup_overload_event_bridge(orch)

    logger.info("\n" + orch.summary())

    duration = float(scenario.get("duration", 120.0))
    snap_interval = float(scenario.get("snapshot_interval", 1.0))

    t0 = time.perf_counter()
    orch.run(duration=duration, snapshot_interval=snap_interval)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Simulation finished in {elapsed:.2f}s wall-clock, "
        f"{len(orch.history)} snapshots collected."
    )

    output_dir = Path("output_coldwave_ev")
    output_dir.mkdir(exist_ok=True)

    hu2: HouseholdWinterProfile = orch.simulators.get("HU2")  # type: ignore
    td1: HeatConduction = orch.simulators.get("TD1")  # type: ignore
    gr1: GridTransformer = orch.simulators.get("GR1")  # type: ignore

    if hu2 is None or td1 is None or gr1 is None:
        logger.error("Missing core simulators for visualization.")
        return

    static_meta = {
        "positions": hu2.state["positions"],
        "world_size": hu2.world_size,
        "cell_size": td1.cell_size,
        "grid_size": td1.grid_size,
        "capacity": gr1.capacity,
        "num_households": hu2.num_households,
        "heating_setpoint": hu2.state["heating_setpoint"],
        "has_ev": hu2.state["has_ev"],
    }

    snapshots = orch.history
    if not snapshots:
        logger.warning("No snapshots, skipping visualization.")
        return

    series = _build_time_series(snapshots)
    _render_summary_png(snapshots, series, static_meta, output_dir / "coldwave_ev_summary.png")
    _render_animation(
        snapshots, series, static_meta,
        gif_path=output_dir / "coldwave_ev_dynamic.gif",
        mp4_path=output_dir / "coldwave_ev_dynamic.mp4",
        fps=10,
    )

    final = snapshots[-1]
    logger.info("=" * 60)
    logger.info(f"Final t={final['time']:.2f}s")
    gr1_out = final.get("GR1", {}) or {}
    mk1_out = final.get("MK1", {}) or {}
    pd8_out = final.get("PD8", {}) or {}
    la1_out = final.get("LA1", {}) or {}
    logger.info(
        f"  total_load = {gr1_out.get('total_load', 0):.2f} kW "
        f"(heat={la1_out.get('total_heating', 0):.2f}, "
        f"ev={la1_out.get('total_ev', 0):.2f}, "
        f"ρ={gr1_out.get('overload_ratio', 0):.3f})"
    )
    logger.info(
        f"  electricity_price = {mk1_out.get('electricity_price', 0):.3f} yuan/kWh "
        f"(x{mk1_out.get('price_multiplier', 1):.2f})"
    )
    logger.info(f"  avg EV charge limit = {pd8_out.get('avg_ev_limit', 1):.3f}")


def _build_time_series(snapshots: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    times, total_load, ratio, price, avg_ev_limit = [], [], [], [], []
    total_heat, total_ev, n_heat_on, n_ev_on = [], [], [], []
    for s in snapshots:
        gr1 = s.get("GR1", {}) or {}
        mk1 = s.get("MK1", {}) or {}
        pd8 = s.get("PD8", {}) or {}
        la1 = s.get("LA1", {}) or {}
        eh1 = s.get("EH1", {}) or {}
        ev1 = s.get("EV1", {}) or {}
        times.append(s["time"])
        total_load.append(float(gr1.get("total_load", 0.0)))
        ratio.append(float(gr1.get("overload_ratio", 0.0)))
        price.append(float(mk1.get("electricity_price", 0.0)))
        avg_ev_limit.append(float(pd8.get("avg_ev_limit", 1.0)))
        total_heat.append(float(la1.get("total_heating", 0.0)))
        total_ev.append(float(la1.get("total_ev", 0.0)))
        heat_on = eh1.get("heating_on", [])
        ev_on = ev1.get("charging", [])
        n_heat_on.append(int(np.sum(np.asarray(heat_on, dtype=bool))) if len(heat_on) else 0)
        n_ev_on.append(int(np.sum(np.asarray(ev_on, dtype=bool))) if len(ev_on) else 0)
    return {
        "time": np.asarray(times, dtype=np.float64),
        "total_load": np.asarray(total_load),
        "overload_ratio": np.asarray(ratio),
        "price": np.asarray(price),
        "avg_ev_limit": np.asarray(avg_ev_limit),
        "total_heating": np.asarray(total_heat),
        "total_ev": np.asarray(total_ev),
        "n_heat_on": np.asarray(n_heat_on),
        "n_ev_on": np.asarray(n_ev_on),
    }


def _setup_axes(fig, static: Dict[str, Any], series: Dict[str, np.ndarray]):
    axes = fig.subplots(2, 2)
    ax_field, ax_bars, ax_load, ax_price = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    H, W = static["grid_size"]
    cell = static["cell_size"]
    extent = (0, W * cell, H * cell, 0)

    ax_field.set_title("户外温度场 + 家庭分布", pad=22, fontsize=10)
    ax_field.set_xlabel("x (m)")
    ax_field.set_ylabel("y (m)")
    init_field = np.full((H, W), -8.0)
    im = ax_field.imshow(
        init_field, origin="upper", cmap="coolwarm", vmin=-20.0, vmax=5.0,
        extent=extent, aspect="equal", interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax_field, fraction=0.046, pad=0.04, label="°C")

    pos = static["positions"]
    sc = ax_field.scatter(
        pos[:, 1], pos[:, 0],
        s=25, c="lime", edgecolors="black", linewidths=0.7, zorder=5,
    )
    title_field = ax_field.text(
        0.02, 1.04, "", transform=ax_field.transAxes,
        fontsize=9, fontweight="bold", color="#333333",
    )

    n = static["num_households"]
    ax_bars.set_title("逐户负荷 (取暖 + EV 堆叠, kW)")
    ax_bars.set_xlabel("Household ID")
    ax_bars.set_ylabel("Load (kW)")
    bar_x = np.arange(n)
    bars_heat = ax_bars.bar(bar_x, np.zeros(n), color="#E85D4C", edgecolor="#8B2E22", width=0.8, label="heating")
    bars_ev = ax_bars.bar(
        bar_x, np.zeros(n), bottom=np.zeros(n),
        color="#4C7AFF", edgecolor="#1F3A8B", width=0.8, label="EV",
    )
    ax_bars.set_ylim(0, 12.0)
    ax_bars.set_xlim(-0.6, n - 0.4)
    ax_bars.legend(loc="upper right", fontsize=8)

    cap = static["capacity"]
    ax_load.set_title("总负荷 vs 变压器容量")
    ax_load.set_xlabel("time (s)")
    ax_load.set_ylabel("Total load (kW)")
    ax_load.set_xlim(series["time"][0], series["time"][-1])
    y_max = max(series["total_load"].max() * 1.1, cap * 1.3, 1.0)
    ax_load.set_ylim(0, y_max)
    ax_load.axhline(cap, color="red", linestyle="--", linewidth=1.2,
                    label=f"capacity={cap:.1f}kW")
    (load_line,) = ax_load.plot([], [], color="#1F4E79", linewidth=1.6, label="total")
    (heat_line,) = ax_load.plot([], [], color="#E85D4C", linewidth=1.2, linestyle="--", label="heating")
    (ev_line,) = ax_load.plot([], [], color="#4C7AFF", linewidth=1.2, linestyle=":", label="EV")
    cursor_load = ax_load.axvline(series["time"][0], color="black", linewidth=0.5, alpha=0.6)
    ax_load.legend(loc="upper left", fontsize=7)

    ax_price.set_title("电价 (主轴) & 平均 EV 充电上限 (副轴)")
    ax_price.set_xlabel("time (s)")
    ax_price.set_ylabel("Price (yuan/kWh)", color="#B63A2A")
    ax_price.set_xlim(series["time"][0], series["time"][-1])
    p_max = max(series["price"].max() * 1.15, 1.0)
    ax_price.set_ylim(0, p_max)
    (price_line,) = ax_price.plot([], [], color="#B63A2A", linewidth=1.7)
    ax_price.tick_params(axis="y", labelcolor="#B63A2A")
    cursor_price = ax_price.axvline(series["time"][0], color="black", linewidth=0.5, alpha=0.6)

    ax_price2 = ax_price.twinx()
    ax_price2.set_ylabel("Avg EV limit", color="#2F5A7A")
    ax_price2.set_ylim(0, 1.05)
    (limit_line,) = ax_price2.plot([], [], color="#2F5A7A", linewidth=1.4, linestyle="-.")
    ax_price2.tick_params(axis="y", labelcolor="#2F5A7A")

    legend_elems = [
        Patch(facecolor="#B63A2A", label="electricity_price"),
        Patch(facecolor="#2F5A7A", label="avg EV charge limit"),
    ]
    ax_price.legend(handles=legend_elems, loc="upper left", fontsize=8)

    return {
        "im": im, "sc": sc, "title_field": title_field,
        "bars_heat": bars_heat, "bars_ev": bars_ev,
        "load_line": load_line, "heat_line": heat_line, "ev_line": ev_line,
        "price_line": price_line, "limit_line": limit_line,
        "cursor_load": cursor_load, "cursor_price": cursor_price,
        "axes": axes,
    }


def _update_frame(idx: int, snapshots, series, static, artists):
    s = snapshots[idx]
    t = s["time"]
    td1 = s.get("TD1", {}) or {}
    eh1 = s.get("EH1", {}) or {}
    ev1 = s.get("EV1", {}) or {}
    gr1 = s.get("GR1", {}) or {}
    la1 = s.get("LA1", {}) or {}

    field = td1.get("temperature_field")
    if field is not None:
        artists["im"].set_data(np.asarray(field))

    pos = static["positions"]
    heat_load = np.asarray(eh1.get("heating_load", np.zeros(static["num_households"])))
    ev_load = np.asarray(ev1.get("ev_load", np.zeros(static["num_households"])))
    heat_on = np.asarray(eh1.get("heating_on", np.zeros(static["num_households"], dtype=bool)))
    ev_on = np.asarray(ev1.get("charging", np.zeros(static["num_households"], dtype=bool)))

    sizes = 25 + 60 * np.clip((heat_load + ev_load) / 10.0, 0.0, 1.0)
    colors = np.where(heat_on, "#FF6B4A", "#88CCEE")
    colors = np.where(ev_on, "#4C7AFF", colors)
    artists["sc"].set_offsets(np.column_stack([pos[:, 1], pos[:, 0]]))
    artists["sc"].set_sizes(sizes)
    artists["sc"].set_color(colors)

    avg_T = float(np.asarray(field).mean()) if field is not None else 0.0
    artists["title_field"].set_text(
        f"t={t:6.2f}s   T_avg={avg_T:5.2f}°C   "
        f"heat_on={int(np.sum(heat_on))}  ev_on={int(np.sum(ev_on))}"
    )

    for bar, h in zip(artists["bars_heat"], heat_load):
        bar.set_height(h)
    for bar, e, h in zip(artists["bars_ev"], ev_load, heat_load):
        bar.set_height(e)
        bar.set_y(h)

    times = series["time"][: idx + 1]
    artists["load_line"].set_data(times, series["total_load"][: idx + 1])
    artists["heat_line"].set_data(times, series["total_heating"][: idx + 1])
    artists["ev_line"].set_data(times, series["total_ev"][: idx + 1])
    artists["cursor_load"].set_xdata([t, t])
    artists["price_line"].set_data(times, series["price"][: idx + 1])
    artists["limit_line"].set_data(times, series["avg_ev_limit"][: idx + 1])
    artists["cursor_price"].set_xdata([t, t])

    return (
        artists["im"], artists["sc"], artists["title_field"],
        *artists["bars_heat"], *artists["bars_ev"],
        artists["load_line"], artists["heat_line"], artists["ev_line"],
        artists["price_line"], artists["limit_line"],
        artists["cursor_load"], artists["cursor_price"],
    )


def _render_summary_png(snapshots, series, static, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 9))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig.suptitle("冬季寒潮 + EV 晚高峰 — 联合仿真总览（最后一帧）", fontsize=14, fontweight="bold")
    artists = _setup_axes(fig, static, series)
    _update_frame(len(snapshots) - 1, snapshots, series, static, artists)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Summary PNG saved: {out_path}")


def _render_animation(snapshots, series, static, gif_path: Path, mp4_path: Path, fps: int = 10) -> None:
    fig = plt.figure(figsize=(14, 9))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig.suptitle("冬季寒潮 + EV 晚高峰 — 联合仿真动态过程", fontsize=14, fontweight="bold")
    artists = _setup_axes(fig, static, series)
    n_frames = len(snapshots)

    def _frame(i: int):
        return _update_frame(i, snapshots, series, static, artists)

    anim = FuncAnimation(fig, _frame, frames=n_frames, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    try:
        anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=90)
        logger.info(f"GIF saved: {gif_path}")
    except Exception as e:
        logger.warning(f"GIF export failed: {e}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Winter Cold Wave + EV Evening Peak Co-Simulation"
    )
    parser.add_argument(
        "-c", "--config", default="config/topology_coldwave_ev.yaml",
        help="Path to topology YAML config",
    )
    args = parser.parse_args()
    main(args.config)
