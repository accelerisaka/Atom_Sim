"""
分布式光伏 + 家用储能 VPP 博弈 联合仿真主入口
================================================
加载 config/topology_vpp_pv.yaml，运行 Orchestrator，并动态可视化。

可视化面板 (2x2)：
  ┌────────────────────────┬────────────────────────┐
  │ 社区家庭分布 + 净负荷   │ 逐户负荷分解 (基础/PV)  │
  ├────────────────────────┼────────────────────────┤
  │ 净负荷 vs 容量 (时序)   │ 电价 & 储能放电意愿     │
  └────────────────────────┴────────────────────────┘
导出 PNG / GIF / MP4 至 output_vpp_pv/。
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

from atoms.physical.hu3_vpp_households import HouseholdVPPProfile
from atoms.physical.sr1_solar_irradiance import SolarIrradiance
from atoms.physical.pv1_pv_generation import PVGeneration
from atoms.physical.hl1_household_base_load import HouseholdBaseLoad
from atoms.physical.hb1_home_battery import HomeBatteryStorage
from atoms.physical.nl1_net_load_aggregator import NetLoadAggregator
from atoms.physical.gr1_transformer import GridTransformer
from atoms.social.mk2_vpp_price_market import VPPPriceMarket
from atoms.social.pd9_battery_arbitrage import BatteryArbitrageAppraisal


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_vpp_pv")


_SIM_REGISTRY: Dict[str, type] = {
    "atoms.physical.HouseholdVPPProfile": HouseholdVPPProfile,
    "atoms.physical.SolarIrradiance": SolarIrradiance,
    "atoms.physical.PVGeneration": PVGeneration,
    "atoms.physical.HouseholdBaseLoad": HouseholdBaseLoad,
    "atoms.physical.HomeBatteryStorage": HomeBatteryStorage,
    "atoms.physical.NetLoadAggregator": NetLoadAggregator,
    "atoms.physical.GridTransformer": GridTransformer,
    "atoms.social.VPPPriceMarket": VPPPriceMarket,
    "atoms.social.BatteryArbitrageAppraisal": BatteryArbitrageAppraisal,
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
    for key in (
        "world_size",
        "pv_capacity_range",
        "battery_capacity_range",
        "battery_power_range",
        "arbitrage_sensitivity_range",
        "base_load_scale_range",
    ):
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


def _setup_gr1_vpp_pulses(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    """为 GR1 追加 export_pulse_ratio（不修改 atoms 源码）。"""
    gr1: GridTransformer = orch.simulators.get("GR1")  # type: ignore
    if gr1 is None:
        return

    export_kw_thr = float(scenario.get("export_kw_threshold", -4.0))
    gr1.state.setdefault("export_pulse_ratio", 0.0)
    gr1.state.setdefault("was_exporting", False)

    original_step = gr1.step
    original_get_outputs = gr1.get_outputs

    def _patched_step(dt: float) -> None:
        original_step(dt)
        total = float(gr1.state.get("total_load", 0.0))
        ratio = float(gr1.state.get("overload_ratio", 0.0))
        is_export = total < export_kw_thr
        edge = (not gr1.state["was_exporting"]) and is_export
        gr1.state["export_pulse_ratio"] = (
            float(min(1.0, abs(total) / max(abs(export_kw_thr), 1e-6))) if edge else 0.0
        )
        gr1.state["was_exporting"] = bool(is_export)

        overload_pulse = gr1.state.get("overload_pulse_ratio", 0.0)
        if overload_pulse > 0.0:
            orch.inject_event(
                EventMessage(
                    event_type="OVERLOAD",
                    source_sim_id="GR1",
                    timestamp=gr1.current_time,
                    payload={"overload_pulse_ratio": float(overload_pulse)},
                )
            )
            logger.info(
                f"  OVERLOAD @ t={gr1.current_time:.2f}s  ρ={overload_pulse:.3f}  "
                f"load={gr1.state['total_load']:.2f}kW"
            )

        export_pulse = gr1.state.get("export_pulse_ratio", 0.0)
        if export_pulse > 0.0:
            orch.inject_event(
                EventMessage(
                    event_type="GRID_EXPORT",
                    source_sim_id="GR1",
                    timestamp=gr1.current_time,
                    payload={"export_pulse_ratio": float(export_pulse)},
                )
            )
            logger.info(
                f"  GRID_EXPORT @ t={gr1.current_time:.2f}s  pulse={export_pulse:.3f}  "
                f"load={gr1.state['total_load']:.2f}kW  ρ={ratio:.3f}"
            )

    def _patched_get_outputs() -> Dict[str, Any]:
        out = original_get_outputs()
        out["export_pulse_ratio"] = float(gr1.state.get("export_pulse_ratio", 0.0))
        return out

    gr1.step = _patched_step  # type: ignore
    gr1.get_outputs = _patched_get_outputs  # type: ignore

    def _handle_market_event(_event: EventMessage) -> None:
        mk2: VPPPriceMarket = orch.simulators.get("MK2")  # type: ignore
        if mk2 is None:
            return
        mk2.step(0.0)
        mk2.record_output()

    orch.register_event_handler("OVERLOAD", _handle_market_event)
    orch.register_event_handler("GRID_EXPORT", _handle_market_event)


def _setup_evening_load_boost(orch: Orchestrator, scenario: Dict[str, Any]) -> None:
    """晚间用电高峰叠加（补丁 HL1.evening_peak_kw）。"""
    boost = scenario.get("evening_load_boost", {}) or {}
    if not boost.get("enabled", False):
        return
    trigger = float(boost.get("trigger_time", 95.0))
    extra = float(boost.get("extra_evening_kw", 0.35))

    hl1: HouseholdBaseLoad = orch.simulators.get("HL1")  # type: ignore
    if hl1 is None:
        return

    triggered = {"flag": False}
    original_step = hl1.step

    def _patched_step(dt: float) -> None:
        if (not triggered["flag"]) and hl1.current_time >= trigger:
            triggered["flag"] = True
            hl1.evening_peak_kw += extra
            logger.info(
                f"  EVENING_BOOST @ t={hl1.current_time:.2f}s  "
                f"+{extra:.2f} kW evening peak"
            )
        original_step(dt)

    hl1.step = _patched_step  # type: ignore


def main(config_path: str = "config/topology_vpp_pv.yaml") -> None:
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
    _setup_gr1_vpp_pulses(orch, scenario)
    _setup_evening_load_boost(orch, scenario)

    logger.info("\n" + orch.summary())

    duration = float(scenario.get("duration", 180.0))
    snap_interval = float(scenario.get("snapshot_interval", 1.0))

    t0 = time.perf_counter()
    orch.run(duration=duration, snapshot_interval=snap_interval)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Simulation finished in {elapsed:.2f}s wall-clock, "
        f"{len(orch.history)} snapshots collected."
    )

    output_dir = Path("output_vpp_pv")
    output_dir.mkdir(exist_ok=True)

    hu3: HouseholdVPPProfile = orch.simulators.get("HU3")  # type: ignore
    gr1: GridTransformer = orch.simulators.get("GR1")  # type: ignore
    if hu3 is None or gr1 is None:
        logger.error("Missing core simulators for visualization.")
        return

    static_meta = {
        "positions": hu3.state["positions"],
        "world_size": hu3.world_size,
        "capacity": gr1.capacity,
        "num_households": hu3.num_households,
        "pv_capacity": hu3.state["pv_capacity_kw"],
        "has_battery": hu3.state["has_battery"],
    }

    snapshots = orch.history
    if not snapshots:
        logger.warning("No snapshots, skipping visualization.")
        return

    series = _build_time_series(snapshots)
    _render_summary_png(snapshots, series, static_meta, output_dir / "vpp_pv_summary.png")
    _render_animation(
        snapshots,
        series,
        static_meta,
        gif_path=output_dir / "vpp_pv_dynamic.gif",
        mp4_path=output_dir / "vpp_pv_dynamic.mp4",
        fps=10,
    )

    final = snapshots[-1]
    logger.info("=" * 60)
    logger.info(f"Final t={final['time']:.2f}s")
    gr1_out = final.get("GR1", {}) or {}
    mk2_out = final.get("MK2", {}) or {}
    pd9_out = final.get("PD9", {}) or {}
    pv1_out = final.get("PV1", {}) or {}
    logger.info(
        f"  total_load = {gr1_out.get('total_load', 0):.2f} kW "
        f"(ρ={gr1_out.get('overload_ratio', 0):.3f}, "
        f"PV_total={pv1_out.get('total_pv', 0):.2f} kW)"
    )
    logger.info(
        f"  electricity_price = {mk2_out.get('electricity_price', 0):.3f} yuan/kWh "
        f"(x{mk2_out.get('price_multiplier', 1):.2f}, "
        f"negative={mk2_out.get('is_negative_price', False)})"
    )
    logger.info(
        f"  avg discharge fraction = {pd9_out.get('avg_discharge_frac', 0):.3f}"
    )


def _build_time_series(snapshots: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    times, total_load, ratio, price, avg_discharge = [], [], [], [], []
    total_pv, total_base, avg_soc = [], [], []
    for s in snapshots:
        times.append(s["time"])
        gr1 = s.get("GR1", {}) or {}
        mk2 = s.get("MK2", {}) or {}
        pd9 = s.get("PD9", {}) or {}
        pv1 = s.get("PV1", {}) or {}
        nl1 = s.get("NL1", {}) or {}
        hb1 = s.get("HB1", {}) or {}
        total_load.append(float(gr1.get("total_load", 0.0)))
        ratio.append(float(gr1.get("overload_ratio", 0.0)))
        price.append(float(mk2.get("electricity_price", 0.0)))
        avg_discharge.append(float(pd9.get("avg_discharge_frac", 0.0)))
        total_pv.append(float(pv1.get("total_pv", 0.0)))
        total_base.append(float(nl1.get("total_base", 0.0)))
        soc = hb1.get("battery_soc", [])
        avg_soc.append(float(np.mean(soc)) if len(soc) else 0.0)
    return {
        "time": np.asarray(times, dtype=np.float64),
        "total_load": np.asarray(total_load),
        "overload_ratio": np.asarray(ratio),
        "price": np.asarray(price),
        "avg_discharge": np.asarray(avg_discharge),
        "total_pv": np.asarray(total_pv),
        "total_base": np.asarray(total_base),
        "avg_soc": np.asarray(avg_soc),
    }


def _setup_axes(fig, static: Dict[str, Any], series: Dict[str, np.ndarray]):
    axes = fig.subplots(2, 2)
    ax_map, ax_bars, ax_load, ax_price = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    w, h = static["world_size"]
    ax_map.set_title("社区家庭分布 (颜色=净负荷)", pad=22, fontsize=10)
    ax_map.set_xlim(0, w)
    ax_map.set_ylim(h, 0)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    pos = static["positions"]
    sc = ax_map.scatter(
        pos[:, 0], pos[:, 1], s=30, c="#7AC9FF",
        edgecolors="black", linewidths=0.6, zorder=5,
    )
    title_map = ax_map.text(
        0.02, 1.04, "", transform=ax_map.transAxes,
        fontsize=9, fontweight="bold", color="#333333",
    )

    n = static["num_households"]
    ax_bars.set_title("逐户负荷分解 (kW)")
    ax_bars.set_xlabel("Household ID")
    ax_bars.set_ylabel("Power (kW)")
    bar_x = np.arange(n)
    bars_base = ax_bars.bar(bar_x, np.zeros(n), width=0.35, label="base", color="#4C9AFF")
    bars_pv = ax_bars.bar(
        bar_x + 0.35, np.zeros(n), width=0.35, label="PV gen", color="#FFD24A"
    )
    ax_bars.axhline(0, color="gray", linewidth=0.5)
    ax_bars.set_xlim(-0.6, n - 0.4)
    ax_bars.legend(loc="upper right", fontsize=8)

    cap = static["capacity"]
    ax_load.set_title("社区净负荷 vs 变压器容量")
    ax_load.set_xlabel("time (s)")
    ax_load.set_ylabel("Net load (kW)")
    ax_load.set_xlim(series["time"][0], series["time"][-1])
    y_lo = min(series["total_load"].min() * 1.15, -cap * 0.5, -5.0)
    y_hi = max(series["total_load"].max() * 1.15, cap * 1.3, 5.0)
    ax_load.set_ylim(y_lo, y_hi)
    ax_load.axhline(0, color="gray", linewidth=0.8)
    ax_load.axhline(cap, color="red", linestyle="--", linewidth=1.2, label=f"cap={cap:.0f}kW")
    ax_load.axhline(-cap * 0.1, color="#2F7A48", linestyle=":", linewidth=1.0, label="export zone")
    (load_line,) = ax_load.plot([], [], color="#1F4E79", linewidth=1.6, label="net load")
    (pv_line,) = ax_load.plot([], [], color="#E07B00", linewidth=1.0, linestyle="--", label="total PV")
    cursor_load = ax_load.axvline(series["time"][0], color="black", linewidth=0.5, alpha=0.6)
    ax_load.legend(loc="upper left", fontsize=7)

    ax_price.set_title("电价 (主轴) & 平均放电意愿 (副轴)")
    ax_price.set_xlabel("time (s)")
    ax_price.set_ylabel("Price (yuan/kWh)", color="#B63A2A")
    ax_price.set_xlim(series["time"][0], series["time"][-1])
    p_max = max(series["price"].max() * 1.15, 0.6)
    p_min = min(series["price"].min() * 0.9, 0.05)
    ax_price.set_ylim(p_min, p_max)
    (price_line,) = ax_price.plot([], [], color="#B63A2A", linewidth=1.7)
    ax_price.tick_params(axis="y", labelcolor="#B63A2A")
    cursor_price = ax_price.axvline(series["time"][0], color="black", linewidth=0.5, alpha=0.6)

    ax_price2 = ax_price.twinx()
    ax_price2.set_ylabel("Discharge fraction", color="#2F7A48")
    ax_price2.set_ylim(0, 1.05)
    (discharge_line,) = ax_price2.plot(
        [], [], color="#2F7A48", linewidth=1.4, linestyle="-."
    )
    ax_price2.tick_params(axis="y", labelcolor="#2F7A48")

    legend_elems = [
        Patch(facecolor="#B63A2A", label="electricity_price"),
        Patch(facecolor="#2F7A48", label="avg discharge frac"),
    ]
    ax_price.legend(handles=legend_elems, loc="upper left", fontsize=8)

    return {
        "sc": sc,
        "title_map": title_map,
        "bars_base": bars_base,
        "bars_pv": bars_pv,
        "load_line": load_line,
        "pv_line": pv_line,
        "price_line": price_line,
        "discharge_line": discharge_line,
        "cursor_load": cursor_load,
        "cursor_price": cursor_price,
        "axes": axes,
    }


def _update_frame(idx: int, snapshots, series, static, artists):
    s = snapshots[idx]
    t = s["time"]
    nl1 = s.get("NL1", {}) or {}
    pv1 = s.get("PV1", {}) or {}
    hl1 = s.get("HL1", {}) or {}
    gr1 = s.get("GR1", {}) or {}
    mk2 = s.get("MK2", {}) or {}
    pd9 = s.get("PD9", {}) or {}
    sr1 = s.get("SR1", {}) or {}

    net = np.asarray(nl1.get("household_loads", np.zeros(static["num_households"])))
    pos = static["positions"]
    sizes = 25 + 70 * np.clip(np.abs(net) / 4.0, 0.0, 1.0)
    colors = np.where(
        net < -0.05, "#2F7A48",
        np.where(net > 1.0, "#B63A2A", "#7AC9FF"),
    )
    artists["sc"].set_offsets(pos)
    artists["sc"].set_sizes(sizes)
    artists["sc"].set_color(colors)

    irr = float(sr1.get("irradiance_factor", 0.0))
    artists["title_map"].set_text(
        f"t={t:6.2f}s   irradiance={irr:.2f}   "
        f"net={gr1.get('total_load', 0):.1f}kW  ρ={gr1.get('overload_ratio', 0):.2f}"
    )

    base = np.asarray(hl1.get("base_load", np.zeros(static["num_households"])))
    pv = np.asarray(pv1.get("pv_power", np.zeros(static["num_households"])))
    for bar, h in zip(artists["bars_base"], base):
        bar.set_height(h)
    for bar, h in zip(artists["bars_pv"], -pv):
        bar.set_height(h)

    times = series["time"][: idx + 1]
    artists["load_line"].set_data(times, series["total_load"][: idx + 1])
    artists["pv_line"].set_data(times, -series["total_pv"][: idx + 1])
    artists["cursor_load"].set_xdata([t, t])

    artists["price_line"].set_data(times, series["price"][: idx + 1])
    artists["discharge_line"].set_data(times, series["avg_discharge"][: idx + 1])
    artists["cursor_price"].set_xdata([t, t])

    return (
        artists["sc"],
        artists["title_map"],
        *artists["bars_base"],
        *artists["bars_pv"],
        artists["load_line"],
        artists["pv_line"],
        artists["price_line"],
        artists["discharge_line"],
        artists["cursor_load"],
        artists["cursor_price"],
    )


def _render_summary_png(snapshots, series, static, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 9))
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig.suptitle(
        "VPP 光伏储能博弈 — 联合仿真总览（最后一帧）",
        fontsize=14, fontweight="bold",
    )
    artists = _setup_axes(fig, static, series)
    _update_frame(len(snapshots) - 1, snapshots, series, static, artists)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Summary PNG saved: {out_path}")


def _render_animation(
    snapshots, series, static, gif_path: Path, mp4_path: Path, fps: int = 10
) -> None:
    fig = plt.figure(figsize=(14, 9))
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig.suptitle(
        "VPP 光伏储能博弈 — 联合仿真动态过程",
        fontsize=14, fontweight="bold",
    )
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
        description="Distributed PV + Home Battery VPP Co-Simulation"
    )
    parser.add_argument(
        "-c", "--config", default="config/topology_vpp_pv.yaml",
        help="Path to topology YAML config",
    )
    args = parser.parse_args()
    main(args.config)
