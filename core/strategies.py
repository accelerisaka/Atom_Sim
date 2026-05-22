"""
跨速率交换策略引擎 —— ZOH / AVG / INTERP / EVENT 的具体实现。

所有策略解析结果在返回给目标仿真器前都会经过 `conn.apply_transform(value, ctx)`，
其中 `ctx` 是 `core.transforms.TransformContext`，用于支持需要跨仿真器多源数据的变换。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .protocol import ExchangeStrategy, S2SConnection


class StrategyEngine:
    """根据 S2S 连接声明的策略，从源历史数据中生成目标输入值。"""

    @staticmethod
    def resolve(
        conn: S2SConnection,
        history: List[Dict[str, Any]],
        target_time: float,
        ctx: Optional[Any] = None,
    ) -> Any:
        """
        Parameters
        ----------
        conn : S2SConnection
        history : 源仿真器的输出历史列表，每项 {"time": float, "data": {...}}
        target_time : 目标仿真器当前时刻
        ctx : 可选的 TransformContext，透传至 transform
        """
        if conn.strategy == ExchangeStrategy.ZOH:
            return StrategyEngine._zoh(conn, history, ctx)
        elif conn.strategy == ExchangeStrategy.AVG:
            return StrategyEngine._avg(conn, history, target_time, ctx)
        elif conn.strategy == ExchangeStrategy.INTERP:
            return StrategyEngine._interp(conn, history, target_time, ctx)
        elif conn.strategy == ExchangeStrategy.EVENT:
            return StrategyEngine._event(conn, history, ctx)
        raise ValueError(f"Unknown strategy: {conn.strategy}")

    # ------------------------------------------------------------------
    # 策略 1: 零阶保持
    # ------------------------------------------------------------------
    @staticmethod
    def _zoh(
        conn: S2SConnection,
        history: List[Dict[str, Any]],
        ctx: Optional[Any] = None,
    ) -> Any:
        if not history:
            return None
        latest = history[-1]["data"]
        raw = latest.get(conn.source.port)
        return conn.apply_transform(raw, ctx)

    # ------------------------------------------------------------------
    # 策略 2: 时间平均
    # ------------------------------------------------------------------
    @staticmethod
    def _avg(
        conn: S2SConnection,
        history: List[Dict[str, Any]],
        target_time: float,
        ctx: Optional[Any] = None,
    ) -> Any:
        if not history:
            return None
        values = [h["data"].get(conn.source.port) for h in history]
        values = [v for v in values if v is not None]
        if not values:
            return None

        sample = values[0]
        if isinstance(sample, np.ndarray):
            avg = np.mean(values, axis=0)
        elif isinstance(sample, (int, float)):
            avg = sum(values) / len(values)
        else:
            avg = values[-1]

        return conn.apply_transform(avg, ctx)

    # ------------------------------------------------------------------
    # 策略 3: 线性插值
    # ------------------------------------------------------------------
    @staticmethod
    def _interp(
        conn: S2SConnection,
        history: List[Dict[str, Any]],
        target_time: float,
        ctx: Optional[Any] = None,
    ) -> Any:
        if not history:
            return None
        if len(history) < 2:
            return StrategyEngine._zoh(conn, history, ctx)

        h0, h1 = history[-2], history[-1]
        t0, t1 = h0["time"], h1["time"]
        v0 = h0["data"].get(conn.source.port)
        v1 = h1["data"].get(conn.source.port)
        if v0 is None or v1 is None:
            return conn.apply_transform(v1, ctx)

        dt = t1 - t0
        if dt == 0:
            return conn.apply_transform(v1, ctx)

        alpha = min(max((target_time - t0) / dt, 0.0), 1.0)

        if isinstance(v0, np.ndarray):
            interp = v0 * (1 - alpha) + v1 * alpha
        elif isinstance(v0, (int, float)):
            interp = v0 * (1 - alpha) + v1 * alpha
        else:
            interp = v1
        return conn.apply_transform(interp, ctx)

    # ------------------------------------------------------------------
    # 策略 4: 事件触发（直接透传最新值）
    # ------------------------------------------------------------------
    @staticmethod
    def _event(
        conn: S2SConnection,
        history: List[Dict[str, Any]],
        ctx: Optional[Any] = None,
    ) -> Any:
        if not history:
            return None
        latest = history[-1]["data"]
        raw = latest.get(conn.source.port)
        return conn.apply_transform(raw, ctx)
