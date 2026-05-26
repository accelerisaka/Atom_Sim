"""
S2S 连接数据变换库（Transform Library）
========================================

本文件是整个联合仿真系统中 **所有** S2S 连接上 `transform()` 函数的唯一注册中心。

设计原则
--------
1. **原子仿真器纯粹性**：每个原子仿真器的 `step()` 只处理"本领域内"的计算逻辑，
   不假设任何源仿真器的数据语义、形状或单位。输入端口规约（port schema）即契约。
2. **变换集中化**：所有跨原子的数据翻译——单位换算、场聚合、形状适配、
   空间采样、语义改写——都在此文件中实现，并通过 `@register_transform` 注册。
3. **可复用与可 introspect**：每个 Transform 都附带
   - `source_schema` / `target_schema`：上下游数据形式描述
   - `description`：物理/语义含义
   - `connections`：该 Transform 被哪些连接使用（反向索引）
4. **形式化**：即使是"端口重命名"这样的平凡情况，也应显式声明 `identity`，
   以保证每一条 S2S 连接都经过统一的 Transform 协议。

两种 Transform 签名
-------------------
- **一参函数**  `fn(value) -> value`
    适用于纯函数式变换（单位换算、场聚合、逐元素映射等）。

- **二参函数**  `fn(value, ctx: TransformContext) -> value`
    适用于需要**多源数据**或**仿真器元信息**的变换，典型如"把场在每个个体位置处采样"。
    `ctx` 提供对所有仿真器状态、全局总线快照及连接两端 sim_id 的只读访问。

    注册时会自动侦测 arity，一参函数会被包裹成二参形式（忽略 ctx）。
    因此从 `apply_transform` 调用点看，所有注册 transform 都是统一的二参契约。

新增一条连接的流程
------------------
1. 在此文件中定义并 `@register_transform(...)` 装饰一个变换函数；
2. 在该仿真环境的`config/topology_xxx.yaml` 对应连接下声明 `transform: <name>`；
3. `main.py._build_connection` 会自动按名解析并注入到 `S2SConnection.transform`。

所有公开 API
------------
- `TransformContext`         变换运行时上下文
- `TransformSpec`            变换规约数据类
- `TRANSFORM_REGISTRY`       名称 → 规约的全局字典
- `register_transform(...)`  装饰器，用于注册
- `get_transform(name)`      按名取出可调用对象（统一二参形式）
- `describe_transform(name)` 获取结构化元数据（给 coding agent 用）
- `list_transforms()`        列出全部已注册变换名
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# -----------------------------------------------------------------------------
# Transform 运行时上下文
# -----------------------------------------------------------------------------


@dataclass
class TransformContext:
    """
    Transform 运行时上下文。
    为需要多源数据/仿真器元信息的 transform 提供只读访问。

    Fields
    ------
    simulators : Dict[str, AtomicSimulator]
        所有已注册仿真器（按 sim_id 索引）。transform 可读取任意仿真器的 state
        （例如 CM1.state["positions"]、TD1.cell_size 等元参数）。
        **规定：只读**。transform 不应修改任何仿真器状态。
    bus : Dict[str, Dict[str, Any]]
        全局总线快照——上一次 _pull_outputs 产生的所有仿真器输出副本。
        对于 event 路径可能略早于最新 step()，此时应优先从 `simulators` 读 state。
    source_sim_id, target_sim_id : str
        当前连接两端的仿真器 ID，便于 transform 做自适应（可选使用）。
    target_time : float
        目标仿真器当前时刻（全局时间）。
    """

    simulators: Dict[str, Any]
    bus: Dict[str, Dict[str, Any]]
    source_sim_id: str
    target_sim_id: str
    target_time: float

    def sim_state(self, sim_id: str, key: str, default: Any = None) -> Any:
        """读取指定仿真器 state 字段；不存在则返回 default。"""
        sim = self.simulators.get(sim_id)
        if sim is None:
            return default
        return sim.state.get(key, default)

    def sim_attr(self, sim_id: str, attr: str, default: Any = None) -> Any:
        """读取指定仿真器的属性（如 TD1.cell_size）；不存在则返回 default。"""
        sim = self.simulators.get(sim_id)
        if sim is None:
            return default
        return getattr(sim, attr, default)


# -----------------------------------------------------------------------------
# 注册机制
# -----------------------------------------------------------------------------


@dataclass
class TransformSpec:
    """S2S 连接变换规约——描述一个 transform 函数的语义和适用场景。"""

    name: str
    fn: Callable[[Any, Optional[TransformContext]], Any]  # 已统一为二参形式
    source_schema: str
    target_schema: str
    description: str
    connections: List[str] = field(default_factory=list)
    context_aware: bool = False  # 原始 fn 是否声明了 ctx 形参

    def __call__(self, value: Any, ctx: Optional[TransformContext] = None) -> Any:
        return self.fn(value, ctx)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_schema": self.source_schema,
            "target_schema": self.target_schema,
            "description": self.description.strip(),
            "connections": list(self.connections),
            "context_aware": self.context_aware,
        }


TRANSFORM_REGISTRY: Dict[str, TransformSpec] = {}


def _count_positional_params(fn: Callable[..., Any]) -> int:
    try:
        sig = inspect.signature(fn)
        return sum(
            1
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
    except (TypeError, ValueError):
        return 1


def register_transform(
    name: str,
    source_schema: str,
    target_schema: str,
    description: str,
    connections: Optional[List[str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    装饰器：注册命名 Transform。

    支持两种签名：
      * `fn(value) -> value`               （纯函数，自动包裹）
      * `fn(value, ctx) -> value`          （可读取跨仿真器上下文）
    """

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform '{name}' already registered.")

        arity = _count_positional_params(fn)
        context_aware = arity >= 2

        if context_aware:
            wrapped = fn
        else:
            # 一参函数：自动包裹成二参（忽略 ctx）
            def wrapped(value: Any, ctx: Optional[TransformContext] = None, _inner=fn) -> Any:
                return _inner(value)

        TRANSFORM_REGISTRY[name] = TransformSpec(
            name=name,
            fn=wrapped,
            source_schema=source_schema,
            target_schema=target_schema,
            description=description,
            connections=list(connections or []),
            context_aware=context_aware,
        )
        return fn  # 返回原函数给上层使用（不改变调用习惯）

    return _decorator


def get_transform(name: str) -> Callable[[Any, Optional[TransformContext]], Any]:
    """按名取出 transform 函数（已统一为二参形式）；未注册则抛出 KeyError。"""
    if name not in TRANSFORM_REGISTRY:
        available = ", ".join(sorted(TRANSFORM_REGISTRY.keys()))
        raise KeyError(
            f"Transform '{name}' is not registered. Available: [{available}]"
        )
    return TRANSFORM_REGISTRY[name].fn


def describe_transform(name: str) -> Dict[str, Any]:
    """返回指定 Transform 的结构化元数据。"""
    if name not in TRANSFORM_REGISTRY:
        raise KeyError(f"Transform '{name}' is not registered.")
    return TRANSFORM_REGISTRY[name].to_dict()


def list_transforms() -> List[str]:
    """列出所有已注册 Transform 名称。"""
    return sorted(TRANSFORM_REGISTRY.keys())


# -----------------------------------------------------------------------------
# 共享工具：场→逐个体采样
# -----------------------------------------------------------------------------


def _sample_field_at_positions(
    field_2d: Any,
    positions: Any,
    cell_size: float,
    default: float = 0.0,
) -> Optional[np.ndarray]:
    """
    在每个个体位置处采样二维场，返回长度为 n 的一维数组。

    Parameters
    ----------
    field_2d : ndarray[H, W]
    positions : ndarray[n, 2]  世界坐标（与 cell_size 单位一致）
    cell_size : float
    default   : 越界/空输入时的填充值

    Returns
    -------
    ndarray[n] 或 None（输入非法）
    """
    if field_2d is None or positions is None or cell_size <= 0:
        return None

    fld = np.asarray(field_2d, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if fld.ndim != 2 or pos.ndim != 2 or pos.shape[1] < 2:
        return None

    H, W = fld.shape
    gr = np.clip((pos[:, 0] / cell_size).astype(int), 0, H - 1)
    gc = np.clip((pos[:, 1] / cell_size).astype(int), 0, W - 1)
    return fld[gr, gc].astype(np.float64)


def _get_agent_positions(ctx: Optional[TransformContext]) -> Optional[np.ndarray]:
    """优先从最新 CM1.state 读取位置；回落到 bus 快照。"""
    if ctx is None:
        return None
    pos = ctx.sim_state("CM1", "positions")
    if pos is None:
        cm1_bus = ctx.bus.get("CM1", {}) if ctx.bus else {}
        pos = cm1_bus.get("positions")
    return pos


def _get_grid_cell_size(ctx: Optional[TransformContext], sim_id: str, fallback: float = 0.5) -> float:
    if ctx is None:
        return fallback
    cs = ctx.sim_attr(sim_id, "cell_size", fallback)
    try:
        return float(cs)
    except (TypeError, ValueError):
        return fallback


# =============================================================================
# 变换实现
# =============================================================================


# -----------------------------------------------------------------------------
# 0. 恒等 —— 用于"端口重命名"或数据形式完全相同的连接
# -----------------------------------------------------------------------------


@register_transform(
    name="identity",
    source_schema="Any",
    target_schema="Any (identical)",
    description=(
        "恒等变换。源与目标数据形式完全一致，仅做端口语义重命名。"
        "在此形式化声明以保证所有连接统一经过 Transform 协议。"
    ),
    connections=[
        "pd3_panic_to_bp6",
        "bp6_eff_to_cm1",
        "s2_bias_to_cm1",
        "cm1_pos_to_s2",
        "cm1_vel_to_s2",
        "bp5_mask_to_s2",
        "cm1_fallen_to_sd4",
        "cm1_active_to_sd4",
        "cm1_pos_to_sd4",
        "cm1_pos_to_fl4",
        "fl4_density_to_p6",
        "p6_constraint_to_cm1",
    ],
)
def identity(value: Any) -> Any:
    return value


# -----------------------------------------------------------------------------
# 1. TD1 温度场  →  PD3 逐个体危险度
# -----------------------------------------------------------------------------


@register_transform(
    name="temperature_field_to_per_agent_danger",
    source_schema="ndarray[H, W] float, 单位 °C",
    target_schema="ndarray[n] float, [0, 1]",
    description=(
        "在每个个体位置处采样温度场，映射为该个体的热暴露危险度。\n"
        "需要 ctx: CM1.state['positions'] (世界坐标, ndarray[n, 2])、"
        "TD1.cell_size (网格尺度, float)。\n"
        "物理约定：低于 40°C 无感 (0)，达到 340°C 危险上限 (1)。\n"
        "公式：danger_i = clip((T(pos_i) - 40) / 300, 0, 1)"
    ),
    connections=[
        "td1_temp_to_pd3",
        "td1_temp_spike_to_pd3_event",  # TEMP_SPIKE 事件桥复用
    ],
)
def temperature_field_to_per_agent_danger(
    field_T: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_T is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "TD1")
    sampled = _sample_field_at_positions(field_T, positions, cell_size)
    if sampled is None:
        return None
    return np.clip((sampled - 40.0) / 300.0, 0.0, 1.0)


# -----------------------------------------------------------------------------
# 2. FL3 烟雾浓度场  →  PD3/BP5 逐个体烟雾暴露度
# -----------------------------------------------------------------------------


@register_transform(
    name="smoke_field_to_per_agent_exposure",
    source_schema="ndarray[H, W] float, 浓度 [0, 1]",
    target_schema="ndarray[n] float, [0, 1]",
    description=(
        "在每个个体位置处采样烟雾浓度场，得到该个体面临的烟雾暴露度。\n"
        "需要 ctx: CM1.state['positions']、FL3.cell_size。\n"
        "公式：exposure_i = clip(C(pos_i), 0, 1)"
    ),
    connections=["fl3_smoke_to_pd3", "fl3_vis_to_bp5"],
)
def smoke_field_to_per_agent_exposure(
    field_C: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_C is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "FL3")
    sampled = _sample_field_at_positions(field_C, positions, cell_size)
    if sampled is None:
        return None
    return np.clip(sampled, 0.0, 1.0)


# -----------------------------------------------------------------------------
# 3. FL3 能见度场  →  PD3 逐个体可见度
# -----------------------------------------------------------------------------


@register_transform(
    name="visibility_field_to_per_agent_visibility",
    source_schema="ndarray[H, W] float, 能见度 [0, 1]",
    target_schema="ndarray[n] float, [0, 1] (1=完全可见)",
    description=(
        "在每个个体位置处采样能见度场，得到该个体的局部可见度。\n"
        "需要 ctx: CM1.state['positions']、FL3.cell_size。\n"
        "目标侧使用 (1 - visibility_i) 作为出口阻碍度。\n"
        "公式：vis_i = clip(V(pos_i), 0, 1)"
    ),
    connections=["fl3_vis_to_pd3"],
)
def visibility_field_to_per_agent_visibility(
    field_V: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_V is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "FL3")
    sampled = _sample_field_at_positions(field_V, positions, cell_size, default=1.0)
    if sampled is None:
        return None
    return np.clip(sampled, 0.0, 1.0)


# -----------------------------------------------------------------------------
# 4. TD1 温度场  →  FL3 产烟源强度场 (场 → 场，不需要 ctx)
# -----------------------------------------------------------------------------


@register_transform(
    name="temperature_field_to_smoke_source_rate",
    source_schema="ndarray[H, W] float, 单位 °C",
    target_schema="ndarray[H, W] float, 产烟速率 (1/s)",
    description=(
        "将温度场映射为逐格的产烟源强度（场形态不变，仅做每格元素值的语义转换）。"
        "物理约定：温度低于 100°C 不产烟；800°C 对应单位产烟速率。\n"
        "公式：rate(i, j) = max(0, (T(i, j) - 100) / 800)"
    ),
    connections=["td1_heat_to_fl3"],
)
def temperature_field_to_smoke_source_rate(field_T: Any) -> Optional[np.ndarray]:
    if field_T is None:
        return None
    arr = np.asarray(field_T, dtype=np.float64)
    return np.where(arr > 100.0, (arr - 100.0) / 800.0, 0.0)


# -----------------------------------------------------------------------------
# 可选：保留场→全局标量的聚合变换（当前无连接绑定，供未来复用）
# -----------------------------------------------------------------------------


@register_transform(
    name="temperature_field_to_danger_index",
    source_schema="ndarray[H, W] float, °C",
    target_schema="float in [0, 1]",
    description=(
        "【备用】将温度场聚合为全局危险指数（取空间最大值）。"
        "当前已由 per-agent 版本 `temperature_field_to_per_agent_danger` 取代，"
        "保留于此以便未来场景级耦合复用。"
    ),
    connections=[],
)
def temperature_field_to_danger_index(field_T: Any) -> float:
    if field_T is None:
        return 0.0
    arr = np.asarray(field_T, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    max_T = float(arr.max())
    return float(np.clip((max_T - 40.0) / 300.0, 0.0, 1.0))


@register_transform(
    name="smoke_field_to_exposure_scalar",
    source_schema="ndarray[H, W] float, [0, 1]",
    target_schema="float in [0, 1]",
    description=(
        "【备用】将烟雾浓度场聚合为场景整体烟雾暴露度（取空间最大值）。"
        "当前已由 per-agent 版本 `smoke_field_to_per_agent_exposure` 取代。"
    ),
    connections=[],
)
def smoke_field_to_exposure_scalar(field_C: Any) -> float:
    if field_C is None:
        return 0.0
    arr = np.asarray(field_C, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.clip(arr.max(), 0.0, 1.0))


@register_transform(
    name="visibility_field_to_exit_visibility",
    source_schema="ndarray[H, W] float, [0, 1]",
    target_schema="float in [0, 1]",
    description=(
        "【备用】将能见度场聚合为出口可见度标量（取空间平均值）。"
        "当前已由 per-agent 版本 `visibility_field_to_per_agent_visibility` 取代。"
    ),
    connections=[],
)
def visibility_field_to_exit_visibility(field_V: Any) -> float:
    if field_V is None:
        return 1.0
    arr = np.asarray(field_V, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    return float(np.clip(arr.mean(), 0.0, 1.0))


# =============================================================================
# Coding Agent Hint
# =============================================================================
#
# 查询某连接使用了哪个 Transform：
#
#   from core.transforms import TRANSFORM_REGISTRY
#   for spec in TRANSFORM_REGISTRY.values():
#       if "td1_temp_to_pd3" in spec.connections:
#           print(spec.to_dict())
#
# 列出全部 Transforms：
#
#   from core.transforms import list_transforms, describe_transform
#   for n in list_transforms():
#       print(describe_transform(n))
#
# 判断某 Transform 是否需要 ctx：
#
#   TRANSFORM_REGISTRY["temperature_field_to_per_agent_danger"].context_aware  # True
# =============================================================================


# =============================================================================
# 智能电网负载均衡场景（Smart Grid Load Balancing）追加 Transforms
# =============================================================================
#
# 设计说明
# --------
# 该场景中只有 1 条连接需要"非 identity"翻译：
#   TD1.temperature_field (ndarray[H, W], °C) → AC1/PD7.local_temp (ndarray[N], °C)
# 即在 HU1.positions 处对温度场做空间采样。
# 其余跨原子连接（HU1→PD7、PD7→AC1、AC1→GR1、GR1→MK1、MK1→PD7）数据形状/单位
# 完全一致，统一声明为 identity，仅做端口语义重命名。
# =============================================================================


def _get_household_positions(ctx: Optional[TransformContext]) -> Optional[np.ndarray]:
    """优先从最新 HU1.state 读取家庭位置；回落到 bus 快照。"""
    if ctx is None:
        return None
    pos = ctx.sim_state("HU1", "positions")
    if pos is None:
        hu1_bus = ctx.bus.get("HU1", {}) if ctx.bus else {}
        pos = hu1_bus.get("positions")
    return pos


@register_transform(
    name="temperature_field_to_per_household_temp",
    source_schema="ndarray[H, W] float, 单位 °C",
    target_schema="ndarray[N] float, 单位 °C",
    description=(
        "在每户家庭位置处采样户外温度场，得到该户的局部环境温度。\n"
        "需要 ctx: HU1.state['positions'] (世界坐标 ndarray[N, 2])、"
        "TD1.cell_size (网格尺度, float)。\n"
        "公式：T_local_i = TD1_field[ floor(pos_i[0] / cell_size), "
        "floor(pos_i[1] / cell_size) ]"
    ),
    connections=[
        "td1_temp_to_ac1",
        "td1_temp_to_pd7",
    ],
)
def temperature_field_to_per_household_temp(
    field_T: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_household_positions(ctx)
    if field_T is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "TD1", fallback=1.0)
    return _sample_field_at_positions(field_T, positions, cell_size, default=30.0)


# =============================================================================
# 地铁站有毒重气体泄漏 + 智能排风疏散场景（Subway Heavy Gas Leak）追加 Transforms
# =============================================================================
#
# 设计说明
# --------
# FL5 输出三种二维场（浓度 / 毒性危害 / 能见度），需在 CM1.positions 处采样为
# 逐个体 ndarray[n]，再分别送入 PD3（local_danger / smoke_exposure / exit_visibility）
# 与 BP5（smoke_exposure）。VT1 ↔ FL5 的排风场形状一致，使用 identity。
# 泄漏源强度场由 run_subway_gas.py 在场景初始化时直接写入 FL5.inputs，
# 不经 S2S 连接。
# =============================================================================


@register_transform(
    name="gas_field_to_per_agent_exposure",
    source_schema="ndarray[H, W] float, 气体浓度 [0, 1]",
    target_schema="ndarray[n] float, [0, 1]",
    description=(
        "在每个个体位置处采样 FL5 气体浓度场，得到该个体的气体暴露度。\n"
        "需要 ctx: CM1.state['positions']、FL5.cell_size。\n"
        "公式：exposure_i = clip(C(pos_i), 0, 1)"
    ),
    connections=["fl5_gas_to_pd3_exposure", "fl5_gas_to_bp5"],
)
def gas_field_to_per_agent_exposure(
    field_C: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_C is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "FL5")
    sampled = _sample_field_at_positions(field_C, positions, cell_size)
    if sampled is None:
        return None
    return np.clip(sampled, 0.0, 1.0)


@register_transform(
    name="gas_hazard_field_to_per_agent_danger",
    source_schema="ndarray[H, W] float, 毒性危害 [0, 1]",
    target_schema="ndarray[n] float, [0, 1]",
    description=(
        "在每个个体位置处采样 FL5 毒性危害场，映射为 PD3 的 local_danger。\n"
        "需要 ctx: CM1.state['positions']、FL5.cell_size。\n"
        "公式：danger_i = clip(H(pos_i), 0, 1)"
    ),
    connections=["fl5_hazard_to_pd3_danger", "fl5_hazard_spike_to_pd3_event"],
)
def gas_hazard_field_to_per_agent_danger(
    field_H: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_H is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "FL5")
    sampled = _sample_field_at_positions(field_H, positions, cell_size)
    if sampled is None:
        return None
    return np.clip(sampled, 0.0, 1.0)


@register_transform(
    name="gas_visibility_field_to_per_agent_visibility",
    source_schema="ndarray[H, W] float, 能见度 [0, 1] (1=完全可见)",
    target_schema="ndarray[n] float, [0, 1]",
    description=(
        "在每个个体位置处采样 FL5 能见度场，得到该个体的局部可见度。\n"
        "需要 ctx: CM1.state['positions']、FL5.cell_size。\n"
        "公式：vis_i = clip(V(pos_i), 0, 1)"
    ),
    connections=["fl5_vis_to_pd3"],
)
def gas_visibility_field_to_per_agent_visibility(
    field_V: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_V is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "FL5")
    sampled = _sample_field_at_positions(field_V, positions, cell_size, default=1.0)
    if sampled is None:
        return None
    return np.clip(sampled, 0.0, 1.0)


@register_transform(
    name="leak_points_to_gas_source_rate",
    source_schema="list[[row, col, rate]] — 泄漏点 (网格坐标, 1/s)",
    target_schema="ndarray[H, W] float, 产气速率 (1/s)",
    description=(
        "将离散泄漏点列表映射为 FL5 所需的二维产气源强度场。\n"
        "需要 ctx: FL5.grid_size (H, W)。\n"
        "每个泄漏点在对应网格格点叠加 rate；越界点忽略。"
    ),
    connections=[],
)
def leak_points_to_gas_source_rate(
    leak_points: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    if leak_points is None or ctx is None:
        return None
    grid_size = ctx.sim_attr("FL5", "grid_size", (50, 50))
    H, W = int(grid_size[0]), int(grid_size[1])
    rate_field = np.zeros((H, W), dtype=np.float64)
    for pt in leak_points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 3:
            continue
        r, c, rate = int(pt[0]), int(pt[1]), float(pt[2])
        if 0 <= r < H and 0 <= c < W:
            rate_field[r, c] += rate
    return rate_field


# =============================================================================
# 体育场突发灾害踩踏 + 应急广播引导场景（Stadium Crush Evacuation）追加 Transforms
# =============================================================================
#
# 设计说明
# --------
# EX1 爆炸危害场 / 烟尘能见度场 → 在 CM1.positions 采样为逐个体向量，送入 PD3。
# CF1 挤压应力场 → 采样为逐个体冲击冲量 (CM1.external_impulses) 与附加危险度 (PD3)。
# EG1 服从权重 + BC1 播报出口 → 在 transform 中结合 CM1 位置生成 CM1.desired_velocity。
# 爆心种子场由 run_stadium_crush.py 场景初始化写入 EX1.inputs，不经 S2S。
# =============================================================================


@register_transform(
    name="blast_hazard_field_to_per_agent_danger",
    source_schema="ndarray[H, W] float, 爆炸危害 [0, 1]",
    target_schema="ndarray[n] float, [0, 1]",
    description=(
        "在每个个体位置处采样 EX1 爆炸危害场，映射为 PD3 的 local_danger。\n"
        "需要 ctx: CM1.state['positions']、EX1.cell_size。"
    ),
    connections=["ex1_blast_to_pd3_danger"],
)
def blast_hazard_field_to_per_agent_danger(
    field_H: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_H is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "EX1")
    sampled = _sample_field_at_positions(field_H, positions, cell_size)
    if sampled is None:
        return None
    return np.clip(sampled, 0.0, 1.0)


@register_transform(
    name="dust_visibility_field_to_per_agent_visibility",
    source_schema="ndarray[H, W] float, 能见度 [0, 1] (1=完全可见)",
    target_schema="ndarray[n] float, [0, 1]",
    description=(
        "在每个个体位置处采样 EX1 烟尘能见度场，得到 PD3 的 exit_visibility。\n"
        "需要 ctx: CM1.state['positions']、EX1.cell_size。"
    ),
    connections=["ex1_dust_to_pd3_visibility"],
)
def dust_visibility_field_to_per_agent_visibility(
    field_V: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_V is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "EX1")
    sampled = _sample_field_at_positions(field_V, positions, cell_size, default=1.0)
    if sampled is None:
        return None
    return np.clip(sampled, 0.0, 1.0)


@register_transform(
    name="crush_stress_field_to_per_agent_danger",
    source_schema="ndarray[H, W] float, 挤压应力 [0, 1]",
    target_schema="ndarray[n] float, [0, 1]",
    description=(
        "在每个个体位置处采样 CF1 挤压应力场，作为 PD3 的 smoke_exposure 端口输入"
        "（语义：物理挤压带来的窒息/压迫感）。\n"
        "需要 ctx: CM1.state['positions']、CF1 对应 FL4.cell_size。"
    ),
    connections=["cf1_crush_to_pd3_exposure"],
)
def crush_stress_field_to_per_agent_danger(
    field_S: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_S is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "FL4")
    sampled = _sample_field_at_positions(field_S, positions, cell_size)
    if sampled is None:
        return None
    return np.clip(sampled, 0.0, 1.0)


@register_transform(
    name="crush_stress_field_to_per_agent_impulses",
    source_schema="ndarray[H, W] float, 挤压应力 [0, 1]",
    target_schema="ndarray[n, 2] float, 冲击力 (N 量纲缩放)",
    description=(
        "将 CF1 挤压应力场在个体位置采样，并沿局部密度梯度方向生成冲击冲量，"
        "注入 CM1.external_impulses。\n"
        "需要 ctx: CM1.state['positions']、FL4.cell_size；可选读取 CF1 同网格应力。"
    ),
    connections=["cf1_crush_to_cm1_impulses"],
)
def crush_stress_field_to_per_agent_impulses(
    field_S: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if field_S is None or positions is None:
        return None
    cell_size = _get_grid_cell_size(ctx, "FL4")
    stress = _sample_field_at_positions(field_S, positions, cell_size)
    if stress is None:
        return None

    n = positions.shape[0]
    impulses = np.zeros((n, 2), dtype=np.float64)
    fld = np.asarray(field_S, dtype=np.float64)
    H, W = fld.shape

    for i in range(n):
        gr = int(np.clip(positions[i, 0] / cell_size, 1, H - 2))
        gc = int(np.clip(positions[i, 1] / cell_size, 1, W - 2))
        grad_r = fld[gr + 1, gc] - fld[gr - 1, gc]
        grad_c = fld[gr, gc + 1] - fld[gr, gc - 1]
        direction = np.array([grad_r, grad_c], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.random.randn(2) * 0.01
            norm = np.linalg.norm(direction)
        direction /= norm
        magnitude = float(stress[i]) * 1800.0
        impulses[i] = direction * magnitude

    return impulses


@register_transform(
    name="compliance_weight_to_desired_velocity",
    source_schema="ndarray[n] float, 广播服从权重 [0, 1]",
    target_schema="ndarray[n, 2] float, 期望速度 (m/s)",
    description=(
        "将 EG1 服从权重与 BC1 播报的出口方向结合 CM1 位置，生成朝备用出口的"
        "期望速度向量，写入 CM1.desired_velocity。\n"
        "需要 ctx: CM1.positions、BC1.get_announced_exit_position() 或 state。"
    ),
    connections=["eg1_compliance_to_cm1_velocity"],
)
def compliance_weight_to_desired_velocity(
    compliance: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    positions = _get_agent_positions(ctx)
    if compliance is None or positions is None or ctx is None:
        return None

    comp = np.asarray(compliance, dtype=np.float64).ravel()
    n = positions.shape[0]
    if comp.size == 1:
        comp = np.full(n, float(comp[0]), dtype=np.float64)
    elif comp.size < n:
        comp = np.concatenate([comp, np.zeros(n - comp.size)])
    comp = comp[:n]

    bc1 = ctx.simulators.get("BC1")
    if bc1 is not None and hasattr(bc1, "get_announced_exit_position"):
        exit_pos = bc1.get_announced_exit_position()
    else:
        exit_pos = np.array([0.0, 0.0], dtype=np.float64)

    base_speed = 1.4
    if hasattr(ctx.simulators.get("CM1"), "state"):
        cm1 = ctx.simulators.get("CM1")
        if cm1 is not None:
            ds = cm1.state.get("desired_speed")
            if ds is not None:
                base_speed = float(np.mean(np.asarray(ds)))

    desired = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        direction = exit_pos - positions[i]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        desired[i] = direction * base_speed * comp[i]

    return desired


@register_transform(
    name="blast_points_to_seed_field",
    source_schema="list[[row, col, intensity]] — 爆炸点 (网格坐标, [0,1])",
    target_schema="ndarray[H, W] float, 初始危害脉冲",
    description=(
        "将离散爆炸点列表映射为 EX1 所需的二维初始脉冲场。\n"
        "需要 ctx: EX1.grid_size (H, W)。"
    ),
    connections=[],
)
def blast_points_to_seed_field(
    blast_points: Any, ctx: Optional[TransformContext]
) -> Optional[np.ndarray]:
    if blast_points is None or ctx is None:
        return None
    grid_size = ctx.sim_attr("EX1", "grid_size", (50, 50))
    H, W = int(grid_size[0]), int(grid_size[1])
    seed = np.zeros((H, W), dtype=np.float64)
    for pt in blast_points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 3:
            continue
        r, c, intensity = int(pt[0]), int(pt[1]), float(pt[2])
        if 0 <= r < H and 0 <= c < W:
            seed[r, c] = max(seed[r, c], intensity)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    seed[rr, cc] = max(seed[rr, cc], intensity * 0.6)
    return seed
