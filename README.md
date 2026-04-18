# 建筑火灾疏散联合仿真系统 (Building Fire Evacuation Co-Simulation)

一套基于 **原子仿真器 (AtomicSimulator)** 架构的多物理场 + 社会心理耦合联合仿真系统。  
系统通过 **S2S (Simulator-to-Simulator) 数据总线**，将 10 个异构仿真器在 4 种时间尺度上编排运行，模拟建筑火灾中人员疏散的完整过程。

---

## 目录

- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [架构设计](#架构设计)
- [仿真器清单](#仿真器清单)
- [S2S 拓扑连接](#s2s-拓扑连接)
- [跨速率交换策略](#跨速率交换策略)
- [配置说明](#配置说明)
- [可视化输出](#可视化输出)

---

## 核心特性

- **原子化设计** — 每个仿真器内部自洽、外部解耦，通过 `step()` / `get_outputs()` 统一接口交互
- **多速率编排** — 4 个时间分组 (0.02s / 0.1s / 0.5s / Event)，Orchestrator 自动对齐时序
- **4 种跨速率策略** — ZOH (零阶保持)、AVG (时间平均)、INTERP (线性插值)、EVENT (事件脉冲)
- **因果一致性守卫** — 自动检测代数环 (Tarjan SCC)，防止未来数据引用
- **事件驱动中断** — 优先事件队列 (PEQ) 支持跌倒/温度突变等突发事件的零延迟响应
- **配置驱动** — 全部拓扑连接、仿真参数通过 YAML 文件定义
- **可视化** — 四面板渲染 (温度场 / 烟雾场 / 人群位置 / 恐慌分布) + JSON 历史导出

---

## 项目结构

```
Atom_Sim/
├── main.py                         # 主入口：加载配置 → 运行仿真 → 输出结果
├── requirements.txt                # Python 依赖
├── config/
│   └── topology.yaml               # S2S 拓扑连接 + 场景参数配置
├── core/                           # 核心框架
│   ├── base.py                     # AtomicSimulator 抽象基类
│   ├── protocol.py                 # S2SConnection / EventMessage / ExchangeStrategy
│   ├── strategies.py               # 跨速率交换策略引擎 (ZOH/AVG/INTERP/EVENT)
│   ├── causality.py                # CausalityGuard 因果一致性守卫 + TimeGroup
│   └── orchestrator.py             # Orchestrator 多速率编排器 (PEQ + 同步管道)
├── atoms/
│   ├── physical/                   # 物理原子仿真器
│   │   ├── td1_heat.py             # TD1: 热传导 (傅里叶定律, FDM)
│   │   ├── fl3_smoke.py            # FL3: 烟雾扩散 (菲克定律, FDM)
│   │   ├── fl4_crowd_fluid.py      # FL4: 宏观人流密度 (连续性方程)
│   │   ├── cm1_rigid_body.py       # CM1: 微观碰撞动力学 (社会力模型)
│   │   └── p6_bottleneck.py        # P6: 瓶颈通行约束 (水流容量模型)
│   └── social/                     # 社会心理原子仿真器
│       ├── pd3_emotion.py          # PD3: 恐慌情绪 (Scherer CPM)
│       ├── bp6_stress.py           # BP6: 压力-表现曲线 (Yerkes-Dodson)
│       ├── s2_herd.py              # S2: 从众行为 (社会干扰理论)
│       ├── sd4_bystander.py        # SD4: 旁观者效应 (责任扩散, 事件驱动)
│       └── bp5_attention.py        # BP5: 注意力分配 (认知资源限制)
├── visualization/
│   └── renderer.py                 # 四面板可视化渲染器 + JSON 导出
└── output/                         # 仿真输出 (运行后生成)
    ├── frame_XXXX.png              # 可视化帧图像
    └── history.json                # 全量状态历史 (供前端消费)
```

---

## 快速开始

### 环境要求

- Python >= 3.9
- 依赖：numpy, pyyaml, matplotlib

### 安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行默认场景 (60秒仿真)
python main.py

# 3. 指定自定义配置
python main.py -c config/topology.yaml
```

运行后将在 `output/` 目录下生成：
- `history.json` — 全量仿真状态历史，可供前端可视化渲染
- `frame_XXXX.png` — 四面板快照图像（温度/烟雾/人群/恐慌）

---

## 架构设计

### 原子仿真器基类

所有仿真器继承自 `AtomicSimulator`，遵循统一契约：

| 方法 | 职责 |
|------|------|
| `step(dt)` | 基于当前 `inputs` 和 `state` 推演 dt 时间，仅修改 `state` |
| `get_outputs()` | 暴露给 S2S 总线的数据快照 |
| `schema()` | 自描述接口，声明 Inputs/Outputs 字段 |
| `record_output()` | 将输出压入历史队列（供 AVG/INTERP 策略使用）|

### Orchestrator 执行管道

每个 Global Tick 严格按五阶段执行：

1. **寻址同步点** — 找到各 TimeGroup 的最近步进时刻或 PEQ 中最近事件
2. **事件消化** — 处理已到期的 EVENT 脉冲（冻结时钟，零时间推演）
3. **分组追赶** — 各时间组独立步进至同步点
4. **状态拉取** — 收集所有仿真器输出到全局总线
5. **策略注入** — 按连接声明的策略 (ZOH/AVG/INTERP) 计算并注入目标 Inputs

---

## 仿真器清单

### 物理原子 (Physical Atoms)

| ID  | 名称 | 物理规律 | dt |
|-----|------|----------|------|
| TD1 | HeatConduction | 傅里叶导热定律 ∂T/∂t = α∇²T | 0.5s |
| FL3 | Diffusion2D | 菲克扩散定律 ∂C/∂t = D∇²C | 0.2s |
| FL4 | CrowdFluid | 连续性方程 ∂ρ/∂t + ∇·(ρv) = 0 | 0.1s |
| CM1 | RigidBody2D | 牛顿第二定律 F=ma (社会力模型) | 0.02s |
| P6  | BottleneckGate | 容量约束 Q = min(Q_in, C) | 0.1s |

### 社会心理原子 (Social/Psychological Atoms)

| ID  | 名称 | 理论依据 | dt |
|-----|------|----------|------|
| PD3 | EmotionAppraisal | Scherer 认知评估模型 (CPM) | 0.5s |
| BP6 | StressPerformanceCurve | 耶克斯-多德森定律 | 0.5s |
| S2  | HerdBehavior | 社会干扰理论 | 0.1s |
| SD4 | BystanderEffect | 责任扩散 P(help) ∝ 1/√N | Event |
| BP5 | AttentionAllocator | 认知资源有限理论 | 0.1s |

---

## S2S 拓扑连接

系统包含 **17 条** S2S 连接，形成 4 大数据流回路：

```
火灾环境 → 心理:  TD1/FL3  ──AVG──▶  PD3/BP5
心理 → 物理:      PD3 → BP6 ──INTERP──▶ CM1
社会互动:         CM1 ◄──ZOH──▶ S2,  CM1 ──EVENT──▶ SD4
物理反馈:         CM1 → FL4 → P6 → CM1 (闭环)
                  TD1 → FL3 (热-烟耦合)
```

---

## 跨速率交换策略

| 策略 | 场景 | 示例 |
|------|------|------|
| **ZOH** (零阶保持) | 快读慢，离散控制信号 | P6 (0.1s) → CM1 (0.02s) |
| **AVG** (时间平均) | 慢读快，防高频噪音 | FL3 (0.2s) → PD3 (0.5s) |
| **INTERP** (线性插值) | 快读慢，需平滑导数 | BP6 (0.5s) → CM1 (0.02s) |
| **EVENT** (事件脉冲) | 突发离散事件 | CM1 ↔ SD4 (跌倒/救助) |

---

## 配置说明

所有参数在 `config/topology.yaml` 中以声明式定义：

- **time_groups** — 4 个时间分组 (micro/meso/macro/event)
- **simulators** — 10 个仿真器的类名、所属组、构造参数
- **connections** — 17 条 S2S 连接的源端口、目标端口、交换策略
- **scenario** — 场景初始条件（着火点、出口位置、墙壁布局、仿真时长）

修改 YAML 即可调整仿真场景，无需改动代码。

---

## 可视化输出

每帧生成四面板图像：

| 面板 | 内容 |
|------|------|
| 左上 | TD1 温度场热力图（黑→红→橙→黄→白） |
| 右上 | FL3 烟雾浓度场（白→灰→黑） |
| 左下 | CM1 人群位置散点图（蓝=行走 / 红×=跌倒 / 绿=逃出） |
| 右下 | PD3 恐慌值柱状图（绿→黄→红） |

同时导出 `history.json`，每个快照包含所有仿真器的完整状态输出，可对接任意前端渲染框架。
