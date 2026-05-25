# Atom_Sim Web 前端

React + Vite + TypeScript 实现的联合仿真可视化界面，配合 `backend/` (FastAPI) 与 `cursor_agent/` (Cursor SDK) 一起工作。

## 启动

```powershell
# 仓库根目录一次性
pip install -r ../backend/requirements.txt
npm install                                # 在 frontend/ 下
cd ../cursor_agent && npm install && cd ..

# 启动后端（仓库根）
python -m uvicorn backend.main:app --reload --port 8000

# 启动前端（frontend/）
npm run dev   # http://localhost:5173
```

或直接在仓库根用：

```powershell
pwsh ./start_web.ps1 -Install   # 安装并启动
pwsh ./start_web.ps1            # 只启动
```

## 功能一览

- **首页 `/`**
  - 圆圈网格列出 `atoms/physical/` 和 `atoms/social/` 下所有原子仿真器；
  - 点击圆 → 右侧面板显示该原子的文件 docstring（即文件开头注释）；
  - 入口卡片：
    - `+ 生成联合仿真环境`：弹出场景名 / 多行背景表单 → 调用 `POST /api/generate` → 后端 spawn `cursor_agent/npm start -- --non-interactive ...` → 流式显示日志 → 完成后跳转到对应 `/scenario/<name>`；
    - `run_fire · 建筑火灾疏散` 与 `run_grid · 智能电网负载均衡` 两个内置演示，跳转到对应场景视图；
    - 之前由 cursor_agent 生成的拓扑也会作为已生成卡片出现。

- **场景视图 `/scenario/<name>`**
  - 使用 `@xyflow/react` 把 `config/topology_<name>.yaml` 的 simulators 渲染为圆形节点，按 `time_groups` 自动分列布局；
  - 三种颜色对应三种来源：复用（蓝） / 继承（橙） / 新增（绿）；
  - 连接（S2S）以箭头绘制，标签为 transform 名；`EVENT` 策略的边为虚线 + 流光；
  - 点击圆 → 右侧显示该 simulator 的 class、group、params、文件路径与 docstring；
  - 点击箭头 → 右侧显示 connection 的 source/target/strategy/transform/description；
  - 顶部 `运行联合仿真` 按钮 → 弹出 scenario 参数配置 → `POST /api/run` → 流式日志 → 自动加载 `output_<name>/` 中的 GIF/MP4 并播放。

## 与后端的接口

详见 [`backend/main.py`](../backend/main.py) 中的 FastAPI 路由列表，以及 [`src/api.ts`](src/api.ts) 中的封装。
