# Atom_Sim Web 后端

基于 FastAPI 的桥接服务，负责：

1. 扫描 `atoms/{physical,social}/*.py`，输出每个原子仿真器的元数据（首页圆圈数据源）。
2. 解析 `config/topology_*.yaml`，给前端 react-flow 提供节点/边数据。
3. 启动 `cursor_agent`（Node）生成新场景；通过 atoms 目录的快照对比，给每个 simulator 打上 `reused / inherited / new` 标签。
4. 启动 `run_<name>.py` 跑仿真，通过 WebSocket 把 stdout 实时推送给前端，并暴露 `output_<name>/` 中的 GIF/MP4。

## 启动

```powershell
# 仓库根目录
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

## API

| 方法 | 路径 | 说明 |
| :-- | :-- | :-- |
| GET   | `/api/health`                       | 健康检查 |
| GET   | `/api/atoms`                        | 列出全部原子仿真器（docstring + class） |
| GET   | `/api/scenarios`                    | 列出 `config/topology_*.yaml` |
| GET   | `/api/scenarios/{name}`             | 拓扑详情 + simulators 分类 |
| POST  | `/api/generate`                     | 启动 `cursor_agent` 非交互生成 |
| POST  | `/api/run`                          | 启动 `run_{name}.py` |
| GET   | `/api/jobs` / `/api/jobs/{id}`      | 任务列表 / 详情 |
| POST  | `/api/jobs/{id}/cancel`             | 取消任务 |
| WS    | `/api/jobs/{id}/stream`             | 订阅流式日志 |
| GET   | `/api/results/{name}`               | 列出 `output_{name}/` 中的媒体文件 |
| GET   | `/api/output/{name}/{filename}`     | 下载 / 流式输出某个文件 |

## 依赖现有项目

- `cursor_agent/` 必须先 `npm install` 一遍，且 `cursor_agent/config.json` 已经填好 `apiKey` / `model`（否则 `POST /api/generate` 会失败，但日志会通过 WebSocket 透传给前端）。
- `run_<name>.py` 必须接受 `-c <yaml>` 参数（项目里 `run_fire.py` / `run_grid.py` 已支持）。

## 设计要点

- **快照分类**：每次 `POST /api/generate` 之前，后端记录 `atoms/{physical,social}/*.py` 的当前文件集合。Agent 结束后再次扫描，新出现的文件视为本次新增；解析其顶层 class 的父类，若父类是另一个已存在的仿真器类，则标记为 `inherited`，否则 `new`。
- **运行时 yaml**：`POST /api/run` 不会覆盖原始 `topology_*.yaml`，而是写到 `config/_runtime_<name>.yaml`，把前端的 scenario override 浅合并进去，再传给 `run_<name>.py`。
