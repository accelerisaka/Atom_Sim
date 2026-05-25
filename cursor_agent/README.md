# Atom_Sim × Cursor Agent

将 Cursor 的 TypeScript SDK (`@cursor/sdk`) 集成到本仓库，让 Cursor Agent 按照 `core/`
下已有的 `AtomicSimulator` 协议、跨速率交换策略与 `transforms.py` 集中转换规范，
自动为你生成一个**全新的联合仿真场景**所需的：

- 物理原子 / 社会-心理原子（继承 `core.base.AtomicSimulator`）
- 追加到 `core/transforms.py` 末尾的 `@register_transform` 函数
- 全新的 `config/topology_grid.yaml`（每条 S2S 连接均显式声明 `transform`）
- 可执行 + 可视化的入口 `run_xxx.py`（matplotlib GIF / MP4）

Agent 在**本地运行时**直接读取与写入本仓库的源码，不上云。

---

## 目录结构

```
cursor_agent/
├── package.json
├── tsconfig.json
├── config.example.json        # 配置样板
├── config.json                # 真实配置（已 gitignore，请填入你的 API Key）
├── prompt_template.txt        # 给 Cursor Agent 的提示词模板，包含 [1] [2] 占位符
├── src/
│   └── index.ts               # 入口：交互式收集场景信息 → 启动本地 Cursor Agent
└── .gitignore
```

---

## 1. 安装依赖

需要 Node.js ≥ 18（建议 20+）。

```powershell
cd cursor_agent
npm install
```

> 第一次安装会下载 `@cursor/sdk`、`tsx`、`typescript`、`@types/node`。

---

## 2. 配置 API Key 与模型

编辑 `cursor_agent/config.json`：

```json
{
  "apiKey": "cursor_xxx_你的密钥",
  "model": {
    "id": "composer-2",
    "params": [
      { "id": "thinking", "value": "high" }
    ]
  },
  "projectRoot": "..",
  "promptTemplatePath": "./prompt_template.txt",
  "settingSources": [],
  "showThinking": false,
  "showToolCalls": true
}
```

| 字段 | 说明 |
| :-- | :-- |
| `apiKey` | Cursor 用户 / 服务账户 API Key（来自 [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations)）。留空时回退到环境变量 `CURSOR_API_KEY`。 |
| `model.id` | 模型 ID。常用 `composer-2` / `auto`。可先调用 `Cursor.models.list()` 查询当前账户可用的模型。 |
| `model.params` | 模型参数（如 `thinking=high / low`）。不同模型支持的参数不同。 |
| `projectRoot` | Atom_Sim 仓库根目录（相对 `cursor_agent/`，默认 `..`）。Agent 的 `cwd` 即此路径。 |
| `promptTemplatePath` | prompt 模板路径（相对 `cursor_agent/`）。 |
| `settingSources` | 本地设置来源；默认 `[]`（不加载本机 `~/.cursor/` 或项目 `.cursor/` 设置，安全干净）。可选 `"project"`/`"user"`/`"plugins"`/`"all"`。 |
| `showThinking` | 是否在终端打印模型的 thinking 流。 |
| `showToolCalls` | 是否打印 Agent 工具调用摘要（默认 `true`）。 |

> 也可以不写 `apiKey` 字段，改在 PowerShell 里设：
>
> ```powershell
> $env:CURSOR_API_KEY = "cursor_xxx"
> ```

---

## 3. 运行

```powershell
cd cursor_agent
npm start
```

随后会进入交互：

1. `请输入【联合仿真场景名】` → 单行回答，将被填入 prompt 模板的 `[1]`。
2. `请逐条输入【场景背景】` → 每行一条；回车空行结束。脚本会自动按 `1. … 2. …` 编号填入 `[2]`。

示例：

```
请输入【联合仿真场景名】（将填入 prompt 模板 [1]）: 智能电网负载均衡

请逐条输入【场景背景】，每行一条；直接回车（空行）即结束：
  [1] 夏季高温日，户外大量空调开启
  [2] 局部变压器接近容量上限，可能过载
  [3] 动态电价随负载实时上升
  [4] 用户对电价敏感，会上调空调设定温度
  [5]
```

然后脚本会拼好最终 prompt（打印一遍预览），创建本地 Agent，并实时打印流式输出。

Agent run 结束后：

- 退出码 `0`：成功生成所需文件。
- 退出码 `1`：Cursor SDK 启动失败（认证 / 网络 / 配置）。
- 退出码 `2`：Agent 启动成功但 run 过程出错（看上方流式输出）。
- 退出码 `3`：被取消。

---

## 4. Agent 会写入哪里？

由于 `local.cwd` 已经指向 Atom_Sim 项目根目录，Agent 会直接：

- 在 `atoms/physical/` / `atoms/social/` 下**新建**原子文件
- 在 `core/transforms.py` **末尾追加** `@register_transform` 函数（禁止改前文）
- 在 `config/` 下创建新的 `topology_grid.yaml`（若同名已存在，会覆盖 / 改写）
- 在仓库根目录创建 `run_xxx.py`（xxx 由 Agent 命名）

> prompt 模板里已经把"绝对架构规则"喂给 Agent：
> 禁止修改 `main.py` / `core/` 已有文件（除 `transforms.py` 仅允许追加），
> 不准在 `atoms/` 现有文件上动手，全部跨原子翻译必须经 `register_transform`。

---

## 5. 常见问题

**Q: 报错"未找到 Cursor API Key"？**
A: 在 `config.json` 写 `apiKey` 或设环境变量 `CURSOR_API_KEY`。

**Q: 报错 `AuthenticationError`？**
A: API Key 失效或粘贴时带了空白字符。重新生成并粘贴。

**Q: 想看模型可选列表？**
A: 在 Node REPL 里跑：
```ts
import { Cursor } from "@cursor/sdk";
console.log(await Cursor.models.list({ apiKey: process.env.CURSOR_API_KEY }));
```

**Q: 想换 prompt 模板？**
A: 直接编辑 `cursor_agent/prompt_template.txt`，保留 `[1]` `[2]` 占位符即可。
   也可以在 `config.json` 把 `promptTemplatePath` 指向别的文件。

**Q: 想多次复用同一个 Agent？**
A: 当前入口是"一问一答即销毁"。如需多轮对话，把 `Agent.create` 改为 `Agent.resume(agentId, ...)` 并保存 `agent.agentId` 即可。

---

## 6. PowerShell 中文乱码

如果 PowerShell 终端出现中文乱码，先把控制台切到 UTF-8 再启动：

```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
npm start
```

---

## 7. 不修改原有架构

本目录是**独立子工程**：

- 不污染仓库根的 Python 依赖（`requirements.txt` 完全不变）
- 不修改 `core/` 任何文件
- `config.json` 已通过 `.gitignore` 排除，避免 API Key 入库
