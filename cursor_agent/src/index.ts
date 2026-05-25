/**
 * Atom_Sim × Cursor SDK 集成入口。
 *
 * 支持两种运行模式：
 *   1) 交互式（默认）：readline 询问场景名 + 多行背景
 *   2) 非交互式：通过 CLI 参数 / 环境变量传入，便于被前端后端 spawn 调用
 *
 * 非交互式 CLI：
 *   npm start -- --non-interactive \
 *     --scenario-name "智能电网负载均衡" \
 *     --background-file ./bg.txt        # 每行一条
 *
 * 也支持：
 *   --background "条目1" --background "条目2" ...
 *   或者环境变量 ATOMSIM_SCENARIO_NAME / ATOMSIM_BACKGROUND_FILE
 *
 * 流程：
 *   1) 读取 cursor_agent/config.json（apiKey / model / 路径等）。
 *   2) 收集场景名 + 背景（交互式或 CLI）。
 *   3) 使用本地 Runtime 启动 Cursor Agent（cwd 指向 Atom_Sim 项目根目录）。
 *   4) 流式打印 assistant 文本 / 工具调用 / 状态。
 *   5) 结束时写入 cursor_agent/last_run.json 便于后端定位生成的 yaml。
 */

import { readFileSync, existsSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface, type Interface as ReadlineInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { Agent, CursorAgentError } from "@cursor/sdk";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PKG_ROOT = resolve(__dirname, "..");

// ----------------------------------------------------------
// 配置类型与加载
// ----------------------------------------------------------

type SettingSource = "project" | "user" | "team" | "mdm" | "plugins" | "all";

interface ModelParam {
  id: string;
  value: string;
}

interface ModelSelectionConfig {
  id: string;
  params?: ModelParam[];
}

interface CursorAgentConfig {
  apiKey?: string;
  model?: string | ModelSelectionConfig;
  projectRoot?: string;
  promptTemplatePath?: string;
  settingSources?: SettingSource[];
  showThinking?: boolean;
  showToolCalls?: boolean;
}

function loadConfig(): CursorAgentConfig {
  const configPath = resolve(PKG_ROOT, "config.json");
  if (!existsSync(configPath)) {
    throw new Error(
      `[配置错误] 未找到 ${configPath}\n` +
        `请从 cursor_agent/config.example.json 复制一份为 config.json 并填写 apiKey / model。`,
    );
  }
  try {
    return JSON.parse(readFileSync(configPath, "utf-8")) as CursorAgentConfig;
  } catch (err) {
    throw new Error(`[配置错误] 解析 config.json 失败：${(err as Error).message}`);
  }
}

function normalizeModel(model: CursorAgentConfig["model"]): ModelSelectionConfig {
  if (!model) return { id: "composer-2" };
  if (typeof model === "string") return { id: model };
  if (!model.id) throw new Error("[配置错误] model.id 不能为空");
  return model;
}

// ----------------------------------------------------------
// CLI 参数解析
// ----------------------------------------------------------

interface CliInputs {
  nonInteractive: boolean;
  scenarioName?: string;
  backgroundLines: string[];
  backgroundFile?: string;
}

function parseCliArgs(argv: string[]): CliInputs {
  const result: CliInputs = {
    nonInteractive: false,
    backgroundLines: [],
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--non-interactive" || a === "-N") {
      result.nonInteractive = true;
    } else if (a === "--scenario-name" || a === "-s") {
      result.scenarioName = argv[++i];
    } else if (a === "--background-file" || a === "-f") {
      result.backgroundFile = argv[++i];
    } else if (a === "--background" || a === "-b") {
      const v = argv[++i];
      if (v) result.backgroundLines.push(v);
    }
  }
  if (!result.scenarioName && process.env.ATOMSIM_SCENARIO_NAME) {
    result.scenarioName = process.env.ATOMSIM_SCENARIO_NAME;
  }
  if (!result.backgroundFile && process.env.ATOMSIM_BACKGROUND_FILE) {
    result.backgroundFile = process.env.ATOMSIM_BACKGROUND_FILE;
  }
  // 若给了 scenarioName 但没显式声明 --non-interactive，也视为非交互
  if (result.scenarioName && !result.nonInteractive) {
    result.nonInteractive = true;
  }
  return result;
}

function readBackgroundFile(path: string): string[] {
  const text = readFileSync(path, "utf-8");
  return text
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

// ----------------------------------------------------------
// 交互式输入：场景名 + 多条场景背景
// ----------------------------------------------------------

async function ask(rl: ReadlineInterface, prompt: string): Promise<string> {
  const ans = await rl.question(prompt);
  return ans.trim();
}

async function collectBackgroundLines(rl: ReadlineInterface): Promise<string[]> {
  console.log("");
  console.log("请逐条输入【场景背景】，每行一条；直接回车（空行）即结束：");
  const items: string[] = [];
  while (true) {
    const idx = items.length + 1;
    const line = await ask(rl, `  [${idx}] `);
    if (!line) break;
    items.push(line);
  }
  return items;
}

function buildPrompt(template: string, scenarioName: string, backgroundLines: string[]): string {
  if (!template.includes("[1]")) {
    throw new Error("[模板错误] prompt 模板里找不到占位符 [1]");
  }
  if (!template.includes("[2]")) {
    throw new Error("[模板错误] prompt 模板里找不到占位符 [2]");
  }
  const backgroundBlock = backgroundLines.map((line, i) => `${i + 1}. ${line}`).join("\n");
  return template.replace("[1]", scenarioName).replace("[2]", backgroundBlock);
}

// ----------------------------------------------------------
// 流式事件渲染
// ----------------------------------------------------------

interface StreamRenderOptions {
  showThinking: boolean;
  showToolCalls: boolean;
}

function renderStreamEvent(event: any, opts: StreamRenderOptions): void {
  switch (event.type) {
    case "assistant": {
      const content = event.message?.content ?? [];
      for (const block of content) {
        if (block.type === "text" && typeof block.text === "string") {
          process.stdout.write(block.text);
        }
      }
      break;
    }
    case "thinking": {
      if (opts.showThinking && typeof event.text === "string") {
        process.stdout.write(`\n[thinking] ${event.text}\n`);
      }
      break;
    }
    case "tool_call": {
      if (!opts.showToolCalls) break;
      const name = event.name ?? "(tool)";
      const status = event.status ?? "?";
      const tag = status === "completed" ? "ok" : status === "error" ? "fail" : status;
      process.stdout.write(`\n[tool] ${name} -> ${tag}\n`);
      break;
    }
    case "status": {
      process.stdout.write(`\n[status] ${event.status}${event.message ? ": " + event.message : ""}\n`);
      break;
    }
    case "task": {
      if (event.text) process.stdout.write(`\n[task] ${event.text}\n`);
      break;
    }
    default:
      break;
  }
}

// ----------------------------------------------------------
// 结束后写 last_run.json：扫 config/ 找最新的 topology_*.yaml
// ----------------------------------------------------------

function detectLatestTopology(projectRoot: string): { topologyYaml?: string; runScript?: string } {
  const configDir = join(projectRoot, "config");
  let latest: { file: string; mtime: number } | null = null;
  try {
    for (const name of readdirSync(configDir)) {
      if (!/^topology_.*\.ya?ml$/i.test(name)) continue;
      const full = join(configDir, name);
      const m = statSync(full).mtimeMs;
      if (!latest || m > latest.mtime) latest = { file: name, mtime: m };
    }
  } catch {
    return {};
  }
  if (!latest) return {};
  const stem = latest.file.replace(/\.ya?ml$/i, "").replace(/^topology_/, "");
  const runScript = `run_${stem}.py`;
  const runPath = join(projectRoot, runScript);
  return {
    topologyYaml: latest.file,
    runScript: existsSync(runPath) ? runScript : undefined,
  };
}

function writeLastRun(payload: Record<string, unknown>): void {
  const path = resolve(PKG_ROOT, "last_run.json");
  try {
    writeFileSync(path, JSON.stringify(payload, null, 2), "utf-8");
  } catch (err) {
    console.warn(`[警告] 无法写入 last_run.json: ${(err as Error).message}`);
  }
}

// ----------------------------------------------------------
// 主流程
// ----------------------------------------------------------

async function main(): Promise<void> {
  const cli = parseCliArgs(process.argv.slice(2));
  const cfg = loadConfig();
  const apiKey = (cfg.apiKey && cfg.apiKey.trim()) || process.env.CURSOR_API_KEY;
  if (!apiKey) {
    throw new Error(
      "[认证错误] 未找到 Cursor API Key。请在 cursor_agent/config.json 中填写 apiKey，" +
        "或设置环境变量 CURSOR_API_KEY。",
    );
  }

  const projectRoot = resolve(PKG_ROOT, cfg.projectRoot ?? "..");
  if (!existsSync(projectRoot)) {
    throw new Error(`[路径错误] projectRoot 不存在: ${projectRoot}`);
  }

  const templatePath = resolve(PKG_ROOT, cfg.promptTemplatePath ?? "./prompt_template.txt");
  if (!existsSync(templatePath)) {
    throw new Error(`[模板错误] 找不到 prompt 模板: ${templatePath}`);
  }
  const template = readFileSync(templatePath, "utf-8");

  let scenarioName = "";
  let backgroundLines: string[] = [];
  let prompt: string;

  if (cli.nonInteractive) {
    // -------- 非交互式 --------
    console.log("============================================================");
    console.log("  Atom_Sim × Cursor Agent —— 联合仿真场景生成 (非交互式)");
    console.log("============================================================");
    if (!cli.scenarioName) {
      throw new Error("[输入错误] 非交互模式需要 --scenario-name 或 ATOMSIM_SCENARIO_NAME");
    }
    scenarioName = cli.scenarioName.trim();
    if (cli.backgroundFile) {
      backgroundLines = readBackgroundFile(cli.backgroundFile);
    }
    if (cli.backgroundLines.length > 0) {
      backgroundLines.push(...cli.backgroundLines);
    }
    if (backgroundLines.length === 0) {
      throw new Error(
        "[输入错误] 非交互模式需要至少一条背景：用 --background-file <文件> 或多个 --background <条目>",
      );
    }
    console.log(`[输入] scenarioName="${scenarioName}"`);
    console.log(`[输入] backgroundLines:`);
    for (let i = 0; i < backgroundLines.length; i++) {
      console.log(`  [${i + 1}] ${backgroundLines[i]}`);
    }
    prompt = buildPrompt(template, scenarioName, backgroundLines);
  } else {
    // -------- 交互式 (兜底) --------
    const rl = createInterface({ input, output });
    try {
      console.log("============================================================");
      console.log("  Atom_Sim × Cursor Agent —— 联合仿真场景生成");
      console.log("============================================================");

      scenarioName = await ask(rl, "请输入【联合仿真场景名】（将填入 prompt 模板 [1]）: ");
      if (!scenarioName) throw new Error("[输入错误] 场景名不能为空");

      backgroundLines = await collectBackgroundLines(rl);
      if (backgroundLines.length === 0) {
        throw new Error("[输入错误] 至少需要输入一条场景背景");
      }
      prompt = buildPrompt(template, scenarioName, backgroundLines);
    } finally {
      rl.close();
    }
  }

  console.log("");
  console.log("------------------ 最终 Prompt 预览 ------------------");
  console.log(prompt);
  console.log("------------------------------------------------------");
  console.log("");

  const modelSelection = normalizeModel(cfg.model);
  const localOptions: Record<string, unknown> = { cwd: projectRoot };
  if (cfg.settingSources && cfg.settingSources.length > 0) {
    localOptions.settingSources = cfg.settingSources;
  }

  console.log(
    `[启动] model=${modelSelection.id}` +
      (modelSelection.params?.length ? ` params=${JSON.stringify(modelSelection.params)}` : "") +
      ` cwd=${projectRoot}`,
  );

  const agent = await Agent.create({
    apiKey,
    model: modelSelection,
    local: localOptions as any,
  });
  console.log(`[agent] id=${agent.agentId}`);

  const renderOpts: StreamRenderOptions = {
    showThinking: !!cfg.showThinking,
    showToolCalls: cfg.showToolCalls !== false,
  };

  let exitStatus = "unknown";
  try {
    const run = await agent.send(prompt);
    console.log(`[run] id=${run.id}`);
    console.log("");
    console.log("------------------ Agent Streaming ------------------");

    for await (const event of run.stream()) {
      renderStreamEvent(event, renderOpts);
    }

    const result = await run.wait();
    console.log("");
    console.log("------------------------------------------------------");
    console.log(`[run] status=${result.status} duration=${result.durationMs ?? 0}ms`);
    exitStatus = result.status;

    if (result.status === "error") {
      console.error("[错误] Agent run 进入 error 状态，请检查上方流式输出。");
      process.exitCode = 2;
    } else if (result.status === "cancelled") {
      console.warn("[警告] Agent run 被取消。");
      process.exitCode = 3;
    } else {
      console.log("[完成] Agent run 成功结束，生成的文件已写入项目目录。");
      console.log(`        项目根: ${projectRoot}`);
    }
  } catch (err) {
    if (err instanceof CursorAgentError) {
      console.error(
        `\n[启动失败] ${err.message} (code=${err.code ?? "?"} retryable=${err.isRetryable})`,
      );
      process.exitCode = 1;
      exitStatus = "launch_error";
    } else {
      throw err;
    }
  } finally {
    const detected = detectLatestTopology(projectRoot);
    writeLastRun({
      scenarioName,
      backgroundLines,
      topologyYaml: detected.topologyYaml ?? null,
      runScript: detected.runScript ?? null,
      exitStatus,
      finishedAt: new Date().toISOString(),
    });
    await agent[Symbol.asyncDispose]();
  }
}

main().catch((err) => {
  console.error("\n[未捕获异常]", err);
  process.exit(1);
});
