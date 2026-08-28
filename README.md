# NJU-CodingAgent

一个从零实现的本地 TUI Coding Agent。项目首先建立稳定、可追踪的 Baseline Agent，随后通过 Capability Profiling，在同一个 Agent Loop 内进行 Step-level 模型路由。

## 当前阶段

当前开发阶段为 **P0-3 Observability**。P0-0 工程骨架、P0-1 流式运行时与 P0-2 Workspace Tool Loop 已经完成；当前 Baseline 包含：

- 统一 LLM Request/Response/Tool Call 类型
- OpenAI-compatible Chat Completions 适配器
- `SessionState` 与最小 Agent Loop
- 受 Workspace 边界保护的完整文件与命令工具链
- FakeLLM 确定性双轮闭环
- Textual 全屏终端界面和 Slash Command
- 用户目录下的本地模型与凭据配置
- OpenAI-compatible SSE 流式输出和增量工具调用组装
- 用户、Agent、工具与系统状态分层消息视图
- 读后再写、过期读取检测、原子替换与换行格式保持
- 受控测试执行、只读 Git diff、循环检测与工具调用数量限制
- 当前窗口连续上下文与本地会话恢复
- Provider 中立事件、StepTrace、Token/Cost/Latency 与本地运行审计

Context Snapshot 和 Anthropic 原生协议将在后续独立模块中实现。

## 本地安装

要求 Python 3.11+。使用 uv 安装开发依赖：

```powershell
py -3.11 -m uv sync --extra dev
```

将开发版本注册为当前用户的终端命令（只需执行一次）：

```powershell
py -3.11 -m uv tool install --python 3.11 --editable .
```

之后可在任意新终端直接启动：

```powershell
capycode
```

也可以从当前代码目录直接恢复最近会话或指定会话：

```powershell
capycode --continue
capycode --resume <会话 ID>
```

查看当前代码目录最近的运行记录和某次运行详情：

```powershell
capycode runs
capycode inspect-run <run ID 或唯一前缀>
```

`capycode` 会直接进入终端交互界面。输入普通文本执行任务，输入 `/` 展开命令菜单。当前提供：

- `/help`：查看命令说明
- `/config`：填写 Base URL 和 API Key，自动获取模型列表后选择模型
- `/models`、`/model [model-id]`：查看真实模型列表并打开键盘选择器
- `/pricing`：为当前真实模型配置输入/输出价格、币种和上下文窗口
- `/workspace [path]`：查看和切换工作区
- `/resume [会话 ID]`、`/continue`：选择历史会话或继续最近会话
- `/sessions`、`/new`：列出历史会话或开始新会话
- `/runs`：用键盘选择运行记录，并查看步骤、工具、Token、费用和最终 diff
- `/status`、`/clear`、`/quit`：会话控制

Slash Command 菜单支持方向键移动、Tab 补全和 Esc 关闭；普通输入支持方向键查找本次会话的历史任务。`/model` 打开独立模型选择面板，方向键选择、Enter 确认、Esc 取消。

模型请求采用真实 SSE 流式传输：首个 Token 返回前显示轻量思考动画，开始输出后原位更新同一个 Markdown 消息；工具调用显示进行中、成功或失败状态，并可用鼠标、Enter 或空格展开参数、耗时、退出码和有界输出。启动时显示短暂的 CapyCode 封面，进入会话后使用随终端宽度折叠的状态栏，实时展示 Run、Token、费用和耗时。Ctrl+C 会先取消当前任务、补全 Trace 并恢复输入焦点，再次 Ctrl+C 可退出。

`/config` 会按照 OpenAI-compatible 协议请求 `<Base URL>/models`。CapyCode 直接使用并保存服务端返回的真实模型 ID，不再建立 `small`、`medium`、`strong` 等人工分级。直接执行 `/model` 可使用方向键切换真实模型；`/pricing` 为当前模型独立保存每百万 Token 输入/输出价格、币种、价格日期和上下文窗口。后续能力等级由 P1/P2 的能力测试与 Profile 生成，不根据模型名称猜测。本地配置保存在 `~/.capycode/settings.json`，不会写入项目仓库；旧版别名配置在首次读取时自动迁移。项目名 **CapyCode** 来自 Capability + Code，并与源码包 `capycode`、运行产物目录 `.capy` 保持一致。

同一终端中的后续任务会沿用当前模型上下文。会话在每次用户消息、模型响应和工具结果后增量保存到 `~/.capycode/sessions/`；关闭终端后，在同一代码目录再次启动 `capycode`，输入 `/resume` 可通过键盘选择历史会话，输入 `/continue` 可直接继续最近会话。默认启动新会话，不会在未确认时自动加载历史内容；`/new` 会开始新会话，但不会删除历史记录。恢复范围严格限制为当前工作区，API Key 不进入会话文件。中断产生的未配对工具消息会在恢复时清理，文件会要求重新读取，旧终端中的后台进程不会恢复。

每次用户任务都会创建新的 Run，并在 `<workspace>/.capy/runs/<run-id>/` 下生成追加式 `trace.jsonl` 和原子写入的 `summary.json`。恢复会话后继续沿用原 `session_id`，但使用新的 `run_id`。Trace 记录事件顺序、模型与工具延迟、Token、按配置估算的费用、测试状态、修改文件和终止原因；API Key、Authorization、Bearer Token 等敏感值在写入前统一脱敏。异常和取消路径同样生成 Summary，已发出的工具请求会补齐结构化失败结果。

Linux/WSL：

```bash
uv sync --extra dev
```

## Workspace 工具与安全约束

模型当前可以调用：

- `list_files`、`search_code`、`read_file`
- `write_file`、`replace_text`
- `run_command`、`run_tests`
- `process_status`、`stop_process`
- `git_diff`

所有文件路径都必须位于当前工作区。绝对路径、UNC 网络路径、路径穿越和越界符号链接会被拒绝；`.git`、`.venv`、`node_modules` 等生成目录不会进入文件发现结果。

修改已有文件前必须先完整调用 `read_file`。CapyCode 会记录内容摘要、修改时间、文件大小、编码和换行格式；如果文件在读取后被外部程序改动，写入会被拒绝并要求重新读取。写入使用同目录临时文件和原子替换，已有 UTF-8 BOM、LF、CRLF 或 CR 风格会被保留。

命令工具只接受 `argv` 数组，不调用二级 Shell。开发程序默认可运行；PowerShell、CMD、Bash 等二级 Shell 和高风险系统入口会被静态策略阻止，Git 在通用命令中仅开放只读子命令。命令工作目录和显式路径参数必须留在工作区内，敏感环境变量不会传入子进程，stdout 与 stderr 使用有界首尾保留。

服务器和其他持续运行的程序应设置 `run_in_background=true`。工具会立即返回任务 ID，后续使用 `process_status` 查看状态，使用 `stop_process` 结束；Agent 运行结束时仍存活的后台任务会被统一清理。前台命令超时后会提示改用后台模式。`python -c`、Node `-e/--eval` 可用于一次性检查，不再因形式本身被拒绝。

`search_code` 的 `path` 可以是单个文件或目录。`git_diff` 会先检查仓库状态；对于普通非 Git 目录，它返回“diff 不可用”而不是执行错误，Agent 不会再通过通用命令反复尝试 Git。

## 骨架验收

查看命令：

```powershell
py -3.11 -m uv run capycode --help
```

直接调试终端界面：

```powershell
py -3.11 -m uv run capycode
```

检查示例配置：

```powershell
py -3.11 -m uv run capycode doctor `
  --models config/models.example.yaml `
  --profiles config/profiles.example.yaml
```

执行质量检查：

```powershell
py -3.11 -m uv run ruff check src tests
py -3.11 -m uv run ruff format --check src tests
py -3.11 -m uv run mypy src
py -3.11 -m uv run pytest
py -3.11 -m uv build
```

`doctor` 只检查配置和环境，不会调用任何模型。缺少 API 环境变量时会显示 warning；使用 `--strict-secrets` 可将其视为失败。

## 模块边界

```text
src/capycode/
  app/          Textual TUI、CLI、事件到界面的映射
  core/         Agent Loop、SessionState、Context、Termination
  capability/   Capability、Profile、Router、Escalation
  llm/          OpenAI-compatible 适配、统一响应、Usage、Pricing
  tools/        模型可调用的动作接口
  workspace/    本地文件、进程、Git 与路径边界
  trace/        Event、StepTrace、JSONL 与运行产物
  profiling/    微基准、策略评测、指标与榜单
  config/       YAML Schema、加载与跨配置校验
```

依赖方向要求：`app → core → capability/llm/tools/trace`，业务核心不得反向依赖 Textual。

## 凭据约束

- 仓库配置只保存 `base_url_env` 和 `api_key_env` 的变量名称。
- API Key、真实 Base URL、`.env` 和原始运行轨迹不得提交。
- 实验报告可以记录端点类型、模型版本和价格快照，但必须脱敏。

## 协作规范

开发分支、模块验收、提交和 PR 规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。
