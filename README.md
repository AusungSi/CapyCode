# NJU-CodingAgent

一个从零实现的本地 TUI Coding Agent。项目首先建立稳定、可追踪的 Baseline Agent，随后通过 Capability Profiling，在同一个 Agent Loop 内进行 Step-level 模型路由。

## 当前阶段

当前开发阶段为 **P0-1 Runtime Skeleton**。P0-0 工程骨架已经完成；本阶段新增：

- 统一 LLM Request/Response/Tool Call 类型
- OpenAI-compatible Chat Completions 适配器
- `SessionState` 与最小 Agent Loop
- 受 Workspace 边界保护的 `read_file`
- FakeLLM 确定性双轮闭环
- Textual 全屏终端界面和 Slash Command
- 用户目录下的本地模型与凭据配置
- OpenAI-compatible SSE 流式输出和增量工具调用组装
- 用户、Agent、工具与系统状态分层消息视图

完整 Coding Tool Loop 和 Trace 将在后续独立模块中实现。

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

`capycode` 会直接进入终端交互界面。输入普通文本执行任务，输入 `/` 展开命令菜单。当前提供：

- `/help`：查看命令说明
- `/config`：填写 Base URL 和 API Key，自动获取模型列表后选择模型
- `/models`、`/model [model-id]`：查看真实模型列表并打开键盘选择器
- `/workspace [path]`：查看和切换工作区
- `/status`、`/clear`、`/quit`：会话控制

Slash Command 菜单支持方向键移动、Tab 补全和 Esc 关闭；普通输入支持方向键查找本次会话的历史任务。`/model` 打开独立模型选择面板，方向键选择、Enter 确认、Esc 取消。

模型请求采用真实 SSE 流式传输：首个 Token 返回前显示轻量思考动画，开始输出后原位更新同一个 Markdown 消息；工具调用显示进行中、成功或失败状态。主界面移除了占用空间的欢迎横幅和 Footer，将终端高度优先留给会话内容。

`/config` 会按照 OpenAI-compatible 协议请求 `<Base URL>/models`。CapyCode 会保存服务端返回的完整模型列表以及当前选择；界面和状态栏只显示真实模型 ID，不显示内部路由别名。直接执行 `/model` 可使用方向键选择模型，Enter 确认，Esc 取消。本地配置保存在 `~/.capycode/settings.json`，不会写入项目仓库；模型环境变量仍作为无本地配置时的兼容回退。项目名 **CapyCode** 来自 Capability + Code，并与源码包 `capycode`、运行产物目录 `.capy` 保持一致。

Linux/WSL：

```bash
uv sync --extra dev
```

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
