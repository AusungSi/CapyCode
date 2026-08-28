# NJU-CodingAgent

一个从零实现的本地 TUI Coding Agent。项目首先建立稳定、可追踪的 Baseline Agent，随后通过 Capability Profiling，在同一个 Agent Loop 内进行 Step-level 模型路由。

## 当前阶段

当前开发阶段为 **P0-0 Project Scaffold**，仅包含：

- Python 3.11 与 uv 工程配置
- `capycode` 模块边界
- 模型与 Profile YAML 配置校验
- 本地环境自检命令
- Windows/Linux CI
- 分支、提交与 PR 审计规范

Agent Loop、工具执行、Trace 和 Textual TUI 将在后续独立模块中实现。

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

项目名 **CapyCode** 来自 Capability + Code，并与源码包 `capycode`、运行产物目录 `.capy` 保持一致。P0-0 启动后展示品牌、工作区和可用自检命令；完整交互式 Agent 将在 Runtime 与 Textual TUI 模块接入同一入口。

Linux/WSL：

```bash
uv sync --extra dev
```

## 骨架验收

查看命令：

```powershell
py -3.11 -m uv run capycode --help
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
