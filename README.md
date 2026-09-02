# NJU-CodingAgent

一个从零实现的本地 TUI Coding Agent。项目首先建立稳定、可追踪的 Baseline Agent，随后通过 Capability Profiling，在同一个 Agent Loop 内进行 Step-level 模型路由。

## 当前阶段

当前开发阶段为 **P3 自适应推理策略**。P0-0 至 P0-4 的运行时、工具、可观测性和 Textual TUI 已完成；P1/P2 已实现 Capability Detection、Profile Routing、失败升级与路由评测，P3 新增了可持久化的探索、学习和策略优化运行时：

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
- 基于历史任务结果的 Capability-level effort 探索、成本估计与策略版本管理

Capability 路由使用可解释的本地状态信号；Profile 配置会解析为真实模型 ID，实际路由模型和原因会写入 Step Trace。P2 从冻结基准的真实 Trace 提取 `Profile × Capability × Model` 的成功、成本和延迟统计，只有达到样本量和可靠性阈值的候选才会接管路由；否则保留确定性回退。

P3 自适应运行时会记录每次任务的起始 effort、结果、Capability 成本和主动探索信息。系统先通过 warm-up 与 UCB 补足观测，再用反事实成功率估计，在可靠性容差内选择成本最低的 effort，并将每次更新保存为可回溯的 Policy 版本。没有安装机器学习扩展时，数据收集和探索仍可使用，策略优化会安全保持当前版本。

## 本地安装

要求 Python 3.11+。使用 uv 安装开发依赖：

```powershell
py -3.11 -m uv sync --extra dev
```

需要启用 P3 策略优化模型时，同时安装自适应扩展：

```powershell
py -3.11 -m uv sync --extra dev --extra adaptive
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
- `/config`：填写端点 ID、Base URL 和 API Key，自动获取模型列表后选择模型
- `/endpoints`、`/endpoint [端点 ID]`：查看和切换已保存的 Base URL/API Key 组合
- `/models`、`/model [model-id]`：查看真实模型列表并打开键盘选择器
- `/profiles`：查看当前生效的能力 Profile
- `/pricing`：为当前端点中的真实模型配置普通输入、缓存命中输入、输出价格、币种和上下文窗口
- `/workspace [path]`：查看和切换工作区
- `/resume [会话 ID]`、`/continue`：选择历史会话或继续最近会话
- `/sessions`、`/new`：列出历史会话或开始新会话
- `/runs`：用键盘选择运行记录，并查看步骤、工具、Token、费用和最终 diff
- `/benchmark [p0|swebench]`：在当前真实模型上运行 P0 或 SWE-bench 基准
- `/status`、`/clear`、`/quit`：会话控制

Slash Command 菜单支持方向键移动、Tab 补全和 Esc 关闭；普通输入支持方向键查找本次会话的历史任务。`/model` 打开独立模型选择面板，方向键选择、Enter 确认、Esc 取消。

在 TUI 中输入 `/benchmark swebench` 会打开配置窗口，填写 SWE-bench JSONL 实例清单、每个实例的最大 Agent 步数和并发实例数即可开始；默认是 200 步、并发 2，也可以直接输入 `/benchmark swebench <实例清单路径>` 使用默认值。运行使用当前选中的端点和真实模型 ID，完成后会显示预测文件、报告、Token、费用和延迟。每个实例使用独立工作区，可安全并发运行。

模型请求采用真实 SSE 流式传输：首个 Token 返回前显示轻量思考动画，开始输出后原位更新同一个 Markdown 消息；工具调用显示进行中、成功或失败状态，并可用鼠标、Enter 或空格展开参数、耗时、退出码和有界输出。启动时显示短暂的 CapyCode 封面，进入会话后使用随终端宽度折叠的状态栏，实时展示 Run、Token、费用和耗时。Ctrl+C 会先取消当前任务、补全 Trace 并恢复输入焦点，再次 Ctrl+C 可退出。

`/config` 会按照 OpenAI-compatible 协议请求 `<Base URL>/models`。CapyCode 直接使用并保存服务端返回的真实模型 ID，不再建立 `small`、`medium`、`strong` 等人工分级。直接执行 `/model` 可使用方向键切换真实模型；`/pricing` 为当前端点中的当前模型独立保存普通输入、缓存命中输入和输出的每百万 Token 价格、币种、价格日期和上下文窗口，端点之间互不影响。未填写缓存价时，缓存 Token 按普通输入价计算。本地配置保存在 `~/.capycode/settings.json`，不会写入项目仓库；旧版别名配置在首次读取时自动迁移。项目名 **CapyCode** 来自 Capability + Code，并与源码包 `capycode`、运行产物目录 `.capy` 保持一致。

部分高校或网关只提供聊天接口，不提供 `GET /models`。例如 NUAA 端点的聊天路由是 `https://token.nuaa.edu.cn/v1/chat/completions`，当前探测结果显示该路由存在，而 `https://token.nuaa.edu.cn/v1/models` 返回 `404 Route Not Found`。在这种情况下，在 `/config` 中填入端点 ID（如 `nuaa`）、Base URL `https://token.nuaa.edu.cn/v1` 和 API Key，然后在“模型 ID”输入框手动填写该平台分配的真实模型名，再保存即可。CapyCode 会按以下方式发送请求：

```text
POST https://token.nuaa.edu.cn/v1/chat/completions
Authorization: Bearer <API-Key>
Content-Type: application/json
```

模型 ID 必须以 NUAA 平台提供的名称为准，不能根据 URL 猜测。保存后用 `/endpoint nuaa` 切换，用普通任务或 `/benchmark` 验证；如果聊天请求返回 `401`，检查 API Key，如果返回 `404`，检查 Base URL 是否多填或少填了路径。

### 官方价格快照

仓库示例配置中的价格采用 [OpenAI 官方 API Pricing](https://platform.openai.com/docs/pricing) 页面列出的 Standard 价格，单位为 USD / 1M tokens，快照日期为 2026-08-29：

| 模型 | 普通输入 | 缓存输入 | 输出 |
| --- | ---: | ---: | ---: |
| `gpt-5.4` | $2.50 | 按官方页面配置 | $15.00 |
| `gpt-5.5` | $5.00 | 按官方页面配置 | $30.00 |
| `gpt-5.6-terra` | $2.00 | 按官方页面配置 | $12.00 |
| `gpt-5.6-sol` | $4.00 | 按官方页面配置 | $20.00 |

成本追踪器将每次请求的输入 Token 拆成“缓存命中”与“未命中”两部分：`未命中 × 普通输入价 + 命中 × 缓存输入价 + 输出 × 输出价`。`gpt-5.4`、`gpt-5.5`、`gpt-5.6-terra` 和 `gpt-5.6-sol` 是可直接按官方页面计价的公开 SKU；代理端返回的 `codex-auto-review` 没有对应的公开官方 SKU，应通过 `/pricing` 单独填写实际价格，不能套用以上数值。

对于通过学校或第三方聚合端点使用的国产模型，`/pricing` 保存的是模型提供商的官方价格估算，不代表聚合端点的实际扣费。已按 2026-08-29 的官方页面为 `guochan` 端点保存下列模型：

| 模型 ID | 计费价（普通输入 / 缓存输入 / 输出，每 1M Token） | 计价假设 |
| --- | ---: | --- |
| `deepseek-v4-flash-202605` | CNY 3 / 0.10 / 9 | DeepSeek V4 Flash 高峰时段 |
| `deepseek-v4-pro-202606` | CNY 9 / 0.30 / 27 | DeepSeek V4 Pro 高峰时段 |
| `kimi-k3` | CNY 20 / 2 / 100 | Kimi K3 标准价格 |
| `qwen3.5-flash` | CNY 1.2 / 0.12 / 12 | Qwen3.5-Flash 256K-1M 输入档位 |
| `glm-5.3` | CNY 8 / 2 / 28 | 智谱 GLM-5.3 标准价格 |

来源为 [DeepSeek 官方定价](https://api-docs.deepseek.com/quick_start/pricing)、[Kimi K3 官方定价](https://platform.kimi.com/docs/pricing/chat-k3)、[阿里云百炼官方定价](https://help.aliyun.com/zh/model-studio/model-pricing) 和 [智谱官方定价](https://bigmodel.cn/pricing)。DeepSeek 按高峰时段配置，Qwen 按最高输入长度档位配置；程序不会按时段或上下文长度自动切换价格。当前 `guochan` 端点的实际 OpenAI-compatible 响应已包含 `usage.prompt_tokens_details.cached_tokens`，因此 CapyCode 可以按真实命中量分开记账；学校或第三方聚合端点的实际扣费仍应以该端点账单为准。

### P1 能力 Profile

可选地将 `config/models.example.yaml` 和 `config/profiles.example.yaml` 复制为同目录的 `models.yaml`、`profiles.yaml`。Profile 按任务能力选择模型、工具白名单、上下文策略、单步输出预算和独立步骤上限；模型引用会解析为真实模型 ID。`instruction` 可以是内联系统指令，也可以是提示词文件路径；相对路径会按 Profile 配置目录和项目根目录解析。工具或模型请求连续失败时，运行时会重试或升级到备用 Profile，并把实际模型和路由原因写入 Trace。

### P2 画像与策略评测

先在同一 OpenAI-compatible 端点中配置需要比较的真实模型，并为每个模型设置价格。使用冻结 P0 任务采样，每个实际模型都经过完全相同的工作区、公开/隐藏测试和 Trace 校验：

```powershell
capycode profile p0 `
  --model <真实模型-ID-A> `
  --model <真实模型-ID-B> `
  --task p0-01 --task p0-02 --task p0-03 `
  --repeats 2 `
  --install
```

命令生成 `.capy/profiling/<campaign-id>/profiles.json`、`measurements.jsonl`、`leaderboard.csv`、`leaderboard.md`、`report.json` 和 `report.md`。`manifest.json` 固化模型、任务指纹和重复次数，`progress.json` 在每个模型完成后原子更新，同时保存 `profiles.partial.json`，因此中断时仍能确认已完成范围并审计已有测量。`--install` 会把经过选择的 `profiles.json` 安装为当前工作区的 `.capy/profiles.json`。之后普通 `capycode` 任务会自动加载该文件。选择顺序是：满足最小样本量和可靠性阈值后，最小化单步期望成功成本；若仍相同，再比较延迟。Leaderboard 同时展示样本数、成功率、平均成本、Expected Cost per Success 和 Efficiency，并明确将步骤标记为最终任务结果，因此它是可审计的路由启发式，而不是因果结论。

使用同一 P0 runner 对固定模型和已画像的策略进行 Holdout 比较：

```powershell
capycode evaluate p0 `
  --fixed-model <真实模型-ID-A> `
  --fixed-model <真实模型-ID-B> `
  --profiled-artifact .capy/profiling/<campaign-id>/profiles.json `
  --task p0-04 --task p0-05 `
  --repeats 2
```

评测生成 `manifest.json`、`progress.json`、`report.partial.json`、`comparison.csv`、`report.json` 和 `report.md`，统一报告通过率、Pass@1、总成本、每个成功任务的成本、延迟、平均步骤、普通/缓存/输出 Token、工具失败和基础设施错误。每个策略的完整 P0 报告保存在 `strategies/<strategy-id>/` 下。画像文件会记录训练任务指纹，评测默认同时拒绝任务 ID 重叠和同内容改名后的指纹重叠；仅调试时使用 `--allow-overlap`。策略集合由命令参数提供，P2 工程层不固化具体实验分组、统计方法或消融设计。未配置价格时成本会是零，不能据此作成本优劣结论。

同一终端中的后续任务会沿用当前模型上下文。会话在每次用户消息、模型响应和工具结果后增量保存到 `~/.capycode/sessions/`；关闭终端后，在同一代码目录再次启动 `capycode`，输入 `/resume` 可通过键盘选择历史会话，输入 `/continue` 可直接继续最近会话。默认启动新会话，不会在未确认时自动加载历史内容；`/new` 会开始新会话，但不会删除历史记录。恢复范围严格限制为当前工作区，API Key 不进入会话文件。中断产生的未配对工具消息会在恢复时清理，文件会要求重新读取，旧终端中的后台进程不会恢复。

每次用户任务都会创建新的 Run，并在 `<workspace>/.capy/runs/<run-id>/` 下生成追加式 `trace.jsonl` 和原子写入的 `summary.json`。恢复会话后继续沿用原 `session_id`，但使用新的 `run_id`。Trace 记录事件顺序、模型与工具延迟、Token、按配置估算的费用、测试状态、修改文件和终止原因；API Key、Authorization、Bearer Token 等敏感值在写入前统一脱敏。异常和取消路径同样生成 Summary，已发出的工具请求会补齐结构化失败结果。

## P0 Baseline Gate

P0 Gate 内置 5 个相互隔离的本地缺陷项目，覆盖边界条件、数据解析、多文件契约、状态隔离和文件错误处理。每个任务都带有公开失败测试、Agent 不可见的隐藏验收测试、允许修改文件清单和 SHA-256 内容指纹。

先验证任务集完整性；该命令不会读取模型配置或发起网络请求：

```powershell
capycode benchmark p0 --validate-only
```

使用当前本地配置的模型执行完整 Gate：

```powershell
capycode benchmark p0 --model <model-id> --repeats 2
```

也可以先执行单项冒烟测试；单项结果可以成功，但不足 5 项时不会把整个 Gate 标记为通过：

```powershell
capycode benchmark p0 --model <model-id> --repeats 1 --task p0-01
```

每次运行都使用新的 Git 工作区和 Session。最终判定要求公开测试、隐藏测试、修改范围和 Trace 工具配对同时通过。完整 Gate 要求 5 个任务均至少成功一次、10 次独立运行总体成功率不低于 80%，且不能出现基础设施错误。运行工作区、Trace、增量结果以及 `report.json`、`report.md` 保存在 `.capy/benchmarks/p0/<campaign-id>/`，该目录默认不进入 Git。

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

## 多端点与 SWE-bench

`/config` 中填写端点 ID 后保存即可保留多个 URL/API Key 组合；使用 `/endpoints` 查看，使用 `/endpoint <端点 ID>` 切换。命令行也支持 `capycode endpoints`、`capycode endpoint select <端点 ID>`，以及在 `run`、`benchmark p0`、`tui` 中传入 `--endpoint`。

SWE-bench 运行需要 JSONL 实例清单。每行至少包含 `instance_id`、`problem_statement` 和本地 `workspace`；也可以使用 `repo` 与 `base_commit`，由运行器克隆 GitHub 仓库并检出基线：

```powershell
capycode benchmark swebench --instances swebench.jsonl --endpoint openai --model <真实模型-ID>
```

运行器默认每个实例最多执行 200 步、命令行并发数为 1；可通过 `--max-concurrency 2` 等参数并发处理多个实例。仓库首次下载到 `.capy/benchmarks/swebench/repo-cache/`，后续实例从该只读缓存创建本地共享克隆，不会重复通过网络下载；每个实例仍使用独立工作区，并在 `.capy/benchmarks/swebench/<campaign-id>/` 生成隔离工作区、`predictions.jsonl`、`report.json` 和 `report.md`。工作区默认保留，确认已保存预测文件后可手动删除 campaign 目录。`predictions.jsonl` 交给官方 SWE-bench Docker harness 做隐藏测试和 resolved/unresolved 评测，本地报告额外记录普通输入、缓存输入、输出 Token、费用和延迟。
### 受控容器环境

SWE-bench 的 Agent 命令、测试和构建默认在固定 Docker 镜像中运行，宿主机只提供实例工作区挂载。首次使用前构建镜像：

```powershell
docker build -t capycode/swebench-python:3.11 -f docker/swebench-python-3.11.Dockerfile .
```

命令行可用 `--container-image` 指定已审核的替代镜像；TUI 的 `/benchmark swebench` 自动使用默认镜像。Docker 不可用或镜像不存在时，任务会明确报告基础设施错误，而不会回退到宿主机环境。
