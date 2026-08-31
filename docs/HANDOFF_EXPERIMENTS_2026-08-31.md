# CapyCode / SWE-bench 实验交接文档

更新时间：2026-08-31  
工作目录：D:\project\NJU-CodingAgent

## 1. 当前结论

项目已经支持按 capability 配置模型、工具、上下文预算和 reasoning effort。已有两条实验线：

1. 单模型 baseline：每个任务固定使用一个模型。
2. capability 路由：同一个 loop 内，不同 capability 可以切换推理强度；这不是把任务拆给不同模型，而是让理解、规划、编辑、验证等能力各自选择配置。

在当前短 P0 集合上，最好的已验证结果是：

- DeepSeek V4 Flash 单模型：5/5，成本 ¥0.1346658。
- 两模型 capability 路由（Qwen3.5 Flash 负责 understanding/editing，其余仍用 DeepSeek V4 Flash）：10/10，成本 ¥0.2178442。
- 相同模型的 capability 推理强度切换：5/5，成本 ¥0.1391942；相比 Flash 固定强度没有节省成本，但延迟更低。

SWE-bench Flask-4045 的完整官方 Docker 评测已经成功执行到真实 pytest，说明当前问题不是 Docker 基础设施失败；最后一个 Pro 低 verification 候选仍未通过测试，结果为 117 failed、51 passed、11 errors，不能算解决成功。

## 2. 代码结构与已完成改动

核心代码：

- src/capycode/llm.py：OpenAI-compatible 请求封装，支持 reasoning_effort。
- src/capycode/config/models.py：模型、profile、capability 配置结构。
- src/capycode/config/default_models.yaml：默认模型及 virtual model current。
- src/capycode/runtime.py：解析 profile、当前模型和 capability 配置。
- src/capycode/router.py：按 capability 路由，并限制 escalation 在同一 capability 内。
- src/capycode/quality.py：质量聚合按 run/profile/model/effort/capability 统计，避免把不同 step 混成一个样本。
- src/capycode/trace.py：记录模型、推理强度、token、缓存和工具结果。
- src/capycode/benchmarks/swebench_cli.py：SWE-bench CLI，支持 profiles 和 profiled artifact。
- src/capycode/benchmarks/swebench_runner.py：工作区准备、agent loop 和结果汇总。

配置示例：

- config/experiments/guochan_profiles_current_unrestricted_quality.yaml
- config/experiments/guochan_profiles_current_low_verification.yaml
- config/experiments/deepseek_v4_pro_adaptive_quality.yaml
- config/experiments/deepseek_v4_pro_quality.yaml

实验时的 prompt 约束是：只修改生产代码，先检查仓库约定，运行聚焦测试，保持最小 diff。候选模型通过 virtual model current 解析，方便同一份 profile 在不同端点配置下复用。

## 3. 已验证测试

定向 pytest 已通过：

- 最近一次相关定向测试：39 项通过。
- routing 相关测试此前累计：62 项通过。
- Ruff 检查通过。

建议交接后先重新执行：

~~~powershell
python -m pytest -q tests
ruff check src tests
~~~

## 4. 单模型 baseline

Artifact：

.capy/benchmarks/routing/five-model-baseline-v1/20260831T015203Z-6681dc4d/report.json

每个模型在 5 个 P0 任务上各跑一次：

| 模型 | 通过 | 成本 |
|---|---:|---:|
| deepseek-v4-flash-202605 | 5/5 | ¥0.1346658 |
| qwen3.5-flash | 4/5 | ¥0.225834 |
| GLM | 5/5 | ¥0.527472 |
| deepseek-v4-pro-202606 | 5/5 | ¥0.461485 |
| Kimi | 5/5 | ¥1.36058 |

Flash 是当前性价比最好的单模型 baseline。Qwen3.5 Flash 在这组任务中少通过 1 个任务且价格更高；Pro、GLM、Kimi 没有体现出足够的质量收益来抵消成本。

## 5. capability 模型路由结果

重复 2 次的 artifact：

.capy/benchmarks/routing/five-model-profiled-repeat2-v1/20260831T023934Z-3dadcd93/report.json

| 方案 | 通过 | 成本 | 延迟 | 输入 token | 缓存 token | 输出 token | 工具失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flash 固定 | 10/10 | ¥0.247170 | 149.502s | 217240 | 172800 | 10730 | 10/84 |
| Qwen understanding/editing + Flash 其余 | 10/10 | ¥0.2178442 | 135.009s | 201681 | 164992 | 10142 | 11/82 |

路由方案相对固定 Flash：成本下降 11.9%，延迟下降 9.7%，输入 token 下降 7.2%，输出 token 下降 5.5%；通过率不变，但工具失败率从 11.90% 略升到 13.41%。因此目前可称为成本和延迟更优的候选，不应宣称质量显著提升。

## 6. 同模型不同 reasoning effort

Flash 固定强度 baseline：

- 5/5，¥0.1346658，75.07s。

Selective effort：

- retrieval/understanding/verification 使用 low。
- planning/editing/diagnosis 使用 high。
- 5/5，¥0.1391942，66.79s。
- token：input 105430，cached 73472，output 3997。

Adaptive effort：

- 5/5，¥0.1444838，69.12s。
- token：input 104200，cached 73088，output 4871。

结论：在短 P0 上，推理强度切换降低了延迟，但没有降低成本；成本反而略高。可能原因包括任务短、模型价格和质量接近、缓存占比高，以及 high effort 的额外输出抵消了输入节省。需要在更长的 SWE-bench loop 或差异更大的模型上继续测。

## 7. SWE-bench Flask-4045

Manifest：

manifests/swebench_flask_4045_profiled.jsonl

实例：pallets__flask-4045  
base commit：d8c37f43724cd9fb0870f77877b7c4c7e38a19e0

### Flash reasoning v2

Artifact：

.capy/benchmarks/swebench-flash-reasoning-flask-4045-v2/20260831T031354Z-8f2d01b2

约 77 steps，成本 ¥0.916970；输入 1,504,231，缓存 1,318,400，输出 25,293。补丁可以生成，但官方测试暴露出 blueprint dotted name 通过而自定义 endpoint dotted name 失败，说明模型没有完整覆盖行为要求。

### Pro high verification

Artifact：

.capy/benchmarks/swebench-pro-unrestricted-quality-flask-4045-v3/20260831T034503Z-800a111a

运行到 51 steps 后因超过 15 次重复 verification 且没有新编辑而停止。trace：输入 444397，缓存 218496，输出 14866，成本 ¥2.50004，工具错误 6 次。最终 patch 主要只有 assert，未作为有效解通过官方评测。

### Pro low verification

Artifact：

.capy/benchmarks/swebench-pro-unrestricted-quality-flask-4045-v3/20260831T035109Z-5e4b9141

运行到 58 steps 后同样因重复 verification 停止。trace：输入 689085，缓存 441728，输出 16677，成本 ¥2.80901，工具错误 6 次。生成的核心代码是：

~~~python
if "." in name:
    raise ValueError("'name' may not contain a dot '.' character.")
~~~

官方候选：

.capy/official-eval/deepseek-v4-pro-202606.low-verification-flask-4045-stalled.predictions.jsonl

官方报告：

.capy/official-eval/deepseek-v4-pro-202606-low-verification-stalled.pro-low-verification-flask-4045-stalled-v3-lf.json

最后一次 Docker 官方评测已经进入真实 pytest，结果为 117 failed、51 passed、11 errors。这个候选是手动从 stalled run 导出的，不能当作完整自然收敛的 benchmark 结果。

## 8. 环境与已知问题

1. GitHub TLS clone 偶发失败。runner 的 repo-cache 已可用于预热，但不要把网络可用性当作 benchmark 成功标准。
2. 最高优先级代码问题：SWEbenchRunner._prepare_workspace 使用 git clone --shared。Windows 上生成的 .git/objects/info/alternates 包含 D:\ 路径，进入 Linux agent Docker 后该路径无效，导致 agent 内 git status/log 等工具失败。建议容器模式使用普通 clone，或写入容器可见的 Linux alternates 路径，并补回归测试。
3. Windows 官方 swebench harness 的 CRLF 问题已临时修复：

C:\Users\lyt\AppData\Roaming\Python\Python313\site-packages\swebench\harness\run_evaluation.py

修改为 eval_file.write_text(..., newline="\n")。已通过 py_compile，且后续 v3-lf 评测确实执行了 pytest。该修复在环境重装后可能丢失，最好在项目脚本或补丁说明中固化。
4. predictions JSONL 中 patch 字符串必须包含 JSON 解析后的真实换行 \n，不能是双重转义的字面量 \\n，否则 git apply 会失败。
5. 当前没有正在运行的 CapyCode benchmark 或 SWE-bench Docker 进程；交接时先检查，不要假设已有后台任务。

## 9. 建议的下一步

优先级从高到低：

1. 修复 clone --shared 的 Windows/Linux 容器路径问题，并增加测试。
2. 先用 Flash 固定和 Flash selective-effort 在同一个 SWE-bench 实例上各跑完整 loop，记录通过率、总 token、cached token、成本、延迟、工具失败和重复 verification 次数。
3. 再测试差异更大的组合，例如 Flash 负责 retrieval/verification，Pro 或 GLM 负责 planning/editing；每个组合至少重复 2 次。
4. 将“自然完成”和“因重复检测而停止”分开统计；stalled 结果只能作为诊断数据。
5. 官方 Docker 评测必须以真实 pytest 执行和 patch 应用成功为准，不能只看 runner 是否返回或 agent 是否生成 patch。

常用检查命令：

~~~powershell
Get-Process | Where-Object { $_.ProcessName -match 'python|docker|swebench|capycode' }
Get-ChildItem .capy\benchmarks -Directory
Get-ChildItem .capy\official-eval -File
~~~

## 10. 交接注意事项

- 保留当前工作树中的实验输出和未提交改动，不要使用 git reset --hard 或 git checkout -- 清理。
- 读取 report.json、trace.jsonl、summary.json 和官方 eval report 时，优先以 artifact 内的结构化字段为准。
- 价格不是唯一指标；最终比较至少包含 pass rate、输入/输出 token、cached token、成本、延迟、工具失败率、重复 loop 次数和稳定性。
- 当前最可信的结论是：Flash 是强单模型 baseline；短 P0 上两模型 capability 路由有约 11.9% 成本下降且通过率不变；同模型 effort 切换暂时只表现为延迟优化；SWE-bench 仍需先修复 workspace clone 问题，再做完整重复实验。
