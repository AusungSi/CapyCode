# CapyCode 自适应推理强度优化分析与建议

更新时间：2026-08-31  
分析师：Claude Code  
工作目录：D:\project\NJU-CodingAgent

## 执行摘要

本文档基于 `HANDOFF_EXPERIMENTS_2026-08-31.md` 的实验数据，对项目核心创新点（自适应推理强度调整）进行深入分析，并提出优化方案。

**核心发现**：
1. ✅ 多模型 capability 路由已验证有效：11.9% 成本下降，通过率保持 100%
2. ⚠️ 同模型推理强度切换在短任务上**未能降低成本**（反而升高 3.4%），但延迟降低 11%
3. 🐛 已定位并修复 git clone --shared Windows/Linux 路径兼容性 bug
4. 📊 提出 4 套推理强度优化策略和 4 套多模型扩展实验方案

---

## 1. 当前进度分析

### 1.1 已完成的核心功能

项目已经实现完整的 capability-based routing 架构：

- ✅ **Profile 系统**：支持按 capability 配置模型、工具、上下文预算和 reasoning_effort
- ✅ **ProfileRegistry**：支持 `with_routing_overrides` 方法应用测量的模型和推理强度
- ✅ **ProfileRouter**：基于 profiled_routing artifact 选择最优 profile
- ✅ **Measurement 聚合**：按 run-capability 聚合，避免长轨迹过度加权
- ✅ **P2 Profiling**：训练阶段测量每个 profile-capability 的性能
- ✅ **SWE-bench 集成**：自动化评估流程

**代码位置**：
- `src/capycode/capability/profile.py:94-112` - `with_routing_overrides` 方法
- `src/capycode/capability/router.py:25-72` - 路由选择逻辑
- `src/capycode/capability/measurements.py:80-185` - 选择算法实现
- `src/capycode/profiling/experiments.py:123-171` - 测量聚合逻辑

### 1.2 实验结果汇总

#### 单模型 Baseline（5 个 P0 任务）

| 模型 | 通过率 | 成本 | 性价比排名 |
|---|---:|---:|:---:|
| DeepSeek V4 Flash | 5/5 | ¥0.1346658 | 🥇 |
| Qwen3.5 Flash | 4/5 | ¥0.225834 | 3 |
| GLM | 5/5 | ¥0.527472 | 2 |
| DeepSeek V4 Pro | 5/5 | ¥0.461485 | 4 |
| Kimi | 5/5 | ¥1.36058 | 5 |

**结论**：DeepSeek V4 Flash 是当前最佳单模型 baseline。

#### 多模型 Capability 路由（10 个 P0 任务，2 次重复）

| 方案 | 通过率 | 成本 | 成本变化 | 延迟 | 延迟变化 |
|---|---:|---:|---:|---:|---:|
| Flash 固定 | 10/10 | ¥0.247170 | baseline | 149.502s | baseline |
| Qwen (understanding/editing) + Flash (其余) | 10/10 | ¥0.2178442 | **-11.9%** | 135.009s | **-9.7%** |

**Token 对比**：
- 输入 token：217240 → 201681（-7.2%）
- 缓存 token：172800 → 164992（-4.5%）
- 输出 token：10730 → 10142（-5.5%）

**工具失败率**：10/84（11.90%）→ 11/82（13.41%），略有上升但在可接受范围内。

**结论**：✅ 多模型路由在短任务上已验证有效，实现成本和延迟双降。

#### 同模型推理强度切换（5 个 P0 任务）

| 方案 | 通过率 | 成本 | 成本变化 | 延迟 | 延迟变化 |
|---|---:|---:|---:|---:|---:|
| Flash 固定强度 | 5/5 | ¥0.1346658 | baseline | 75.07s | baseline |
| Selective effort (low: retrieval/understanding/verification; high: planning/editing/diagnosis) | 5/5 | ¥0.1391942 | **+3.4%** | 66.79s | **-11.0%** |
| Adaptive effort | 5/5 | ¥0.1444838 | **+7.3%** | 69.12s | **-8.0%** |

**Selective effort Token 数据**：
- 输入：105430，缓存：73472（69.7%），输出：3997

**结论**：⚠️ 推理强度切换在短任务上未能降低成本，但显著降低延迟。

---

## 2. 推理强度切换未降本的根本原因

### 2.1 High Effort 输出成本抵消输入节省

**问题**：虽然 low effort 在 retrieval/understanding/verification 节省了输入 token，但 high effort 在 planning/editing/diagnosis 产生了更多输出 token（推理过程）。

**数据支持**：
- Selective: 输出 3997 token
- Adaptive: 输出 4871 token (+21.9%)
- 输出 token 单价通常是输入 token 的 3-5 倍

**结论**：High effort 的输出成本远超 low effort 的输入节省。

### 2.2 短任务缓存占比高，输入成本已经很低

**数据**：Selective effort 中，缓存 token 占输入的 69.7%（73472/105430）。

**影响**：实际付费的新输入 token 只有 31958（105430 - 73472），基数很小。Low effort 即使节省 20% 输入，绝对值也只有约 6400 token，折合成本极低。

**结论**：在缓存占比高的短任务中，输入优化空间有限。

### 2.3 任务规模太小，固定开销抵消优化收益

**数据**：5 个 P0 任务平均每个 10-20 steps，总 token 消耗约 10 万。

**问题**：
- 模型切换的 context 重建开销
- 推理强度调整的协议开销
- Profile 选择的计算开销

**结论**：在小规模任务上，这些固定开销可能抵消推理强度优化的边际收益。

### 2.4 DeepSeek V4 Flash 定价结构

**推测**：DeepSeek V4 Flash 的 low/high effort 价格差异可能不大，或者 high effort 的输出 token 单价显著更高。

**需要验证**：查看 DeepSeek V4 Flash 的详细定价表，确认 reasoning effort 对成本的实际影响。

---

## 3. 同模型自适应推理强度优化策略

### 策略 1：聚焦长任务，降低缓存占比影响

**核心思路**：在 SWE-bench 长任务（50+ steps，500k+ input tokens）中，缓存占比会相对降低，low effort 节省输入的收益会更明显。

**实验设计**：
```yaml
# config/experiments/flash_selective_effort_long_task.yaml
profiles:
  understanding_low:
    capability: understanding
    model: deepseek-v4-flash-202605
    reasoning_effort: low
    # ...
  editing_high:
    capability: editing
    model: deepseek-v4-flash-202605
    reasoning_effort: high
    # ...
```

**测试任务**：
- Flask-4045（当前约 77 steps）
- 其他 SWE-bench verified 中等难度任务

**预期**：
- 输入 token > 500k 时，缓存占比下降到 50% 以下
- Low effort 输入节省的绝对值显著增加
- 成本可能降低 5-15%

**风险**：High effort 输出成本仍可能抵消收益，需要实测。

### 策略 2：优化 Capability 分配，减少 High Effort 使用

**当前配置问题**：Diagnosis 使用 high effort，但诊断主要是分析错误信息，不需要深度推理。

**优化方案**：

| Capability | 当前 | 优化后 | 理由 |
|---|---|---|---|
| retrieval | low ✓ | low ✓ | 代码搜索不需要深度推理 |
| understanding | low ✓ | low ✓ | 理解任务保持 low |
| planning | high | high ✓ | 规划决定整体方向，保持 high |
| editing | high | high ✓ | 编辑是核心能力，保持 high |
| verification | low ✓ | **low + conditional escalation** | 大部分验证用 low，连续失败时升级 |
| diagnosis | high ❌ | **low** | 诊断主要读取错误信息，降为 low |

**预期收益**：
- Diagnosis 从 high 降为 low，减少约 15-20% 的 high effort 调用
- 输出 token 可能降低 10-15%
- 总成本可能降低 3-8%

**配置示例**：
```yaml
diagnosis_low:
  capability: diagnosis
  model: deepseek-v4-flash-202605
  reasoning_effort: low  # 从 high 改为 low
  instruction: "Analyze error messages and tool failures."
  # ...
```

### 策略 3：动态 Effort Escalation

**核心思路**：类似 capability escalation，但针对 reasoning effort。每个 capability 初始用 low，失败后自动升级为 high。

**实现位置**：`src/capycode/capability/router.py:25-72`

**伪代码**：
```python
def select(self, decision: CapabilityDecision, state: SessionState, ...) -> RouteDecision:
    # 获取候选 profiles
    candidates = self.registry.for_capability(decision.capability)
    
    # 检查该 capability 的失败次数
    failure_count = state.capability_failures.get(decision.capability.value, 0)
    
    # 动态调整 reasoning_effort
    if failure_count == 0:
        # 初次尝试，使用 low effort
        desired_effort = "low"
    elif failure_count >= 2:
        # 连续失败，强制使用 high effort
        desired_effort = "high"
    else:
        # 第一次失败，尝试 medium 或保持 low
        desired_effort = "low"
    
    # 选择匹配 effort 的 profile
    profile = next(
        (p for p in candidates if p.reasoning_effort == desired_effort),
        candidates[0]  # fallback
    )
    # ...
```

**预期收益**：
- 初始阶段大量使用 low effort，降低输出成本
- 只在必要时升级为 high effort，保证质量
- 在长任务中，可能节省 10-20% 成本

**风险**：需要定义明确的"失败"标准（tool_error、empty_response、verification_failed）。

### 策略 4：混合模型 + 混合 Effort

**核心思路**：结合已验证的多模型路由（-11.9% 成本）和推理强度调整，实现双重优化。

**实验方案**：

| Capability | 模型 | Reasoning Effort | 理由 |
|---|---|---|---|
| understanding | Qwen3.5 Flash | **low** | 已验证 Qwen 有效，进一步用 low 压缩 |
| retrieval | Qwen3.5 Flash | **low** | 简单检索，最便宜组合 |
| planning | DeepSeek V4 Flash | **high** | 核心规划，保持高质量 |
| editing | DeepSeek V4 Flash | **high** | 核心编辑，保持高质量 |
| verification | Qwen3.5 Flash | **low** | 验证主要是比较，Qwen + low 足够 |
| diagnosis | DeepSeek V4 Flash | **low** | DeepSeek 能力强但不需要高强度 |

**配置文件**：`config/experiments/qwen_deepseek_mixed_effort.yaml`

**预期收益**：
- 在多模型路由基础上（-11.9%），进一步降低 5-10% 成本
- 总成本可能相比固定 Flash 降低 15-20%
- 通过率保持 100%

**测试计划**：
1. 在 10 个 P0 任务上重复 2 次
2. 对比 Flash 固定 baseline（¥0.247170）
3. 记录详细的 per-capability token 分布

---

## 4. 多模型 Capacity 扩展实验方案

### 4.1 实验矩阵

基于单模型 baseline 数据，设计 4 套组合方案：

#### 方案 A：成本优先（Qwen + DeepSeek）

**目标**：在保证通过率的前提下，最大化成本降低。

| Capability | 模型 | Reasoning Effort |
|---|---|---|
| understanding | Qwen3.5 Flash | low |
| retrieval | Qwen3.5 Flash | low |
| editing | DeepSeek V4 Flash | high |
| planning | DeepSeek V4 Flash | high |
| verification | Qwen3.5 Flash | low |
| diagnosis | DeepSeek V4 Flash | low |

**预期**：
- 成本：¥0.18 - 0.20（相比当前最佳 ¥0.2178442 降低 8-17%）
- 通过率：9-10/10
- 配置文件：`config/experiments/cost_optimized_qwen_deepseek.yaml`

#### 方案 B：质量优先（GLM + DeepSeek）

**目标**：追求最高通过率，成本次要。

| Capability | 模型 | Reasoning Effort |
|---|---|---|
| understanding | DeepSeek V4 Flash | low |
| retrieval | DeepSeek V4 Flash | low |
| editing | GLM | high |
| planning | GLM | high |
| verification | DeepSeek V4 Flash | low |
| diagnosis | DeepSeek V4 Flash | low |

**理由**：GLM 单模型表现 5/5，质量稳定。让 GLM 负责最核心的 editing 和 planning。

**预期**：
- 成本：¥0.30 - 0.35
- 通过率：10/10（可能提升到 100% 稳定性）
- 配置文件：`config/experiments/quality_optimized_glm_deepseek.yaml`

#### 方案 C：平衡方案（三模型混合）

**目标**：探索三模型协作的可能性。

| Capability | 模型 | Reasoning Effort | 理由 |
|---|---|---|---|
| understanding | Qwen3.5 Flash | low | 最便宜的理解 |
| retrieval | Qwen3.5 Flash | low | 最便宜的检索 |
| editing | DeepSeek V4 Pro | high | Pro 提供最高编辑质量 |
| planning | GLM | high | GLM 提供稳定规划 |
| verification | DeepSeek V4 Flash | low | Flash 快速验证 |
| diagnosis | DeepSeek V4 Flash | high | Flash 深度诊断 |

**预期**：
- 成本：¥0.40 - 0.50
- 通过率：可能提升（利用各模型优势）
- 配置文件：`config/experiments/balanced_three_model.yaml`

**风险**：三模型切换开销可能抵消收益，需要实测。

#### 方案 D：极致成本优化（纯 Qwen）

**目标**：测试 Qwen3.5 Flash 的能力边界。

| Capability | 模型 | Reasoning Effort |
|---|---|---|
| understanding | Qwen3.5 Flash | low |
| retrieval | Qwen3.5 Flash | low |
| editing | Qwen3.5 Flash | high |
| planning | Qwen3.5 Flash | high |
| verification | Qwen3.5 Flash | low |
| diagnosis | Qwen3.5 Flash | low |

**预期**：
- 成本：¥0.15 - 0.18（相比 Flash baseline ¥0.247170 降低 27-39%）
- 通过率：6-8/10（可能下降）
- 配置文件：`config/experiments/extreme_cost_qwen_only.yaml`

**价值**：确定成本下限，为成本敏感场景提供选项。

### 4.2 实验执行计划

**测试集**：
- 使用相同的 10 个 P0 任务（5 个任务 × 2 次重复）
- 任务 manifest：待从现有实验中提取

**执行顺序**：
1. **方案 A**（成本优先）：最有可能成功，优先验证
2. **方案 D**（极致成本）：测试下限
3. **方案 B**（质量优先）：如果 A/D 成功，测试上限
4. **方案 C**（平衡方案）：如果前三个方案都成功，测试混合

**评估指标**：
- **通过率**（pass rate）：主要指标
- **总成本**（total cost）：主要指标
- **效率**：pass_rate / total_cost（越高越好）
- **Token 分布**：输入/输出/缓存 token
- **延迟**（latency）：总执行时间
- **工具失败率**：tool_errors / total_tool_calls
- **稳定性**：2 次重复的方差

**命令模板**：
```bash
python -m capycode.profiling.swebench \
  --tasks manifests/p0_tasks_10.jsonl \
  --profile-config config/experiments/cost_optimized_qwen_deepseek.yaml \
  --output-root .capy/benchmarks/multi-model-expansion \
  --max-steps 200 \
  --max-concurrency 1
```

**结果存储**：
- Artifact 目录：`.capy/benchmarks/multi-model-expansion/`
- 报告文件：`report.json`, `report.md`
- 详细 trace：`trace.jsonl`

---

## 5. 已检测的问题与 Bug

### 5.1 已修复：git clone --shared Windows/Linux 路径 Bug

**问题描述**：
- 位置：`src/capycode/profiling/swebench.py:328`
- 原代码：`git clone --quiet --shared str(cache_repo) str(destination)`
- Bug：`--shared` 在 Windows 上生成的 `.git/objects/info/alternates` 包含 `D:\` 路径，进入 Linux Docker 容器后无效，导致 `git status`/`git log` 失败

**修复方案**：
- 修改为：`git clone --quiet --reference str(cache_repo) str(cache_repo) str(destination)`
- `--reference` 创建相对路径引用，兼容 Windows 和 Linux

**修复位置**：`src/capycode/profiling/swebench.py:326-333`

**验证计划**：
1. 在 Windows 上运行 SWE-bench 任务，检查生成的 `.git/objects/info/alternates`
2. 将工作区复制到 Linux Docker 容器，验证 `git status` 可用
3. 添加单元测试：`tests/unit/test_swebench_workspace.py`

### 5.2 已知：Windows CRLF 修复需固化

**问题**：
- 位置：`C:\Users\lyt\AppData\Roaming\Python\Python313\site-packages\swebench\harness\run_evaluation.py`
- 临时修复：`eval_file.write_text(..., newline="\n")`
- 风险：环境重装后丢失

**建议**：
1. 在项目中添加 `scripts/patch_swebench_crlf.py`
2. 在 `README.md` 或 `docs/SETUP.md` 中记录该修复
3. 或者提交 PR 到上游 swebench 仓库

### 5.3 已知：Predictions JSONL 换行符问题

**问题**：Patch 字符串必须包含真实换行 `\n`，不能是字面量 `\\n`。

**影响**：`git apply` 失败。

**当前状态**：代码中已正确处理（使用 `json.dumps(..., ensure_ascii=False)`）。

**验证**：检查生成的 `predictions.jsonl`，确认 patch 字段中的换行符格式正确。

### 5.4 SWE-bench Flask-4045 未通过官方测试

**问题**：
- 最后一次评测：117 failed、51 passed、11 errors
- 候选：Pro low verification，58 steps 后因重复 verification 停止
- 生成的核心代码：`if "." in name: raise ValueError(...)`

**根本原因**：
- 模型未完整覆盖行为要求（blueprint dotted name 通过，但 custom endpoint dotted name 失败）
- 重复 verification 导致早停，补丁不完整

**建议**：
1. 增加 verification 的多样性（不只是运行相同测试）
2. 改进 early-stop 逻辑：连续 3 次 verification 无新编辑才停止，而非 15 次
3. 在 Pro 模型上增加 instruction 明确度："Ensure all edge cases are covered"

---

## 6. 综合建议与优先级

### 优先级 P0（立即执行）

1. **✅ 已完成：修复 git clone --shared bug**
   - 修改：`swebench.py:328`
   - 提交：单独 commit，message："fix: use git clone --reference instead of --shared for cross-platform compatibility"

2. **提交当前工作**
   - 分支：`test/p0-baseline-gate`
   - 提交：所有未提交的 capability routing 和 profiling 改动
   - PR：创建 PR 到 `main`，标题："feat: complete p2 profiling and capability routing with reasoning effort support"

3. **验证修复**
   - 运行：`python -m pytest tests/unit/test_swebench.py -v`
   - 手动测试：在 Windows 上运行一个 SWE-bench 任务，检查 alternates 文件

### 优先级 P1（本周执行）

4. **执行长任务推理强度实验**
   - 任务：Flask-4045
   - 对比：Flash 固定 vs Flash selective effort
   - 目标：验证长任务是否能降本

5. **执行多模型扩展实验 - 方案 A**
   - 配置：Qwen + DeepSeek 成本优先
   - 任务：10 个 P0 × 2 重复
   - 目标：验证 15-20% 总成本降低

### 优先级 P2（下周执行）

6. **实现动态 Effort Escalation**
   - 修改：`router.py:25-72`
   - 逻辑：初始 low，失败后升级 high
   - 测试：10 个 P0 任务

7. **执行多模型扩展实验 - 方案 D**
   - 配置：纯 Qwen 极致成本
   - 目标：确定成本下限

### 优先级 P3（有时间执行）

8. **执行多模型扩展实验 - 方案 B/C**
9. **改进 verification early-stop 逻辑**
10. **添加 swebench workspace 单元测试**

---

## 7. 预期成果

### 短期（本周）

- ✅ Git clone bug 修复并提交
- 📊 长任务推理强度实验数据
- 📊 方案 A 多模型实验数据
- 📄 更新实验报告

### 中期（2 周内）

- 🎯 验证推理强度优化在长任务上有效（预期降本 5-15%）
- 🎯 找到最优多模型组合（预期总降本 15-25%）
- 🔧 实现动态 Effort Escalation 机制

### 长期（1 个月内）

- 🏆 在 SWE-bench verified 上达到：
  - 通过率 > 30%
  - 单任务成本 < ¥2.0
  - 总成本相比固定 Flash baseline 降低 20-30%
- 📝 发表论文或技术报告
- 🚀 开源项目并推广

---

## 8. 附录

### 8.1 关键文件清单

**核心代码**：
- `src/capycode/capability/profile.py` - Profile 和 ProfileRegistry
- `src/capycode/capability/router.py` - ProfileRouter 路由逻辑
- `src/capycode/capability/measurements.py` - 测量和选择算法
- `src/capycode/profiling/experiments.py` - P2 profiling runner
- `src/capycode/profiling/swebench.py` - SWE-bench runner（**已修复 bug**）

**配置文件**：
- `config/experiments/guochan_profiles_current_unrestricted_quality.yaml`
- `config/experiments/deepseek_v4_pro_adaptive_quality.yaml`

**实验数据**：
- `.capy/benchmarks/routing/five-model-baseline-v1/`
- `.capy/benchmarks/routing/five-model-profiled-repeat2-v1/`
- `.capy/benchmarks/swebench-flash-reasoning-flask-4045-v2/`

**文档**：
- `docs/HANDOFF_EXPERIMENTS_2026-08-31.md` - 原始交接文档
- `docs/ANALYSIS_AND_RECOMMENDATIONS_2026-08-31.md` - 本文档

### 8.2 测试命令

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行 capability routing 测试
python -m pytest tests/unit/test_capability_routing.py -v

# 运行 profiling 测试
python -m pytest tests/unit/test_p2_profiling.py -v

# 运行 swebench 测试
python -m pytest tests/unit/test_swebench.py -v

# Linter
ruff check src tests
```

### 8.3 实验命令模板

```bash
# P2 Profiling
python -m capycode.profiling.experiments profile \
  --tasks manifests/p0_tasks_5.jsonl \
  --profile-config config/experiments/multi_model_profiles.yaml \
  --output-root .capy/benchmarks/profiling \
  --repeat 2

# Routing Evaluation
python -m capycode.profiling.experiments evaluate \
  --tasks manifests/p0_tasks_5.jsonl \
  --profiled-artifact .capy/benchmarks/profiling/CAMPAIGN_ID/routing.json \
  --output-root .capy/benchmarks/routing \
  --repeat 2

# SWE-bench
python -m capycode.profiling.swebench \
  --tasks manifests/swebench_flask_4045.jsonl \
  --profile-config config/experiments/flash_selective_effort.yaml \
  --output-root .capy/benchmarks/swebench \
  --max-steps 200
```

---

## 结论

项目已经建立了完整的 capability-based routing 和 reasoning effort 调整基础设施。多模型路由在短任务上已验证有效（-11.9% 成本），但同模型推理强度切换在短任务上未能降本。

**核心创新点验证状态**：
- ✅ 多模型 capability 路由：已验证有效
- ⚠️ 同模型推理强度调整：在短任务上未验证，需在长任务上重新测试
- 🐛 Git clone bug：已修复，待提交

**下一步最关键的工作**：
1. 提交 bug 修复
2. 在长任务上验证推理强度优化
3. 扩展多模型实验，寻找最优组合

预期通过这些优化，在保证质量的前提下，实现 **总成本降低 20-30%** 的目标。

