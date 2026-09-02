"""TUI 可视化：自适应学习监控命令"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .integration import AdaptiveRuntime


def format_adaptive_status(adaptive_runtime: AdaptiveRuntime) -> str:
    """格式化自适应学习状态为 Markdown"""

    status = adaptive_runtime.get_status()

    if not status["enabled"]:
        return "## 自适应学习系统\n\n状态: **禁用**"

    lines = [
        "## 自适应学习系统",
        "",
        "**状态**: 已启用",
        f"**总任务数**: {status['total_tasks']}",
        f"**Policy 版本**: v{status['policy_version']}",
        f"**模型训练**: {'已训练' if status['model_fitted'] else '数据不足'}",
        "",
        "### 当前 Policy",
        "",
        "| Capability | Effort |",
        "|------------|--------|",
    ]

    for cap, effort in status["current_policy"].items():
        cap_display = cap.replace("_", " ").title()
        lines.append(f"| {cap_display} | **{effort}** |")

    lines.extend(
        [
            "",
            "### 探索统计",
            "",
            "| Capability | 探索次数 |",
            "|------------|----------|",
        ]
    )

    for cap, count in status["exploration_counts"].items():
        cap_display = cap.replace("_", " ").title()
        lines.append(f"| {cap_display} | {count} |")

    return "\n".join(lines)


def format_adaptive_history(adaptive_runtime: AdaptiveRuntime, limit: int = 10) -> str:
    """格式化最近的样本历史"""

    data = adaptive_runtime.historical_data
    samples = data.samples[-limit:] if len(data.samples) > limit else data.samples

    if not samples:
        return "## 样本历史\n\n暂无数据。"

    lines = [
        "## 最近的样本",
        "",
        f"显示最近 {len(samples)} 个样本:",
        "",
        "| Task ID | 结果 | 探索 | 总成本 |",
        "|---------|------|------|--------|",
    ]

    for sample in samples:
        result = "[OK]" if sample.outcome else "[FAIL]"
        explored = sample.explored_capability or "-"

        # 计算总成本
        total_cost = sum(stats.get("total_cost", 0.0) for stats in sample.capability_stats.values())

        lines.append(f"| {sample.task_id[:12]}... | {result} | {explored} | ${total_cost:.3f} |")

    return "\n".join(lines)


def format_adaptive_performance(adaptive_runtime: AdaptiveRuntime) -> str:
    """格式化性能统计"""

    data = adaptive_runtime.historical_data

    if len(data.samples) < 5:
        return "## 性能统计\n\n数据不足（需要至少 5 个样本）。"

    # 计算总体统计
    total_tasks = len(data.samples)
    success_count = sum(1 for s in data.samples if s.outcome)
    success_rate = success_count / total_tasks

    # 计算总成本
    total_cost = 0.0
    for sample in data.samples:
        for stats in sample.capability_stats.values():
            total_cost += stats.get("total_cost", 0.0)

    avg_cost = total_cost / total_tasks

    lines = [
        "## 性能统计",
        "",
        f"**总任务数**: {total_tasks}",
        f"**成功率**: {success_rate:.1%} ({success_count}/{total_tasks})",
        f"**平均成本**: ${avg_cost:.3f}",
        f"**总成本**: ${total_cost:.2f}",
        "",
        "### 按 Capability 统计",
        "",
        "| Capability | 平均成本 | Low | Medium | High |",
        "|------------|----------|-----|--------|------|",
    ]

    cost_estimator = adaptive_runtime.cost_estimator

    for cap in ["retrieval", "understanding", "planning", "editing", "diagnosis", "verification"]:
        cap_samples = data.get_samples_by_capability(cap)
        if not cap_samples:
            continue

        cap_display = cap.replace("_", " ").title()

        # 计算平均成本
        cap_costs = []
        for sample in cap_samples:
            if cap in sample.capability_stats:
                cap_costs.append(sample.capability_stats[cap].get("total_cost", 0.0))

        avg = sum(cap_costs) / len(cap_costs) if cap_costs else 0.0

        # 各 effort 的估计成本
        low_cost = cost_estimator.estimate_cost(cap, "low")
        med_cost = cost_estimator.estimate_cost(cap, "medium")
        high_cost = cost_estimator.estimate_cost(cap, "high")

        lines.append(
            f"| {cap_display} | ${avg:.3f} | ${low_cost:.2f} | ${med_cost:.2f} | ${high_cost:.2f} |"
        )

    return "\n".join(lines)
