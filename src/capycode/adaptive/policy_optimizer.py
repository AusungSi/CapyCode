"""策略优化器：Reliability-Constrained Cost Minimization"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .classifier import SuccessClassifier
    from .cost_estimator import CostEstimator
    from .models import HistoricalData

from .models import AdaptivePolicy


class PolicyOptimizer:
    """策略优化器"""

    CAPABILITIES: ClassVar[tuple[str, ...]] = (
        "retrieval",
        "understanding",
        "planning",
        "editing",
        "diagnosis",
        "verification",
    )

    def __init__(self, performance_tolerance: float = 0.05) -> None:
        self.performance_tolerance = performance_tolerance

    def optimize(
        self,
        current_policy: AdaptivePolicy,
        historical_data: HistoricalData,
        success_model: SuccessClassifier,
        cost_estimator: CostEstimator,
    ) -> AdaptivePolicy:
        """优化策略：Reliability-Constrained Cost Minimization"""

        new_effort: dict[str, str] = {}

        for capability in self.CAPABILITIES:
            # 1. Counterfactual Evaluation：评估所有 effort 的 Performance
            performance = success_model.counterfactual_evaluation(
                current_policy.default_effort,
                capability,
            )

            # 2. 找到最佳性能
            best_performance = max(performance.values())

            # 3. 筛选候选：Performance >= Best - ε
            quality_floor = best_performance - self.performance_tolerance
            candidates = [effort for effort, perf in performance.items() if perf >= quality_floor]

            if not candidates:
                # 降级：如果没有候选，保持当前
                candidates = [current_policy.default_effort[capability]]

            # 4. 估计每个候选的 Cost
            costs = {
                effort: cost_estimator.estimate_cost(capability, effort) for effort in candidates
            }

            # 5. 选择 Cost 最低的
            optimal_effort = min(costs, key=lambda effort: costs[effort])
            new_effort[capability] = optimal_effort

        return AdaptivePolicy(
            default_effort=new_effort,
            version=current_policy.version + 1,
            timestamp=datetime.now().isoformat(),
            total_tasks=historical_data.get_total_tasks(),
        )
