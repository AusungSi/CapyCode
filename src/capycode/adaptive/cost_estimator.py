"""成本估计器：统计和估计每个 (Capability, Starting Effort) 的成本"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import HistoricalData


class CostEstimator:
    """统计和估计成本"""

    def __init__(self, historical_data: HistoricalData):
        self.historical_data = historical_data

    def estimate_cost(self, capability: str, start_effort: str) -> float:
        """估计从 start_effort 开始的平均总成本（包含 escalation）"""

        # 筛选相关样本
        samples = [
            s
            for s in self.historical_data.samples
            if capability in s.start_effort_vector
            and s.start_effort_vector[capability] == start_effort
        ]

        if not samples:
            # 无历史数据，返回估计值
            return self._default_cost(start_effort)

        # 统计平均成本（包含 escalation）
        costs: list[float] = []
        for sample in samples:
            if capability in sample.capability_stats:
                value = sample.capability_stats[capability].get("total_cost")
                if isinstance(value, int | float):
                    costs.append(float(value))

        if not costs:
            return self._default_cost(start_effort)

        return sum(costs) / len(costs)

    def _default_cost(self, effort: str) -> float:
        """默认成本估计"""
        return {
            "low": 0.10,
            "medium": 0.20,
            "high": 0.40,
        }[effort]

    def estimate_all_costs(self, capability: str) -> dict[str, float]:
        """估计某个 Capability 所有 effort 的成本"""
        return {
            "low": self.estimate_cost(capability, "low"),
            "medium": self.estimate_cost(capability, "medium"),
            "high": self.estimate_cost(capability, "high"),
        }
