"""探索策略：选择要探索的 Capability 和 Effort"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .models import HistoricalData


class ExplorationStrategy:
    """管理 Capability 和 Effort 的探索"""

    CAPABILITIES: ClassVar[tuple[str, ...]] = (
        "retrieval",
        "understanding",
        "planning",
        "editing",
        "diagnosis",
        "verification",
    )
    EFFORTS: ClassVar[tuple[str, ...]] = ("low", "medium", "high")

    def __init__(
        self,
        n_warm: int = 5,
        ucb_beta: float = 1.0,
        rng: random.Random | None = None,
    ) -> None:
        self.n_warm = n_warm
        self.ucb_beta = ucb_beta
        self.rng = rng or random.Random()

    def select_capability(self, historical_data: HistoricalData) -> str | None:
        """选择要探索的 Capability"""

        # 统计每个 Capability 的探索次数
        exploration_counts = {
            c: historical_data.get_exploration_count(c) for c in self.CAPABILITIES
        }

        # Warm-up 阶段：优先欠采样的 Capability
        under_sampled = [c for c, count in exploration_counts.items() if count < self.n_warm]
        if under_sampled:
            return self.rng.choice(under_sampled)

        # UCB 阶段
        total_tasks = historical_data.get_total_tasks()
        if total_tasks == 0:
            return self.rng.choice(self.CAPABILITIES)

        ucb_scores = {}

        for capability in self.CAPABILITIES:
            g_c = self._calculate_potential_gain(capability, historical_data)
            n_c = exploration_counts[capability]

            exploration_bonus = self.ucb_beta * math.sqrt(math.log(total_tasks + 1) / (n_c + 1))

            ucb_scores[capability] = g_c + exploration_bonus

        return max(ucb_scores, key=lambda capability: ucb_scores[capability])

    def _calculate_potential_gain(self, capability: str, historical_data: HistoricalData) -> float:
        """计算 Capability 的潜在优化空间 G(c)"""

        samples = historical_data.get_samples_by_capability(capability)
        if not samples:
            return 0.5  # 默认中等优化空间

        # 简化版本：基于成功率判断
        success_count = sum(1 for s in samples if s.outcome)
        success_rate = success_count / len(samples) if samples else 0.5

        # 成功率低或成功率高都有优化空间
        if success_rate < 0.7:
            return 0.8  # 性能不足，有提升空间
        elif success_rate > 0.9:
            return 0.6  # 性能很好，有降低成本的空间
        else:
            return 0.4  # 中等，优先级较低

    def select_effort(
        self,
        capability: str,
        current_effort: str,
        historical_data: HistoricalData,
    ) -> str:
        """选择探索的 Effort（只探索相邻）"""

        # 获取该 Capability 的历史表现
        samples = historical_data.get_samples_by_capability(capability)

        if not samples:
            # 无历史数据，随机选择相邻
            return self._random_neighbor(current_effort)

        # 统计当前 effort 的表现
        current_samples = [
            s for s in samples if s.start_effort_vector.get(capability) == current_effort
        ]

        if not current_samples:
            return self._random_neighbor(current_effort)

        current_success_rate = sum(s.outcome for s in current_samples) / len(current_samples)

        # 决策规则
        if current_success_rate < 0.7:
            # 性能不足，向上探索
            return self._upgrade_effort(current_effort)
        elif current_success_rate > 0.9:
            # 性能很好，尝试降低成本
            return self._downgrade_effort(current_effort)
        else:
            # 不确定，选择样本更少的方向
            low_count = len([s for s in samples if s.start_effort_vector.get(capability) == "low"])
            high_count = len(
                [s for s in samples if s.start_effort_vector.get(capability) == "high"]
            )

            if current_effort == "medium":
                return "low" if low_count < high_count else "high"
            else:
                return "medium"

    def _random_neighbor(self, effort: str) -> str:
        """随机选择相邻 effort"""
        neighbors = {
            "low": ["medium"],
            "medium": ["low", "high"],
            "high": ["medium"],
        }
        return self.rng.choice(neighbors[effort])

    def _upgrade_effort(self, effort: str) -> str:
        """向上升级 effort"""
        upgrade = {
            "low": "medium",
            "medium": "high",
            "high": "high",
        }
        return upgrade[effort]

    def _downgrade_effort(self, effort: str) -> str:
        """向下降级 effort"""
        downgrade = {
            "low": "low",
            "medium": "low",
            "high": "medium",
        }
        return downgrade[effort]

    def prepare_task_config(
        self,
        current_policy: dict[str, str],
        historical_data: HistoricalData,
    ) -> tuple[dict[str, str], str | None]:
        """准备 Task 执行配置

        Returns:
            (start_effort_config, explored_capability)
        """

        # 选择要探索的 Capability
        explored_capability = self.select_capability(historical_data)

        if explored_capability is None:
            # 不需要探索，使用当前 Policy
            return current_policy.copy(), None

        # 选择探索的 Effort
        current_effort = current_policy[explored_capability]
        candidate_effort = self.select_effort(
            explored_capability,
            current_effort,
            historical_data,
        )

        # 应用探索配置
        start_effort_config = current_policy.copy()
        start_effort_config[explored_capability] = candidate_effort

        return start_effort_config, explored_capability
