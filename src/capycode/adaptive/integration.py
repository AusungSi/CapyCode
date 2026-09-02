"""自适应学习系统与 Agent Loop 的集成接口"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from .classifier import SuccessClassifier
from .cost_estimator import CostEstimator
from .exploration import ExplorationStrategy
from .models import AdaptivePolicy, HistoricalData, TaskSample
from .policy_optimizer import PolicyOptimizer

if TYPE_CHECKING:
    from datetime import datetime


class AdaptiveStatus(TypedDict):
    enabled: bool
    total_tasks: int
    policy_version: int
    current_policy: dict[str, str]
    model_fitted: bool
    exploration_counts: dict[str, int]


class AdaptiveRuntime:
    """自适应学习运行时集成"""

    def __init__(
        self,
        storage_dir: Path,
        *,
        enabled: bool = True,
        optimize_interval: int = 10,
        performance_tolerance: float = 0.05,
        n_warm: int = 2,
    ) -> None:
        if optimize_interval <= 0:
            raise ValueError("optimize_interval must be positive")
        self.enabled = enabled
        self.optimize_interval = optimize_interval
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.historical_data = HistoricalData(storage_dir / "samples.jsonl")
        self.historical_data.load()

        self.exploration = ExplorationStrategy(n_warm=n_warm)
        try:
            self.success_model: SuccessClassifier | None = SuccessClassifier()
        except ImportError:
            self.success_model = None
        self.cost_estimator = CostEstimator(self.historical_data)
        self.policy_optimizer = PolicyOptimizer(performance_tolerance=performance_tolerance)

        # 加载或创建 Policy
        self.current_policy = self._load_latest_policy()

        # 当前 Task 状态
        self.current_task_id: str | None = None
        self.current_config: dict[str, str] | None = None
        self.current_explored: str | None = None
        self.task_start_time: datetime | None = None

    def _load_latest_policy(self) -> AdaptivePolicy:
        """加载最新的 Policy"""
        policy_files = [
            *self.storage_dir.glob("policy_v*.json"),
            # Compatibility with early prototypes that stored JSON under a YAML suffix.
            *self.storage_dir.glob("policy_v*.yaml"),
        ]
        if policy_files:
            latest = max(
                policy_files,
                key=lambda path: int(path.stem.removeprefix("policy_v")),
            )
            return AdaptivePolicy.load(latest)

        # 创建默认 Policy
        policy = AdaptivePolicy.create_default()
        policy.save(self.storage_dir / "policy_v0.json")
        return policy

    def prepare_task(self, task_id: str) -> dict[str, str]:
        """Task 开始前准备配置"""
        if not self.enabled:
            return self.current_policy.default_effort.copy()

        from datetime import datetime

        self.current_task_id = task_id
        self.task_start_time = datetime.now()

        # 使用探索策略生成配置
        config, explored = self.exploration.prepare_task_config(
            self.current_policy.default_effort,
            self.historical_data,
        )

        self.current_config = config
        self.current_explored = explored

        return config

    def collect_sample(
        self,
        outcome: bool,
        capability_stats: dict[str, dict[str, Any]],
    ) -> None:
        """Task 结束后收集样本"""
        if not self.enabled or self.current_task_id is None or self.current_config is None:
            return

        from datetime import datetime

        # 创建样本
        sample = TaskSample(
            task_id=self.current_task_id,
            timestamp=datetime.now().isoformat(),
            start_effort_vector=self.current_config,
            outcome=outcome,
            capability_stats=capability_stats,
            explored_capability=self.current_explored,
            is_exploration=self.current_explored is not None,
        )

        # 保存样本
        self.historical_data.add_sample(sample)

        # 重置状态
        self.current_task_id = None
        self.current_config = None
        self.current_explored = None
        self.task_start_time = None

        # 检查是否需要优化 Policy
        if self.historical_data.get_total_tasks() % self.optimize_interval == 0:
            self.optimize_policy()

    def optimize_policy(self) -> None:
        """优化 Policy"""
        if not self.enabled or self.success_model is None:
            return

        # 训练 Success Model
        self.success_model.fit(self.historical_data)

        if not self.success_model.is_fitted:
            return  # 数据不足，跳过

        # 优化 Policy
        new_policy = self.policy_optimizer.optimize(
            self.current_policy,
            self.historical_data,
            self.success_model,
            self.cost_estimator,
        )

        # 保存新 Policy
        policy_path = self.storage_dir / f"policy_v{new_policy.version}.json"
        new_policy.save(policy_path)

        self.current_policy = new_policy

    def get_status(self) -> AdaptiveStatus:
        """获取当前状态"""
        return {
            "enabled": self.enabled,
            "total_tasks": self.historical_data.get_total_tasks(),
            "policy_version": self.current_policy.version,
            "current_policy": self.current_policy.default_effort,
            "model_fitted": self.success_model.is_fitted if self.success_model else False,
            "exploration_counts": {
                cap: self.historical_data.get_exploration_count(cap)
                for cap in ExplorationStrategy.CAPABILITIES
            },
        }
