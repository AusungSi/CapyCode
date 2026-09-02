from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from capycode.adaptive import (
    AdaptivePolicy,
    AdaptiveRuntime,
    CapabilityStats,
    CostEstimator,
    ExplorationStrategy,
    HistoricalData,
    PolicyOptimizer,
    TaskSample,
)
from capycode.adaptive.classifier import SKLEARN_AVAILABLE, SuccessClassifier
from capycode.adaptive.visualization import (
    format_adaptive_history,
    format_adaptive_performance,
    format_adaptive_status,
)


def make_sample(
    task_id: str,
    *,
    outcome: bool = True,
    effort: str = "medium",
    explored: str | None = "planning",
    cost: float = 0.2,
) -> TaskSample:
    efforts = {capability: effort for capability in ExplorationStrategy.CAPABILITIES}
    return TaskSample(
        task_id=task_id,
        timestamp="2026-09-02T12:00:00",
        start_effort_vector=efforts,
        outcome=outcome,
        capability_stats={"planning": {"total_cost": cost}},
        explored_capability=explored,
        is_exploration=explored is not None,
    )


def test_capability_stats_round_trip() -> None:
    stats = CapabilityStats(
        start_effort="low",
        max_effort="medium",
        escalation_count=1,
        total_cost=0.3,
        total_tokens=120,
        total_latency_seconds=1.5,
        step_count=2,
        tool_calls=1,
    )

    assert CapabilityStats.from_dict(stats.to_dict()) == stats


def test_historical_data_persists_jsonl(tmp_path: Path) -> None:
    storage = tmp_path / "samples.jsonl"
    data = HistoricalData(storage)
    sample = make_sample("task-1")

    data.add_sample(sample)
    reloaded = HistoricalData(storage)
    reloaded.load()

    assert reloaded.samples == [sample]
    assert json.loads(storage.read_text(encoding="utf-8"))["task_id"] == "task-1"


def test_exploration_changes_one_capability_deterministically(tmp_path: Path) -> None:
    data = HistoricalData(tmp_path / "samples.jsonl")
    policy = AdaptivePolicy.create_default().default_effort
    strategy = ExplorationStrategy(n_warm=1, rng=random.Random(7))

    config, explored = strategy.prepare_task_config(policy, data)

    assert explored in ExplorationStrategy.CAPABILITIES
    assert config != policy
    assert [key for key in config if config[key] != policy[key]] == [explored]


def test_cost_estimator_uses_observations_and_defaults(tmp_path: Path) -> None:
    data = HistoricalData(tmp_path / "samples.jsonl")
    data.samples = [make_sample("one", cost=0.2), make_sample("two", cost=0.4)]
    estimator = CostEstimator(data)

    assert estimator.estimate_cost("planning", "medium") == pytest.approx(0.3)
    assert estimator.estimate_cost("planning", "high") == pytest.approx(0.4)


class StableSuccessModel:
    def counterfactual_evaluation(
        self, base_policy: dict[str, str], target_capability: str
    ) -> dict[str, float]:
        del base_policy, target_capability
        return {"low": 0.90, "medium": 0.95, "high": 0.99}


def test_optimizer_selects_cheapest_effort_within_quality_floor(tmp_path: Path) -> None:
    data = HistoricalData(tmp_path / "samples.jsonl")
    current = AdaptivePolicy.create_default()
    optimized = PolicyOptimizer(performance_tolerance=0.05).optimize(
        current,
        data,
        StableSuccessModel(),  # type: ignore[arg-type]
        CostEstimator(data),
    )

    assert set(optimized.default_effort.values()) == {"medium"}
    assert optimized.version == current.version + 1


def test_runtime_collects_and_reloads_samples(tmp_path: Path) -> None:
    runtime = AdaptiveRuntime(tmp_path, optimize_interval=20, n_warm=1)
    config = runtime.prepare_task("task-1")
    runtime.collect_sample(True, {"planning": {"total_cost": 0.25}})

    assert runtime.get_status()["total_tasks"] == 1
    assert runtime.current_task_id is None
    assert config
    assert (tmp_path / "policy_v0.json").exists()
    assert AdaptiveRuntime(tmp_path).get_status()["total_tasks"] == 1


def test_runtime_loads_highest_numeric_policy_version(tmp_path: Path) -> None:
    old_policy = AdaptivePolicy.create_default()
    old_policy.version = 9
    old_policy.save(tmp_path / "policy_v9.json")
    new_policy = AdaptivePolicy.create_default()
    new_policy.version = 10
    new_policy.save(tmp_path / "policy_v10.json")

    assert AdaptiveRuntime(tmp_path).current_policy.version == 10


def test_disabled_runtime_returns_policy_copy(tmp_path: Path) -> None:
    runtime = AdaptiveRuntime(tmp_path, enabled=False)

    config = runtime.prepare_task("ignored")
    config["planning"] = "high"
    runtime.collect_sample(True, {})

    assert runtime.current_policy.default_effort["planning"] == "medium"
    assert runtime.get_status()["total_tasks"] == 0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="adaptive ML extra is not installed")
def test_classifier_waits_for_both_outcome_classes(tmp_path: Path) -> None:
    data = HistoricalData(tmp_path / "samples.jsonl")
    data.samples = [make_sample(str(index), outcome=True) for index in range(10)]
    classifier = SuccessClassifier()

    classifier.fit(data)

    assert classifier.is_fitted is False


def test_visualization_formats_runtime_state(tmp_path: Path) -> None:
    runtime = AdaptiveRuntime(tmp_path, optimize_interval=20)
    for index in range(5):
        runtime.historical_data.add_sample(make_sample(str(index), outcome=index != 0))

    assert "自适应学习系统" in format_adaptive_status(runtime)
    assert "最近的样本" in format_adaptive_history(runtime, limit=2)
    assert "80.0%" in format_adaptive_performance(runtime)


def test_runtime_rejects_invalid_optimization_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        AdaptiveRuntime(tmp_path, optimize_interval=0)
