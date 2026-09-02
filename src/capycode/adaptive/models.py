"""自适应学习系统的核心数据结构"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CapabilityStats:
    """单个 Capability 在一次 Task 中的统计"""

    # Effort 路径
    start_effort: str  # 起始 effort (low/medium/high)
    max_effort: str  # 最终达到的 effort
    escalation_count: int  # 升级次数

    # 成本
    total_cost: float  # 总成本（包含所有 escalation）
    total_tokens: int  # 总 Token
    total_latency_seconds: float  # 总延迟

    # 执行统计
    step_count: int  # 该 Capability 执行的步骤数
    tool_calls: int  # 工具调用次数

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityStats:
        return cls(**data)


@dataclass
class TaskSample:
    """单个 Task 的学习样本"""

    # 基本信息
    task_id: str
    timestamp: str  # ISO format

    # 核心数据
    start_effort_vector: dict[str, str]  # {capability: effort}
    outcome: bool  # True = Success, False = Failure
    capability_stats: dict[str, dict[str, Any]]  # {capability: CapabilityStats}

    # 探索信息
    explored_capability: str | None  # 本次主动探索的 Capability
    is_exploration: bool  # 是否为探索任务

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "start_effort_vector": self.start_effort_vector,
            "outcome": self.outcome,
            "capability_stats": self.capability_stats,
            "explored_capability": self.explored_capability,
            "is_exploration": self.is_exploration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSample:
        return cls(
            task_id=data["task_id"],
            timestamp=data["timestamp"],
            start_effort_vector=data["start_effort_vector"],
            outcome=data["outcome"],
            capability_stats=data["capability_stats"],
            explored_capability=data.get("explored_capability"),
            is_exploration=data.get("is_exploration", False),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> TaskSample:
        return cls.from_dict(json.loads(json_str))


class HistoricalData:
    """全部 Task 的历史数据"""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.samples: list[TaskSample] = []

    def add_sample(self, sample: TaskSample) -> None:
        """添加新样本并增量保存"""
        self.samples.append(sample)
        self._save_incremental(sample)

    def load(self) -> None:
        """从 JSONL 加载全部历史"""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            return

        self.samples = []
        with open(self.storage_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = TaskSample.from_json(line)
                        self.samples.append(sample)
                    except Exception as e:
                        print(f"Warning: Failed to parse sample: {e}")

    def _save_incremental(self, sample: TaskSample) -> None:
        """增量保存单个样本"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(sample.to_json() + "\n")

    def get_samples_by_capability(
        self, capability: str, start_effort: str | None = None
    ) -> list[TaskSample]:
        """查询特定 Capability 的样本"""
        result = []
        for sample in self.samples:
            if capability in sample.start_effort_vector:
                if start_effort is None or sample.start_effort_vector[capability] == start_effort:
                    result.append(sample)
        return result

    def get_exploration_count(self, capability: str) -> int:
        """统计 Capability 被探索的次数"""
        return sum(1 for s in self.samples if s.explored_capability == capability)

    def get_total_tasks(self) -> int:
        """获取总任务数"""
        return len(self.samples)


@dataclass
class AdaptivePolicy:
    """自适应学习的策略"""

    # 每个 Capability 的默认 starting effort
    default_effort: dict[str, str]  # {capability: effort}

    # 策略版本和时间戳
    version: int
    timestamp: str  # ISO format

    # 训练信息
    total_tasks: int  # 基于多少个 Task 训练

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_effort": self.default_effort,
            "version": self.version,
            "timestamp": self.timestamp,
            "total_tasks": self.total_tasks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptivePolicy:
        return cls(
            default_effort=data["default_effort"],
            version=data["version"],
            timestamp=data["timestamp"],
            total_tasks=data["total_tasks"],
        )

    def to_yaml(self) -> str:
        """导出为 Profile YAML 格式"""
        lines = [f"# Adaptive Policy v{self.version}"]
        lines.append(f"# Generated: {self.timestamp}")
        lines.append(f"# Based on {self.total_tasks} tasks")
        lines.append("")

        for capability, effort in self.default_effort.items():
            lines.append(f"{capability}_balanced:")
            lines.append(f"  reasoning_effort: {effort}")

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """保存为 JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> AdaptivePolicy:
        """从 JSON 加载"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_default(cls) -> AdaptivePolicy:
        """创建默认策略"""
        return cls(
            default_effort={
                "retrieval": "low",
                "understanding": "medium",
                "planning": "medium",
                "editing": "medium",
                "diagnosis": "medium",
                "verification": "low",
            },
            version=0,
            timestamp=datetime.now().isoformat(),
            total_tasks=0,
        )
