"""Measured capability-routing data produced by the P2 profiler."""

from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import Capability


class MeasurementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileMeasurement(MeasurementModel):
    """One routed step, labelled with its final task outcome."""

    profile_id: str = Field(min_length=1)
    capability: Capability
    model_id: str = Field(min_length=1)
    succeeded: bool
    cost: float = Field(ge=0)
    latency_seconds: float = Field(ge=0)


class ProfileMetric(MeasurementModel):
    profile_id: str = Field(min_length=1)
    capability: Capability
    model_id: str = Field(min_length=1)
    samples: int = Field(ge=1)
    successes: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    mean_cost: float = Field(default=0, ge=0)
    mean_latency_seconds: float = Field(ge=0)
    expected_cost_per_success: float | None = Field(default=None, ge=0)
    efficiency: float | None = Field(default=None, ge=0)


class ProfileSelection(MeasurementModel):
    profile_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    capability: Capability
    samples: int = Field(ge=1)
    success_rate: float = Field(ge=0, le=1)
    mean_cost: float = Field(ge=0)
    expected_cost_per_success: float = Field(ge=0)
    mean_latency_seconds: float = Field(ge=0)


class ProfiledRoutingArtifact(MeasurementModel):
    """Portable, evidence-backed routing choices consumed by ``ProfileRouter``."""

    schema_version: int = 1
    generated_at: datetime
    minimum_samples: int = Field(ge=1)
    reliability_threshold: float = Field(ge=0, le=1)
    metrics: tuple[ProfileMetric, ...]
    selected_by_capability: dict[str, ProfileSelection] = Field(default_factory=dict)
    training_task_fingerprints: dict[str, str] = Field(default_factory=dict)
    source_campaign_id: str | None = None
    candidate_model_ids: tuple[str, ...] = ()

    def selection_for(self, capability: Capability) -> ProfileSelection | None:
        return self.selected_by_capability.get(capability.value)

    @classmethod
    def from_measurements(
        cls,
        measurements: Iterable[ProfileMeasurement],
        *,
        minimum_samples: int = 2,
        reliability_threshold: float = 0.6,
        source_campaign_id: str | None = None,
        candidate_model_ids: Iterable[str] = (),
    ) -> ProfiledRoutingArtifact:
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        if not 0 <= reliability_threshold <= 1:
            raise ValueError("reliability_threshold must be between 0 and 1")
        grouped: dict[tuple[str, Capability, str], list[ProfileMeasurement]] = defaultdict(list)
        for measurement in measurements:
            grouped[(measurement.profile_id, measurement.capability, measurement.model_id)].append(
                measurement
            )
        metrics: list[ProfileMetric] = []
        for (profile_id, capability, model_id), samples in sorted(
            grouped.items(), key=lambda item: (item[0][1].value, item[0][0], item[0][2])
        ):
            count = len(samples)
            successes = sum(sample.succeeded for sample in samples)
            success_rate = successes / count
            mean_cost = sum(sample.cost for sample in samples) / count
            metrics.append(
                ProfileMetric(
                    profile_id=profile_id,
                    capability=capability,
                    model_id=model_id,
                    samples=count,
                    successes=successes,
                    success_rate=success_rate,
                    mean_cost=mean_cost,
                    mean_latency_seconds=sum(sample.latency_seconds for sample in samples) / count,
                    expected_cost_per_success=(mean_cost / success_rate if success_rate else None),
                    efficiency=(success_rate / mean_cost if mean_cost else None),
                )
            )
        selected: dict[str, ProfileSelection] = {}
        for capability in Capability:
            eligible = [
                metric
                for metric in metrics
                if metric.capability == capability
                and metric.samples >= minimum_samples
                and metric.success_rate >= reliability_threshold
                and metric.expected_cost_per_success is not None
            ]
            if not eligible:
                continue
            winner = min(
                eligible,
                key=lambda item: (
                    item.expected_cost_per_success or math.inf,
                    item.mean_latency_seconds,
                    item.profile_id,
                    item.model_id,
                ),
            )
            selected[capability.value] = ProfileSelection(
                profile_id=winner.profile_id,
                model_id=winner.model_id,
                capability=winner.capability,
                samples=winner.samples,
                success_rate=winner.success_rate,
                mean_cost=winner.mean_cost,
                expected_cost_per_success=winner.expected_cost_per_success or 0,
                mean_latency_seconds=winner.mean_latency_seconds,
            )
        return cls(
            generated_at=datetime.now(UTC),
            minimum_samples=minimum_samples,
            reliability_threshold=reliability_threshold,
            metrics=tuple(metrics),
            selected_by_capability=selected,
            source_campaign_id=source_campaign_id,
            candidate_model_ids=tuple(
                dict.fromkeys(
                    model_id.strip() for model_id in candidate_model_ids if model_id.strip()
                )
            ),
        )

    def validate_holdout(
        self,
        task_fingerprints: dict[str, str],
        *,
        allow_overlap: bool = False,
    ) -> None:
        if allow_overlap:
            return
        identifier_overlap = sorted(set(task_fingerprints) & set(self.training_task_fingerprints))
        training_by_fingerprint = {
            fingerprint: task_id for task_id, fingerprint in self.training_task_fingerprints.items()
        }
        content_overlap = sorted(
            task_id
            for task_id, fingerprint in task_fingerprints.items()
            if fingerprint in training_by_fingerprint and task_id not in identifier_overlap
        )
        if identifier_overlap or content_overlap:
            details = identifier_overlap + [
                f"{task_id} (same content as {training_by_fingerprint[task_fingerprints[task_id]]})"
                for task_id in content_overlap
            ]
            raise ValueError(
                "evaluation tasks overlap with the profiling set: "
                + ", ".join(details)
                + ". Select held-out task IDs, or explicitly use --allow-overlap for debugging."
            )

    @classmethod
    def load(cls, path: Path) -> ProfiledRoutingArtifact:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"unable to read profiled routing artifact {path}: {exc}") from exc

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix="profiles-", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(self.model_dump_json(indent=2))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
