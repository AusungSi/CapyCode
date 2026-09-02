"""P2 profiling and holdout evaluation built on the frozen P0 task runner."""

from __future__ import annotations

import asyncio
import csv
import io
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from capycode.capability import Capability, ProfiledRoutingArtifact, ProfileMeasurement
from capycode.llm.types import ReasoningEffort
from capycode.trace import RUN_EVENT_ADAPTER, StepTraceEvent

from .baseline import GateCampaignReport, GateTaskCatalog, P0GateRunner, TaskExecutor

ProgressSink = Callable[[str], None]
ExecutorFactory = Callable[[str], TaskExecutor]


class ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProfileCampaign(ExperimentModel):
    model_id: str
    report_path: str
    passed_runs: int = Field(ge=0)
    total_runs: int = Field(ge=1)


class ProfilingCampaignReport(ExperimentModel):
    schema_version: int = 1
    campaign_id: str
    started_at: datetime
    finished_at: datetime
    model_ids: list[str]
    repeats: int = Field(ge=1)
    minimum_samples: int = Field(ge=1)
    reliability_threshold: float = Field(ge=0, le=1)
    quality_tolerance: float = Field(ge=0, le=1)
    measurements: int = Field(ge=0)
    selected_capabilities: int = Field(ge=0)
    campaigns: list[ProfileCampaign]
    task_fingerprints: dict[str, str]
    artifact_path: str
    measurements_path: str
    leaderboard_csv_path: str
    leaderboard_markdown_path: str


class EvaluationStrategy(ExperimentModel):
    strategy_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    label: str = Field(min_length=1)
    configuration: dict[str, str] = Field(default_factory=dict)


class CampaignManifest(ExperimentModel):
    schema_version: int = 1
    campaign_id: str
    campaign_type: Literal["profiling", "evaluation"]
    created_at: datetime
    repeats: int = Field(ge=1)
    task_fingerprints: dict[str, str]
    model_ids: list[str] = Field(default_factory=list)
    strategies: list[EvaluationStrategy] = Field(default_factory=list)


class CampaignProgress(ExperimentModel):
    schema_version: int = 1
    campaign_id: str
    status: Literal["running", "completed", "failed", "cancelled"]
    completed_items: list[str] = Field(default_factory=list)
    current_item: str | None = None
    error: str | None = None


class StrategyEvaluation(ExperimentModel):
    strategy_id: str
    label: str
    report_path: str
    passed_runs: int = Field(ge=0)
    total_runs: int = Field(ge=1)
    pass_rate: float = Field(ge=0, le=1)
    pass_at_1: float = Field(ge=0, le=1)
    total_cost: float = Field(ge=0)
    cost_per_success: float | None = Field(default=None, ge=0)
    total_latency_seconds: float = Field(ge=0)
    average_steps: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    tool_requests: int = Field(ge=0)
    tool_failures: int = Field(ge=0)
    tool_failure_rate: float = Field(ge=0, le=1)
    infrastructure_errors: int = Field(ge=0)
    currency: str


class RoutingEvaluationReport(ExperimentModel):
    schema_version: int = 1
    campaign_id: str
    started_at: datetime
    finished_at: datetime
    repeats: int = Field(ge=1)
    task_ids: list[str]
    task_fingerprints: dict[str, str]
    strategies: list[StrategyEvaluation]


def _safe_model_directory(model_id: str, position: int) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip(".-")[:48]
    return f"{position:02d}-{normalized or 'model'}"


def measurements_from_report(report: GateCampaignReport) -> list[ProfileMeasurement]:
    """Aggregate routed steps per run/capability and label them with task success.

    Counting every step as an independent sample rewards long trajectories and
    overstates confidence. A run contributes at most one observation for each
    profile/model/effort/capability tuple.
    """
    measurements: list[ProfileMeasurement] = []
    for result in report.results:
        if not result.trace_path:
            continue
        trace_path = Path(result.trace_path)
        try:
            lines = trace_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"unable to read profiling trace {trace_path}: {exc}") from exc
        aggregated: dict[
            tuple[str, Capability, str, ReasoningEffort | None], tuple[float, float]
        ] = {}
        for line in lines:
            if not line.strip():
                continue
            event = RUN_EVENT_ADAPTER.validate_json(line)
            if not isinstance(event, StepTraceEvent):
                continue
            if event.profile_id is None or event.capability is None:
                continue
            key = (
                event.profile_id,
                Capability(event.capability),
                event.model_id,
                event.reasoning_effort,
            )
            cost, latency = aggregated.get(key, (0.0, 0.0))
            aggregated[key] = (cost + event.cost, latency + event.latency_seconds)
        for (profile_id, capability, model_id, reasoning_effort), (
            cost,
            latency,
        ) in aggregated.items():
            measurements.append(
                ProfileMeasurement(
                    profile_id=profile_id,
                    capability=capability,
                    model_id=model_id,
                    reasoning_effort=reasoning_effort,
                    succeeded=result.status == "passed",
                    cost=cost,
                    latency_seconds=latency,
                )
            )
    return measurements


class P2ProfilingRunner:
    """Run candidate real models and materialize a measured routing artifact."""

    def __init__(
        self,
        *,
        catalog: GateTaskCatalog | None = None,
        output_root: Path | None = None,
        progress: ProgressSink | None = None,
    ) -> None:
        self.catalog = catalog or GateTaskCatalog()
        self.output_root = (output_root or Path.cwd() / ".capy" / "profiling").resolve()
        self.progress = progress or (lambda _message: None)

    async def run(
        self,
        executor_factory: ExecutorFactory,
        *,
        model_ids: Sequence[str],
        repeats: int = 1,
        task_ids: Sequence[str] | None = None,
        minimum_samples: int = 2,
        reliability_threshold: float = 0.6,
        quality_tolerance: float = 0.05,
    ) -> tuple[ProfilingCampaignReport, ProfiledRoutingArtifact, Path]:
        selected_models = list(dict.fromkeys(item.strip() for item in model_ids if item.strip()))
        if not selected_models:
            raise ValueError("at least one real model ID is required for profiling")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        training_tasks = self.catalog.select(task_ids)
        task_fingerprints = {task.manifest.task_id: task.fingerprint for task in training_tasks}
        campaign_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        root = self.output_root / campaign_id
        root.mkdir(parents=True, exist_ok=False)
        started_at = datetime.now(UTC)
        self._write_model(
            root / "manifest.json",
            CampaignManifest(
                campaign_id=campaign_id,
                campaign_type="profiling",
                created_at=started_at,
                repeats=repeats,
                task_fingerprints=task_fingerprints,
                model_ids=selected_models,
            ),
        )
        self._write_progress(root, campaign_id, status="running")
        all_measurements: list[ProfileMeasurement] = []
        campaigns: list[ProfileCampaign] = []
        try:
            for position, model_id in enumerate(selected_models, start=1):
                self.progress(f"profile {position}/{len(selected_models)} model: {model_id}")
                self._write_progress(
                    root,
                    campaign_id,
                    status="running",
                    completed_items=[item.model_id for item in campaigns],
                    current_item=model_id,
                )

                def profile_progress(message: str, *, current_model: str = model_id) -> None:
                    self.progress(f"{current_model} · {message}")

                runner = P0GateRunner(
                    catalog=self.catalog,
                    output_root=root / "p0" / _safe_model_directory(model_id, position),
                    progress=profile_progress,
                )
                gate_report, report_root = await runner.run(
                    executor_factory(model_id),
                    model_id=model_id,
                    repeats=repeats,
                    task_ids=task_ids,
                )
                all_measurements.extend(measurements_from_report(gate_report))
                campaigns.append(
                    ProfileCampaign(
                        model_id=model_id,
                        report_path=str(report_root / "report.json"),
                        passed_runs=gate_report.passed_runs,
                        total_runs=gate_report.total_runs,
                    )
                )
                self._write_measurements(root, all_measurements)
                partial_artifact = ProfiledRoutingArtifact.from_measurements(
                    all_measurements,
                    minimum_samples=minimum_samples,
                    reliability_threshold=reliability_threshold,
                    quality_tolerance=quality_tolerance,
                    source_campaign_id=campaign_id,
                    candidate_model_ids=selected_models,
                ).model_copy(update={"training_task_fingerprints": task_fingerprints})
                partial_artifact.write(root / "profiles.partial.json")
                self._write_progress(
                    root,
                    campaign_id,
                    status="running",
                    completed_items=[item.model_id for item in campaigns],
                )
        except asyncio.CancelledError:
            self._write_progress(
                root,
                campaign_id,
                status="cancelled",
                completed_items=[item.model_id for item in campaigns],
            )
            raise
        except Exception as exc:
            self._write_progress(
                root,
                campaign_id,
                status="failed",
                completed_items=[item.model_id for item in campaigns],
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        artifact = ProfiledRoutingArtifact.from_measurements(
            all_measurements,
            minimum_samples=minimum_samples,
            reliability_threshold=reliability_threshold,
            quality_tolerance=quality_tolerance,
            source_campaign_id=campaign_id,
            candidate_model_ids=selected_models,
        )
        artifact = artifact.model_copy(update={"training_task_fingerprints": task_fingerprints})
        artifact.write(root / "profiles.json")
        profiling_report = ProfilingCampaignReport(
            campaign_id=campaign_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            model_ids=selected_models,
            repeats=repeats,
            minimum_samples=minimum_samples,
            reliability_threshold=reliability_threshold,
            quality_tolerance=quality_tolerance,
            measurements=len(all_measurements),
            selected_capabilities=len(artifact.selected_by_capability),
            campaigns=campaigns,
            task_fingerprints=task_fingerprints,
            artifact_path=str(root / "profiles.json"),
            measurements_path=str(root / "measurements.jsonl"),
            leaderboard_csv_path=str(root / "leaderboard.csv"),
            leaderboard_markdown_path=str(root / "leaderboard.md"),
        )
        self._write_profile_outputs(root, profiling_report, artifact)
        self._write_progress(
            root,
            campaign_id,
            status="completed",
            completed_items=[item.model_id for item in campaigns],
        )
        return profiling_report, artifact, root

    @staticmethod
    def _write_model(path: Path, value: ExperimentModel) -> None:
        P2ProfilingRunner._atomic_write(path, value.model_dump_json(indent=2) + "\n")

    @staticmethod
    def _write_measurements(root: Path, measurements: Sequence[ProfileMeasurement]) -> None:
        content = "".join(measurement.model_dump_json() + "\n" for measurement in measurements)
        P2ProfilingRunner._atomic_write(root / "measurements.jsonl", content)

    @staticmethod
    def _write_progress(
        root: Path,
        campaign_id: str,
        *,
        status: Literal["running", "completed", "failed", "cancelled"],
        completed_items: list[str] | None = None,
        current_item: str | None = None,
        error: str | None = None,
    ) -> None:
        P2ProfilingRunner._write_model(
            root / "progress.json",
            CampaignProgress(
                campaign_id=campaign_id,
                status=status,
                completed_items=completed_items or [],
                current_item=current_item,
                error=error,
            ),
        )

    @staticmethod
    def _write_profile_outputs(
        root: Path,
        report: ProfilingCampaignReport,
        artifact: ProfiledRoutingArtifact,
    ) -> None:
        P2ProfilingRunner._atomic_write(
            root / "report.json", report.model_dump_json(indent=2) + "\n"
        )
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "capability",
                "profile_id",
                "model_id",
                "reasoning_effort",
                "samples",
                "successes",
                "success_rate",
                "mean_cost",
                "mean_latency_seconds",
                "expected_cost_per_success",
                "efficiency",
                "selected",
            ]
        )
        selected = {
            (item.capability, item.profile_id, item.model_id)
            for item in artifact.selected_by_capability.values()
        }
        for metric in artifact.metrics:
            writer.writerow(
                [
                    metric.capability.value,
                    metric.profile_id,
                    metric.model_id,
                    metric.reasoning_effort or "default",
                    metric.samples,
                    metric.successes,
                    f"{metric.success_rate:.6f}",
                    f"{metric.mean_cost:.8f}",
                    f"{metric.mean_latency_seconds:.6f}",
                    ""
                    if metric.expected_cost_per_success is None
                    else f"{metric.expected_cost_per_success:.8f}",
                    "" if metric.efficiency is None else f"{metric.efficiency:.8f}",
                    str(
                        (metric.capability, metric.profile_id, metric.model_id) in selected
                    ).lower(),
                ]
            )
        P2ProfilingRunner._atomic_write(root / "leaderboard.csv", stream.getvalue())
        lines = [
            "# Capability Profile Leaderboard",
            "",
            f"- Campaign: `{report.campaign_id}`",
            f"- Models: {', '.join(f'`{model}`' for model in report.model_ids)}",
            f"- Step samples: {report.measurements}",
            (
                f"- Selection: at least {report.minimum_samples} samples and success rate >= "
                f"{report.reliability_threshold:.0%}"
            ),
            (
                "- Ranking: keep profiles within "
                f"{report.quality_tolerance:.0%} of the best success rate, then choose the "
                "lowest expected cost per success and latency."
            ),
            (
                "- Step outcomes are labelled with the final P0 task result; this is not an "
                "independent causal estimate."
            ),
            "",
            (
                "| Capability | Profile | Model | Effort | n | Success | Mean cost | "
                "Mean latency | "
                "Expected cost / success | Efficiency | Selected |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        selected_keys = {
            (item.capability, item.profile_id, item.model_id)
            for item in artifact.selected_by_capability.values()
        }
        for metric in artifact.metrics:
            expected_cost = (
                "-"
                if metric.expected_cost_per_success is None
                else f"{metric.expected_cost_per_success:.6f}"
            )
            efficiency = "-" if metric.efficiency is None else f"{metric.efficiency:.6f}"
            selected_label = (
                "yes"
                if (metric.capability, metric.profile_id, metric.model_id) in selected_keys
                else ""
            )
            lines.append(
                f"| {metric.capability.value} | {metric.profile_id} | {metric.model_id} | "
                f"{metric.reasoning_effort or 'default'} | "
                f"{metric.samples} | {metric.success_rate:.1%} | {metric.mean_cost:.6f} | "
                f"{metric.mean_latency_seconds:.3f}s | "
                f"{expected_cost} | {efficiency} | {selected_label} |"
            )
        P2ProfilingRunner._atomic_write(root / "leaderboard.md", "\n".join(lines) + "\n")
        summary_lines = [
            "# Capability Profiling Campaign",
            "",
            f"- Campaign: `{report.campaign_id}`",
            f"- Models: {', '.join(f'`{model}`' for model in report.model_ids)}",
            f"- Tasks: {len(report.task_fingerprints)}",
            f"- Repeats: {report.repeats}",
            f"- Step measurements: {report.measurements}",
            f"- Selected capabilities: {report.selected_capabilities}",
            "",
            "| Model | Passed runs | Total runs | P0 report |",
            "|---|---:|---:|---|",
        ]
        for campaign in report.campaigns:
            summary_lines.append(
                f"| {campaign.model_id} | {campaign.passed_runs} | "
                f"{campaign.total_runs} | `{campaign.report_path}` |"
            )
        summary_lines.extend(
            [
                "",
                f"- Routing artifact: `{report.artifact_path}`",
                f"- Measurements: `{report.measurements_path}`",
                f"- Leaderboard CSV: `{report.leaderboard_csv_path}`",
                f"- Leaderboard Markdown: `{report.leaderboard_markdown_path}`",
                "",
                (
                    "The artifact is selected only from candidates meeting the configured "
                    "minimum sample and reliability thresholds."
                ),
            ]
        )
        P2ProfilingRunner._atomic_write(root / "report.md", "\n".join(summary_lines) + "\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=path.name + "-", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


class RoutingEvaluationRunner:
    """Compare fixed and profiled policies with the same frozen tasks and runner."""

    def __init__(
        self,
        *,
        catalog: GateTaskCatalog | None = None,
        output_root: Path | None = None,
        progress: ProgressSink | None = None,
    ) -> None:
        self.catalog = catalog or GateTaskCatalog()
        self.output_root = (output_root or Path.cwd() / ".capy" / "evaluations").resolve()
        self.progress = progress or (lambda _message: None)

    async def run(
        self,
        strategies: Sequence[EvaluationStrategy],
        executor_factory: Callable[[EvaluationStrategy], TaskExecutor],
        *,
        repeats: int = 1,
        task_ids: Sequence[str] | None = None,
    ) -> tuple[RoutingEvaluationReport, Path]:
        selected = list(strategies)
        if len(selected) < 2:
            raise ValueError("routing evaluation requires at least two strategies")
        if len({strategy.strategy_id for strategy in selected}) != len(selected):
            raise ValueError("routing evaluation strategy IDs must be unique")
        tasks = self.catalog.select(task_ids)
        task_fingerprints = {task.manifest.task_id: task.fingerprint for task in tasks}
        campaign_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        root = self.output_root / campaign_id
        root.mkdir(parents=True, exist_ok=False)
        started_at = datetime.now(UTC)
        P2ProfilingRunner._write_model(
            root / "manifest.json",
            CampaignManifest(
                campaign_id=campaign_id,
                campaign_type="evaluation",
                created_at=started_at,
                repeats=repeats,
                task_fingerprints=task_fingerprints,
                strategies=selected,
            ),
        )
        P2ProfilingRunner._write_progress(root, campaign_id, status="running")
        results: list[StrategyEvaluation] = []
        try:
            for position, strategy in enumerate(selected, start=1):
                self.progress(f"evaluate {position}/{len(selected)} strategy: {strategy.label}")
                P2ProfilingRunner._write_progress(
                    root,
                    campaign_id,
                    status="running",
                    completed_items=[item.strategy_id for item in results],
                    current_item=strategy.strategy_id,
                )

                def evaluation_progress(message: str, *, label: str = strategy.label) -> None:
                    self.progress(f"{label} · {message}")

                runner = P0GateRunner(
                    catalog=self.catalog,
                    output_root=root / "strategies" / strategy.strategy_id,
                    progress=evaluation_progress,
                )
                gate_report, report_root = await runner.run(
                    executor_factory(strategy),
                    model_id=strategy.label,
                    repeats=repeats,
                    task_ids=task_ids,
                )
                results.append(
                    StrategyEvaluation(
                        strategy_id=strategy.strategy_id,
                        label=strategy.label,
                        report_path=str(report_root / "report.json"),
                        passed_runs=gate_report.passed_runs,
                        total_runs=gate_report.total_runs,
                        pass_rate=gate_report.pass_rate,
                        pass_at_1=gate_report.pass_at_1,
                        total_cost=gate_report.total_cost,
                        cost_per_success=(
                            gate_report.total_cost / gate_report.passed_runs
                            if gate_report.passed_runs
                            else None
                        ),
                        total_latency_seconds=gate_report.total_latency_seconds,
                        average_steps=gate_report.average_steps,
                        input_tokens=gate_report.total_input_tokens,
                        cached_input_tokens=gate_report.total_cached_input_tokens,
                        output_tokens=gate_report.total_output_tokens,
                        tool_requests=gate_report.total_tool_requests,
                        tool_failures=gate_report.total_tool_failures,
                        tool_failure_rate=gate_report.tool_failure_rate,
                        infrastructure_errors=sum(
                            result.status == "infrastructure_error"
                            for result in gate_report.results
                        ),
                        currency=gate_report.currency,
                    )
                )
                P2ProfilingRunner._write_model(
                    root / "report.partial.json",
                    RoutingEvaluationReport(
                        campaign_id=campaign_id,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        repeats=repeats,
                        task_ids=[task.manifest.task_id for task in tasks],
                        task_fingerprints=task_fingerprints,
                        strategies=results,
                    ),
                )
                P2ProfilingRunner._write_progress(
                    root,
                    campaign_id,
                    status="running",
                    completed_items=[item.strategy_id for item in results],
                )
        except asyncio.CancelledError:
            P2ProfilingRunner._write_progress(
                root,
                campaign_id,
                status="cancelled",
                completed_items=[item.strategy_id for item in results],
            )
            raise
        except Exception as exc:
            P2ProfilingRunner._write_progress(
                root,
                campaign_id,
                status="failed",
                completed_items=[item.strategy_id for item in results],
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        report = RoutingEvaluationReport(
            campaign_id=campaign_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            repeats=repeats,
            task_ids=[task.manifest.task_id for task in tasks],
            task_fingerprints=task_fingerprints,
            strategies=results,
        )
        self._write_outputs(root, report)
        P2ProfilingRunner._write_progress(
            root,
            campaign_id,
            status="completed",
            completed_items=[item.strategy_id for item in results],
        )
        return report, root

    @staticmethod
    def _write_outputs(root: Path, report: RoutingEvaluationReport) -> None:
        P2ProfilingRunner._atomic_write(
            root / "report.json", report.model_dump_json(indent=2) + "\n"
        )
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "strategy_id",
                "label",
                "passed_runs",
                "total_runs",
                "pass_rate",
                "pass_at_1",
                "total_cost",
                "cost_per_success",
                "total_latency_seconds",
                "average_steps",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "tool_requests",
                "tool_failures",
                "tool_failure_rate",
                "infrastructure_errors",
                "currency",
            ]
        )
        for item in report.strategies:
            writer.writerow(
                [
                    item.strategy_id,
                    item.label,
                    item.passed_runs,
                    item.total_runs,
                    f"{item.pass_rate:.6f}",
                    f"{item.pass_at_1:.6f}",
                    f"{item.total_cost:.8f}",
                    "" if item.cost_per_success is None else f"{item.cost_per_success:.8f}",
                    f"{item.total_latency_seconds:.6f}",
                    f"{item.average_steps:.6f}",
                    item.input_tokens,
                    item.cached_input_tokens,
                    item.output_tokens,
                    item.tool_requests,
                    item.tool_failures,
                    f"{item.tool_failure_rate:.6f}",
                    item.infrastructure_errors,
                    item.currency,
                ]
            )
        P2ProfilingRunner._atomic_write(root / "comparison.csv", stream.getvalue())
        lines = [
            "# Routing Evaluation",
            "",
            f"- Campaign: `{report.campaign_id}`",
            f"- Tasks: {', '.join(f'`{task_id}`' for task_id in report.task_ids)}",
            f"- Repeats: {report.repeats}",
            "- All strategies use the same task catalog, runner, test oracle, and repeat count.",
            "",
            (
                "| Strategy | Pass rate | Pass@1 | Total cost | Cost / successful task | "
                "Total latency | Avg steps | Tool failures | Infra errors |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in report.strategies:
            lines.append(
                f"| {item.label} | {item.pass_rate:.1%} | {item.pass_at_1:.1%} | "
                f"{item.total_cost:.6f} | "
                f"{'-' if item.cost_per_success is None else f'{item.cost_per_success:.6f}'} | "
                f"{item.total_latency_seconds:.3f}s | {item.average_steps:.2f} | "
                f"{item.tool_failures}/{item.tool_requests} | {item.infrastructure_errors} |"
            )
        lines.extend(
            [
                "",
                "## Reproducibility",
                "",
                "The exact task fingerprints and strategy configurations are stored in "
                "`manifest.json`; each strategy keeps its full P0 report under `strategies/`.",
            ]
        )
        P2ProfilingRunner._atomic_write(root / "report.md", "\n".join(lines) + "\n")
