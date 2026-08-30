from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capycode.capability import (
    Capability,
    CapabilityDecision,
    CapabilityEvidence,
    Profile,
    ProfiledRoutingArtifact,
    ProfileMeasurement,
    ProfileRegistry,
    ProfileRouter,
)
from capycode.core import SessionState
from capycode.profiling import (
    EvaluationStrategy,
    GateCampaignReport,
    GateRunResult,
    GateTaskCatalog,
    P2ProfilingRunner,
    RoutingEvaluationRunner,
    TaskExecutor,
    measurements_from_report,
)
from capycode.trace import StepTraceEvent


def test_profiled_artifact_selects_reliable_lowest_expected_cost_candidate() -> None:
    artifact = ProfiledRoutingArtifact.from_measurements(
        [
            ProfileMeasurement(
                profile_id="cheap",
                capability=Capability.RETRIEVAL,
                model_id="model-cheap",
                succeeded=True,
                cost=1.0,
                latency_seconds=2.0,
            ),
            ProfileMeasurement(
                profile_id="cheap",
                capability=Capability.RETRIEVAL,
                model_id="model-cheap",
                succeeded=True,
                cost=1.0,
                latency_seconds=3.0,
            ),
            ProfileMeasurement(
                profile_id="reliable",
                capability=Capability.RETRIEVAL,
                model_id="model-reliable",
                succeeded=True,
                cost=2.0,
                latency_seconds=1.0,
            ),
            ProfileMeasurement(
                profile_id="reliable",
                capability=Capability.RETRIEVAL,
                model_id="model-reliable",
                succeeded=True,
                cost=2.0,
                latency_seconds=1.0,
            ),
        ],
        minimum_samples=2,
        reliability_threshold=0.8,
    )

    selected = artifact.selection_for(Capability.RETRIEVAL)

    assert selected is not None
    assert selected.profile_id == "cheap"
    assert selected.model_id == "model-cheap"
    assert selected.mean_cost == 1.0
    assert artifact.metrics[0].efficiency == 1.0


def test_profiled_router_uses_selected_measured_profile(tmp_path: Path) -> None:
    artifact = ProfiledRoutingArtifact.from_measurements(
        [
            ProfileMeasurement(
                profile_id="measured",
                capability=Capability.RETRIEVAL,
                model_id="measured-model",
                succeeded=True,
                cost=0.1,
                latency_seconds=1.0,
            ),
            ProfileMeasurement(
                profile_id="measured",
                capability=Capability.RETRIEVAL,
                model_id="measured-model",
                succeeded=True,
                cost=0.1,
                latency_seconds=1.0,
            ),
        ],
    )
    primary = Profile(
        "fallback",
        Capability.RETRIEVAL,
        "fallback-model",
        "retrieve",
        frozenset({"read_file"}),
        "retrieval",
        1,
        1,
    )
    measured = Profile(
        "measured",
        Capability.RETRIEVAL,
        "measured-model",
        "retrieve",
        frozenset({"read_file"}),
        "retrieval",
        10,
        1,
    )
    decision = CapabilityDecision(
        capability=Capability.RETRIEVAL,
        confidence=1,
        evidence=(CapabilityEvidence(signal="start", weight=1),),
    )

    route = ProfileRouter(
        ProfileRegistry({"fallback": primary, "measured": measured}),
        profiled_routing=artifact,
    ).select(decision, SessionState(workspace=str(tmp_path), task="inspect"))

    assert route.profile_id == "measured"
    assert "measured profile" in route.reason


def test_profiled_artifact_rejects_training_task_as_holdout() -> None:
    artifact = ProfiledRoutingArtifact.from_measurements([]).model_copy(
        update={"training_task_fingerprints": {"p0-01": "fingerprint"}}
    )

    with pytest.raises(ValueError, match="overlap"):
        artifact.validate_holdout({"p0-01": "fingerprint"})

    with pytest.raises(ValueError, match="same content"):
        artifact.validate_holdout({"renamed-task": "fingerprint"})

    artifact.validate_holdout({"p0-01": "fingerprint"}, allow_overlap=True)


def test_measurements_are_extracted_from_step_trace_and_terminal_result(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    event = StepTraceEvent(
        run_id="a" * 32,
        session_id="session",
        sequence=1,
        step=1,
        provider="fake",
        model_id="measured-model",
        latency_seconds=0.5,
        input_tokens=10,
        output_tokens=5,
        cost=0.25,
        currency="USD",
        capability=Capability.EDITING.value,
        profile_id="editing",
    )
    trace.write_text(event.model_dump_json() + "\n", encoding="utf-8")
    now = datetime.now(UTC)
    run = GateRunResult(
        task_id="p0-01",
        task_title="task",
        capability="editing",
        repeat_index=1,
        model_id="measured-model",
        task_fingerprint="x",
        status="passed",
        workspace=str(tmp_path),
        trace_path=str(trace),
    )
    report = GateCampaignReport(
        campaign_id="campaign",
        model_id="measured-model",
        started_at=now,
        finished_at=now,
        repeats=1,
        task_count=1,
        total_runs=1,
        passed_runs=1,
        pass_rate=1,
        pass_at_1=1,
        task_coverage=1,
        gate_eligible=False,
        gate_passed=False,
        total_input_tokens=10,
        total_output_tokens=5,
        total_cost=0.25,
        currency="USD",
        total_latency_seconds=0.5,
        average_steps=1,
        average_tokens=15,
        average_cost=0.25,
        average_latency_seconds=0.5,
        total_tool_requests=0,
        total_tool_failures=0,
        tool_failure_rate=0,
        results=[run],
    )

    measurements = measurements_from_report(report)

    assert measurements == [
        ProfileMeasurement(
            profile_id="editing",
            capability=Capability.EDITING,
            model_id="measured-model",
            succeeded=True,
            cost=0.25,
            latency_seconds=0.5,
        )
    ]


@pytest.mark.asyncio
async def test_profiling_runner_writes_auditable_empty_artifact(tmp_path: Path) -> None:
    async def executor(
        task: str,
        workspace: Path,
        model_id: str | None,
        max_steps: int,
    ) -> SessionState:
        return SessionState(
            workspace=str(workspace),
            task=task,
            status="completed",
            current_model=model_id or "fake",
        )

    runner = P2ProfilingRunner(output_root=tmp_path, catalog=GateTaskCatalog())
    report, artifact, root = await runner.run(
        lambda _model: executor,
        model_ids=["fake-model"],
        repeats=1,
        task_ids=["p0-01"],
    )

    assert report.measurements == 0
    assert not artifact.selected_by_capability
    assert artifact.training_task_fingerprints.keys() == {"p0-01"}
    assert artifact.source_campaign_id == report.campaign_id
    assert artifact.candidate_model_ids == ("fake-model",)
    assert (root / "profiles.json").is_file()
    assert (root / "profiles.partial.json").is_file()
    assert (root / "measurements.jsonl").is_file()
    assert (root / "leaderboard.csv").is_file()
    assert (root / "leaderboard.md").is_file()
    assert (root / "report.md").is_file()
    assert '"campaign_type": "profiling"' in (root / "manifest.json").read_text(encoding="utf-8")
    assert '"status": "completed"' in (root / "progress.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_routing_evaluation_writes_reproducible_comparison(tmp_path: Path) -> None:
    async def executor(
        task: str,
        workspace: Path,
        model_id: str | None,
        max_steps: int,
    ) -> SessionState:
        return SessionState(
            workspace=str(workspace),
            task=task,
            status="completed",
            current_model=model_id or "fake",
        )

    runner = RoutingEvaluationRunner(output_root=tmp_path, catalog=GateTaskCatalog())
    report, root = await runner.run(
        [
            EvaluationStrategy(
                strategy_id="fixed-a",
                label="Fixed A",
                configuration={"model_id": "model-a"},
            ),
            EvaluationStrategy(
                strategy_id="profiled",
                label="Profiled",
                configuration={"artifact": "profiles.json"},
            ),
        ],
        lambda _strategy: executor,
        repeats=1,
        task_ids=["p0-01"],
    )

    assert report.task_ids == ["p0-01"]
    assert report.task_fingerprints.keys() == {"p0-01"}
    assert len(report.strategies) == 2
    assert all(item.total_runs == 1 for item in report.strategies)
    assert all(item.infrastructure_errors == 0 for item in report.strategies)
    assert (root / "report.json").is_file()
    assert (root / "report.partial.json").is_file()
    assert (root / "report.md").is_file()
    comparison = (root / "comparison.csv").read_text(encoding="utf-8")
    assert "cached_input_tokens" in comparison
    assert "infrastructure_errors" in comparison
    manifest = (root / "manifest.json").read_text(encoding="utf-8")
    assert '"campaign_type": "evaluation"' in manifest
    assert '"model_id": "model-a"' in manifest
    assert '"status": "completed"' in (root / "progress.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_routing_evaluation_rejects_invalid_strategy_sets(tmp_path: Path) -> None:
    runner = RoutingEvaluationRunner(output_root=tmp_path, catalog=GateTaskCatalog())

    with pytest.raises(ValueError, match="at least two"):
        await runner.run(
            [EvaluationStrategy(strategy_id="only", label="Only")],
            lambda _strategy: pytest.fail("executor must not be requested"),
            task_ids=["p0-01"],
        )

    with pytest.raises(ValueError, match="unique"):
        await runner.run(
            [
                EvaluationStrategy(strategy_id="duplicate", label="A"),
                EvaluationStrategy(strategy_id="duplicate", label="B"),
            ],
            lambda _strategy: pytest.fail("executor must not be requested"),
            task_ids=["p0-01"],
        )


@pytest.mark.asyncio
async def test_profiling_runner_records_failure_progress(tmp_path: Path) -> None:
    def failing_factory(_model_id: str) -> TaskExecutor:
        raise RuntimeError("synthetic executor failure")

    runner = P2ProfilingRunner(output_root=tmp_path, catalog=GateTaskCatalog())

    with pytest.raises(RuntimeError, match="synthetic executor failure"):
        await runner.run(
            failing_factory,
            model_ids=["fake-model"],
            task_ids=["p0-01"],
        )

    campaign_roots = await asyncio.to_thread(
        lambda: [path for path in tmp_path.iterdir() if path.is_dir()]
    )
    assert len(campaign_roots) == 1
    progress = (campaign_roots[0] / "progress.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in progress
    assert "synthetic executor failure" in progress
