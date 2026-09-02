from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from capycode.app import cli
from capycode.app.cli import build_parser, inspect_run, main, run_doctor, show_runs, show_welcome
from capycode.trace import RunSummary


def test_no_arguments_starts_branded_entrypoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = show_welcome(tmp_path)

    assert code == 0
    output = capsys.readouterr().out
    assert "CapyCode 0.1.0" in output
    assert f"workspace: {tmp_path.resolve()}" in output
    assert "stage: P0 baseline gate" in output


def test_main_without_arguments_exits_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = False

    def fake_launch_tui() -> None:
        nonlocal launched
        launched = True

    monkeypatch.setattr(cli, "launch_tui", fake_launch_tui)

    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 0
    assert launched is True


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_benchmark_validate_only_does_not_require_model_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["benchmark", "p0", "--validate-only"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "validated 5 P0 fixture(s)" in output
    assert "p0-05" in output


def test_swebench_command_accepts_profiled_routing_options() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "swebench",
            "--instances",
            "tasks.jsonl",
            "--profiles",
            "profiles.yaml",
            "--profiled-artifact",
            "routing.json",
        ]
    )

    assert args.profiles == Path("profiles.yaml")
    assert args.profiled_artifact == Path("routing.json")


def test_p2_commands_are_exposed_in_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "profile" in output
    assert "evaluate" in output


def test_p2_profile_command_parses_reusable_campaign_options() -> None:
    args = build_parser().parse_args(
        [
            "profile",
            "p0",
            "--model",
            "model-a",
            "--model",
            "model-b",
            "--task",
            "p0-01",
            "--repeats",
            "2",
            "--minimum-samples",
            "3",
            "--reliability-threshold",
            "0.75",
            "--quality-tolerance",
            "0.1",
            "--install",
        ]
    )

    assert args.command == "profile"
    assert args.profile_command == "p0"
    assert args.models == ["model-a", "model-b"]
    assert args.tasks == ["p0-01"]
    assert args.repeats == 2
    assert args.minimum_samples == 3
    assert args.reliability_threshold == 0.75
    assert args.quality_tolerance == 0.1
    assert args.install is True


def test_p2_evaluation_command_parses_configurable_strategies() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "p0",
            "--fixed-model",
            "model-a",
            "--fixed-model",
            "model-b",
            "--profiled-artifact",
            "profiles.json",
            "--task",
            "p0-05",
        ]
    )

    assert args.command == "evaluate"
    assert args.evaluation_command == "p0"
    assert args.fixed_models == ["model-a", "model-b"]
    assert args.profiled_artifact == Path("profiles.json")
    assert args.tasks == ["p0-05"]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [(["--continue"], "latest"), (["--resume", "abc123"], "abc123")],
)
def test_resume_flags_launch_tui_with_requested_session(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected: str,
) -> None:
    received: list[str | None] = []

    def fake_launch_tui(*, initial_resume: str | None = None) -> None:
        received.append(initial_resume)

    monkeypatch.setattr(cli, "launch_tui", fake_launch_tui)

    with pytest.raises(SystemExit) as exc_info:
        main(arguments)

    assert exc_info.value.code == 0
    assert received == [expected]


def test_doctor_accepts_valid_examples_without_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_doctor(
        models_path=Path("config/models.example.yaml"),
        profiles_path=Path("config/profiles.example.yaml"),
        strict_secrets=False,
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "configuration: valid" in output
    assert "warning: missing local environment variables" in output


def test_doctor_strict_mode_rejects_missing_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "CAPYCODE_BASE_URL",
        "CAPYCODE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    code = run_doctor(
        models_path=Path("config/models.example.yaml"),
        profiles_path=Path("config/profiles.example.yaml"),
        strict_secrets=True,
    )

    assert code == 1


def test_runs_and_inspect_run_read_local_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "a" * 32
    run_directory = tmp_path / ".capy" / "runs" / run_id
    run_directory.mkdir(parents=True)
    now = datetime.now(UTC)
    summary = RunSummary(
        run_id=run_id,
        session_id="session-one",
        workspace=str(tmp_path),
        task="test",
        provider="fake",
        model_id="fake-model",
        status="completed",
        termination_reason="completed",
        started_at=now,
        finished_at=now,
        latency_seconds=1.25,
        steps=2,
        input_tokens=100,
        output_tokens=20,
        cost=0.001,
        currency="CNY",
        retry_count=0,
        tool_requests=1,
        tool_successes=1,
        tool_failures=0,
        tests_passed=True,
        modified_files=["demo.py"],
        trace_path=str(run_directory / "trace.jsonl"),
        pricing_snapshot_date="2026-08-28",
        pricing_configured=True,
    )
    (run_directory / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    assert show_runs(tmp_path, limit=10) == 0
    list_output = capsys.readouterr().out
    assert run_id[:8] in list_output
    assert "fake-model" in list_output

    assert inspect_run(tmp_path, run_id[:8]) == 0
    inspect_output = capsys.readouterr().out
    assert f"run_id: {run_id}" in inspect_output
    assert "tokens: 100 input, 0 cached input, 20 output" in inspect_output
    assert "modified_files: demo.py" in inspect_output
