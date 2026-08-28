from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capycode.trace import (
    REDACTED,
    RUN_EVENT_ADAPTER,
    RunSummary,
    RunTracker,
    RunTrackingConfig,
    ToolRequestEvent,
    ToolResultEvent,
    TraceRecorder,
    TraceRedactor,
    UserTaskEvent,
)


def test_trace_recorder_appends_valid_jsonl_and_flushes(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path, "run-one", "session-one")
    recorder.append(
        UserTaskEvent(
            run_id="run-one",
            session_id="session-one",
            sequence=1,
            task="Inspect the project",
        )
    )

    content = recorder.trace_path.read_text(encoding="utf-8")
    payload = json.loads(content)
    event = RUN_EVENT_ADAPTER.validate_python(payload)

    assert event.event_type == "user_task"
    assert event.sequence == 1
    recorder.close()


def test_trace_recorder_enforces_sequence_and_tool_pairing(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path, "run-two", "session-two")
    request = ToolRequestEvent(
        run_id="run-two",
        session_id="session-two",
        sequence=1,
        step=1,
        tool_call_id="call-one",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )
    recorder.append(request)

    with pytest.raises(ValueError, match="sequence"):
        recorder.append(request.model_copy(update={"sequence": 3, "tool_call_id": "call-two"}))
    with pytest.raises(ValueError, match="no request"):
        recorder.append(
            ToolResultEvent(
                run_id="run-two",
                session_id="session-two",
                sequence=2,
                step=1,
                tool_call_id="missing",
                tool_name="read_file",
                status="error",
                content="missing",
                latency_seconds=0,
            )
        )

    recorder.append(
        ToolResultEvent(
            run_id="run-two",
            session_id="session-two",
            sequence=2,
            step=1,
            tool_call_id="call-one",
            tool_name="read_file",
            status="success",
            content="contents",
            latency_seconds=0.1,
        )
    )
    assert recorder.pending_tool_call_ids == frozenset()
    recorder.close()


def test_trace_recorder_redacts_sensitive_values(tmp_path: Path) -> None:
    recorder = TraceRecorder(
        tmp_path,
        "run-three",
        "session-three",
        redactor=TraceRedactor(["local-secret"]),
    )
    recorder.append(
        ToolRequestEvent(
            run_id="run-three",
            session_id="session-three",
            sequence=1,
            step=1,
            tool_call_id="call-one",
            tool_name="run_command",
            arguments={
                "api_key": "local-secret",
                "message": "Authorization: Bearer local-secret",
                "input_tokens": 12,
            },
        )
    )
    recorder.close()

    content = recorder.trace_path.read_text(encoding="utf-8")
    payload = json.loads(content)
    assert "local-secret" not in content
    assert payload["arguments"]["api_key"] == REDACTED
    assert REDACTED in payload["arguments"]["message"]
    assert payload["arguments"]["input_tokens"] == 12


def test_trace_recorder_writes_atomic_summary(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path, "run-four", "session-four")
    now = datetime.now(UTC)
    summary = RunSummary(
        run_id="run-four",
        session_id="session-four",
        workspace=str(tmp_path),
        task="test",
        provider="fake",
        model_id="fake-model",
        status="completed",
        termination_reason="completed",
        started_at=now,
        finished_at=now,
        latency_seconds=0,
        steps=1,
        input_tokens=10,
        output_tokens=2,
        cost=0,
        currency="USD",
        retry_count=0,
        tool_requests=0,
        tool_successes=0,
        tool_failures=0,
        trace_path=str(recorder.trace_path),
        pricing_snapshot_date="2026-08-28",
        pricing_configured=False,
    )

    recorder.write_summary(summary)
    recorder.close()

    assert json.loads(recorder.summary_path.read_text(encoding="utf-8"))["run_id"] == "run-four"
    assert list(recorder.run_directory.glob("*.tmp")) == []


def test_run_tracker_synthesizes_result_for_interrupted_tool(tmp_path: Path) -> None:
    tracker = RunTracker(
        tmp_path,
        "session-five",
        RunTrackingConfig(
            provider="fake",
            model_id="fake-model",
            input_per_million=0,
            output_per_million=0,
            currency="USD",
            pricing_snapshot_date="2026-08-28",
        ),
        run_id="run-five",
    )
    tracker.start("task")
    tracker.record_tool_request(1, "call-one", "read_file", {"path": "README.md"})

    tracker.synthesize_pending_results("cancelled")
    tracker.recorder.close()

    events = [
        json.loads(line) for line in tracker.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    result = next(event for event in events if event["event_type"] == "tool_result")
    assert result["tool_call_id"] == "call-one"
    assert result["status"] == "error"
    assert result["data"]["synthetic"] is True
    assert tracker.recorder.pending_tool_call_ids == frozenset()


def test_run_tracker_emits_the_redacted_persisted_event(tmp_path: Path) -> None:
    received: list[object] = []
    tracker = RunTracker(
        tmp_path,
        "session-six",
        RunTrackingConfig(
            provider="fake",
            model_id="fake-model",
            input_per_million=0,
            output_per_million=0,
            currency="USD",
            pricing_snapshot_date="2026-08-28",
            sensitive_values=("local-secret",),
            event_sink=received.append,
        ),
        run_id="run-six",
    )

    tracker.start("inspect local-secret")
    tracker.record_tool_request(
        1,
        "call-one",
        "run_command",
        {"argv": ["python", "script.py"], "token": "local-secret"},
    )
    tracker.synthesize_pending_results("cancelled")
    tracker.recorder.close()

    serialized = "\n".join(str(event) for event in received)
    assert "local-secret" not in serialized
    assert REDACTED in serialized
    assert len(received) == 4
