from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import anyio
import pytest

from capycode.core import AgentRuntime
from capycode.llm import LLMRequest, LLMResponse, ScriptedLLM, ToolCall, Usage
from capycode.tools import build_p0_runtime_tools
from capycode.trace import RUN_EVENT_ADAPTER, RunCatalog, RunTrackingConfig


def tracking_config() -> RunTrackingConfig:
    return RunTrackingConfig(
        provider="fake",
        model_id="fake-model",
        input_per_million=2.0,
        output_per_million=4.0,
        currency="CNY",
        pricing_snapshot_date="2026-08-28",
        sensitive_values=("local-secret",),
    )


@pytest.mark.asyncio
async def test_coding_run_writes_paired_trace_and_summary(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("secret-free contents\n", encoding="utf-8")
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="read-one", name="read_file", arguments={"path": "README.md"})
                ],
                finish_reason="tool_calls",
                usage=Usage(input_tokens=100, output_tokens=20),
            ),
            LLMResponse(
                content="Done",
                finish_reason="stop",
                usage=Usage(input_tokens=200, output_tokens=30),
            ),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=3)

    state = await runtime.run(
        "Read the project",
        tmp_path,
        "fake-model",
        tracking=tracking_config(),
    )

    summary = RunCatalog(tmp_path).resolve("latest")
    trace_lines = (await anyio.Path(summary.trace_path).read_text(encoding="utf-8")).splitlines()
    events = [RUN_EVENT_ADAPTER.validate_python(json.loads(line)) for line in trace_lines]
    requests = [event for event in events if event.event_type == "tool_request"]
    results = [event for event in events if event.event_type == "tool_result"]

    assert state.status == "completed"
    assert state.current_run_id == summary.run_id
    assert summary.input_tokens == 300
    assert summary.output_tokens == 50
    assert summary.cost == pytest.approx(0.0008)
    assert summary.tool_requests == 1
    assert summary.tool_successes == 1
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [event.tool_call_id for event in requests] == [event.tool_call_id for event in results]
    assert events[-1].event_type == "run_status"


@pytest.mark.asyncio
async def test_runtime_exception_still_writes_failed_summary(tmp_path: Path) -> None:
    runtime = AgentRuntime(ScriptedLLM([]), build_p0_runtime_tools(), max_steps=2)

    with pytest.raises(RuntimeError, match="no response"):
        await runtime.run(
            "Fail predictably",
            tmp_path,
            "fake-model",
            tracking=tracking_config(),
        )

    summary = RunCatalog(tmp_path).resolve("latest")
    assert summary.status == "failed"
    assert summary.termination_reason == "runtime_error"
    assert "no response" in (summary.error or "")


@pytest.mark.asyncio
async def test_resumed_session_creates_a_new_run_id(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        ScriptedLLM([LLMResponse(content="one"), LLMResponse(content="two")]),
        build_p0_runtime_tools(),
        max_steps=2,
    )
    state = await runtime.run(
        "First task",
        tmp_path,
        "fake-model",
        tracking=tracking_config(),
    )
    session_id = state.session_id
    first_run_id = state.current_run_id

    state = await runtime.run(
        "Follow-up task",
        tmp_path,
        "fake-model",
        state=state,
        tracking=tracking_config(),
    )
    summaries = RunCatalog(tmp_path).list()

    assert state.session_id == session_id
    assert state.current_run_id != first_run_id
    assert len(summaries) == 2
    assert {summary.session_id for summary in summaries} == {session_id}
    assert len({summary.run_id for summary in summaries}) == 2


class BlockingLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, request: LLMRequest, on_text_delta: object) -> LLMResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_cancelled_run_writes_cancelled_summary(tmp_path: Path) -> None:
    client = BlockingLLM()
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=2)  # type: ignore[arg-type]
    task = asyncio.create_task(
        runtime.run("Wait", tmp_path, "fake-model", tracking=tracking_config())
    )
    await client.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    summary = RunCatalog(tmp_path).resolve("latest")
    assert summary.status == "cancelled"
    assert summary.termination_reason == "cancelled"


@pytest.mark.asyncio
async def test_bug_fix_trace_reports_failed_then_passing_tests(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n",
        encoding="utf-8",
    )
    (tmp_path / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    await anyio.run_process(["git", "init", "-q"], cwd=tmp_path, check=True)
    await anyio.run_process(["git", "add", "."], cwd=tmp_path, check=True)
    await anyio.run_process(
        [
            "git",
            "-c",
            "user.name=CapyCode Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    test_argv = [Path(sys.executable).name, "-m", "pytest", "-q"]
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="tests-before", name="run_tests", arguments={"argv": test_argv})
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="read-source",
                        name="read_file",
                        arguments={"path": "calculator.py"},
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="fix-source",
                        name="replace_text",
                        arguments={
                            "path": "calculator.py",
                            "old_text": "return left - right",
                            "new_text": "return left + right",
                        },
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(id="tests-after", name="run_tests", arguments={"argv": test_argv}),
                    ToolCall(id="diff", name="git_diff", arguments={}),
                ]
            ),
            LLMResponse(content="Fixed and verified."),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=6)

    state = await runtime.run(
        "Fix the failing test",
        tmp_path,
        "fake-model",
        tracking=tracking_config(),
    )

    summary = RunCatalog(tmp_path).resolve("latest")
    assert state.status == "completed"
    assert summary.tests_passed is True
    assert summary.modified_files == ["calculator.py"]
    assert summary.tool_requests == 5
    assert summary.tool_failures == 1
    assert summary.tool_successes == 4
    assert "return left + right" in summary.final_diff
