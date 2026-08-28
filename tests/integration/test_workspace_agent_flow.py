from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from capycode.core import AgentRuntime
from capycode.llm import LLMResponse, ScriptedLLM, ToolCall
from capycode.tools import build_p0_runtime_tools


def initialize_defect_repository(directory: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
    (directory / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n",
        encoding="utf-8",
    )
    (directory / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=directory, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CapyCode Test",
            "-c",
            "user.email=capycode@example.invalid",
            "commit",
            "-qm",
            "defect fixture",
        ],
        cwd=directory,
        check=True,
    )


@pytest.mark.asyncio
async def test_fake_agent_fails_tests_fixes_code_retests_and_inspects_diff(tmp_path: Path) -> None:
    initialize_defect_repository(tmp_path)
    test_argv = ["python", "-m", "pytest", "-q"]
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
                        id="read-source", name="read_file", arguments={"path": "calculator.py"}
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
                    ToolCall(id="final-diff", name="git_diff", arguments={}),
                ]
            ),
            LLMResponse(content="Fixed add(), verified the test, and reviewed the final diff."),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=8)

    state = await runtime.run("Fix the failing addition test", tmp_path, "fake-model")

    assert state.status == "completed"
    assert state.termination_reason == "completed"
    assert state.modified_files == ["calculator.py"]
    assert state.last_tests_passed is True
    assert state.last_exit_code == 0
    assert "+    return left + right" in state.current_diff
    assert (
        (tmp_path / "calculator.py").read_text(encoding="utf-8").endswith("return left + right\n")
    )

    tool_results = [
        json.loads(message.content or "{}")
        for message in client.requests[-1].messages
        if message.role == "tool"
    ]
    test_results = [result for result in tool_results if "tests_passed" in result.get("data", {})]
    assert [result["data"]["tests_passed"] for result in test_results] == [False, True]


@pytest.mark.asyncio
async def test_agent_pairs_tool_results_when_per_step_limit_is_exceeded(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("content", encoding="utf-8")
    calls = [
        ToolCall(id=f"call-{index}", name="read_file", arguments={"path": "file.txt"})
        for index in range(9)
    ]
    runtime = AgentRuntime(
        ScriptedLLM([LLMResponse(tool_calls=calls)]),
        build_p0_runtime_tools(),
        max_steps=2,
        max_tool_calls_per_step=8,
    )

    state = await runtime.run("Read too many times", tmp_path, "fake-model")

    assert state.status == "failed"
    assert state.termination_reason == "tool_call_limit_exceeded"
    tool_messages = [message for message in state.history if message.role == "tool"]
    assert len(tool_messages) == 9
    assert {message.tool_call_id for message in tool_messages} == {
        f"call-{index}" for index in range(9)
    }
    skipped = json.loads(tool_messages[-1].content or "{}")
    assert skipped["status"] == "error"
    assert "skipped" in skipped["content"]


@pytest.mark.asyncio
async def test_agent_stops_after_three_identical_tool_steps(tmp_path: Path) -> None:
    (tmp_path / "loop.txt").write_text("loop", encoding="utf-8")
    responses = [
        LLMResponse(
            tool_calls=[
                ToolCall(id=f"call-{index}", name="read_file", arguments={"path": "loop.txt"})
            ]
        )
        for index in range(3)
    ]
    runtime = AgentRuntime(ScriptedLLM(responses), build_p0_runtime_tools(), max_steps=5)

    state = await runtime.run("Loop forever", tmp_path, "fake-model")

    assert state.status == "failed"
    assert state.termination_reason == "loop_detected"
    assert len([message for message in state.history if message.role == "tool"]) == 3
