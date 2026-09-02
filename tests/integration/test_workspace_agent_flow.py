from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from capycode.core import AgentRuntime, SessionState
from capycode.llm import (
    LLMError,
    LLMErrorKind,
    LLMRequest,
    LLMResponse,
    Message,
    ScriptedLLM,
    ToolCall,
)
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
async def test_agent_stops_after_bounded_identical_tool_recoveries(tmp_path: Path) -> None:
    (tmp_path / "loop.txt").write_text("loop", encoding="utf-8")
    responses = [
        LLMResponse(
            tool_calls=[
                ToolCall(id=f"call-{index}", name="read_file", arguments={"path": "loop.txt"})
            ]
        )
        for index in range(9)
    ]
    runtime = AgentRuntime(ScriptedLLM(responses), build_p0_runtime_tools(), max_steps=10)

    state = await runtime.run("Loop forever", tmp_path, "fake-model")

    assert state.status == "failed"
    assert state.termination_reason == "loop_detected"
    assert len([message for message in state.history if message.role == "tool"]) == 9
    recovery_messages = [
        message
        for message in state.history
        if message.role == "user" and message.content and "Loop recovery" in message.content
    ]
    assert len(recovery_messages) == 2


@pytest.mark.asyncio
async def test_agent_can_recover_from_repeated_tool_steps(tmp_path: Path) -> None:
    (tmp_path / "loop.txt").write_text("loop", encoding="utf-8")
    repeated = [
        LLMResponse(
            tool_calls=[
                ToolCall(id=f"call-{index}", name="read_file", arguments={"path": "loop.txt"})
            ]
        )
        for index in range(3)
    ]
    client = ScriptedLLM([*repeated, LLMResponse(content="Changed approach and finished.")])
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=5)

    state = await runtime.run("Recover from a loop", tmp_path, "fake-model")

    assert state.status == "completed"
    assert state.final_answer == "Changed approach and finished."


@pytest.mark.asyncio
async def test_agent_compacts_and_retries_explicit_context_limit_error(tmp_path: Path) -> None:
    class ContextLimitOnceLLM:
        def __init__(self) -> None:
            self.requests: list[LLMRequest] = []

        async def stream(self, request: LLMRequest, on_text_delta) -> LLMResponse:
            self.requests.append(request.model_copy(deep=True))
            if len(self.requests) == 1:
                raise LLMError(
                    LLMErrorKind.BAD_REQUEST,
                    "model endpoint returned HTTP 400: maximum context length exceeded",
                    retryable=False,
                    status_code=400,
                )
            return LLMResponse(content="Completed after compaction.")

    client = ContextLimitOnceLLM()
    state = SessionState(
        workspace=str(tmp_path),
        task="Fix the bug",
        history=[
            Message(role="system", content="system"),
            Message(role="user", content="Fix the bug"),
            Message(role="assistant", content="old diagnostic output " * 2_500),
            Message(role="user", content="Continue"),
        ],
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=3)

    result = await runtime.run("Fix the bug", tmp_path, "fake-model", state=state)

    assert result.status == "completed"
    assert result.retry_count == 1
    assert len(client.requests) == 2
    first_characters = sum(len(message.content or "") for message in client.requests[0].messages)
    second_characters = sum(len(message.content or "") for message in client.requests[1].messages)
    assert second_characters < first_characters


@pytest.mark.asyncio
async def test_agent_skips_duplicate_tool_calls_within_one_step(tmp_path: Path) -> None:
    arguments = {"path": "index.html", "content": "<main>demo</main>\n"}
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="write-1", name="write_file", arguments=arguments),
                    ToolCall(id="write-2", name="write_file", arguments=arguments),
                ]
            ),
            LLMResponse(content="Created the page once."),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=3)

    state = await runtime.run("Create a page", tmp_path, "fake-model")

    assert state.status == "completed"
    tool_messages = [message for message in state.history if message.role == "tool"]
    assert len(tool_messages) == 2
    duplicate = json.loads(tool_messages[1].content or "{}")
    assert duplicate["data"]["duplicate_skipped"] is True


@pytest.mark.asyncio
async def test_non_git_single_file_web_workspace_finishes_without_retries(tmp_path: Path) -> None:
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="write-page",
                        name="write_file",
                        arguments={"path": "index.html", "content": "<main>Snake</main>\n"},
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="search-page",
                        name="search_code",
                        arguments={"query": "Snake", "path": "index.html"},
                    ),
                    ToolCall(id="diff-page", name="git_diff", arguments={}),
                ]
            ),
            LLMResponse(content="Created and checked the standalone page."),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=4)

    state = await runtime.run("Create a standalone web page", tmp_path, "fake-model")

    assert state.status == "completed"
    assert state.modified_files == ["index.html"]
    assert "not a Git repository" in state.current_diff
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<main>Snake</main>\n"


@pytest.mark.asyncio
async def test_reasoning_only_response_is_continued(tmp_path: Path) -> None:
    client = ScriptedLLM(
        [
            LLMResponse(reasoning_content="I need to inspect the repository first."),
            LLMResponse(content="Task completed."),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=3)

    state = await runtime.run("Explain the workspace", tmp_path, "fake-model")

    assert state.status == "completed"
    assert state.final_answer == "Task completed."
    assert len(client.requests) == 2
    assert client.requests[1].messages[-1].content == (
        "Continue the coding task. Use the available tools now; do not return reasoning only."
    )
