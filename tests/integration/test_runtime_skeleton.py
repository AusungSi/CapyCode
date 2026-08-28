from __future__ import annotations

import json
from pathlib import Path

import pytest

from capycode.core import AgentRuntime, SessionState
from capycode.llm import LLMResponse, ScriptedLLM, ToolCall, Usage
from capycode.tools import build_p0_runtime_tools


@pytest.mark.asyncio
async def test_model_reads_file_then_completes_task(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nRuntime fixture.\n", encoding="utf-8")
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})
                ],
                finish_reason="tool_calls",
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            LLMResponse(
                content="The repository README describes a runtime fixture.",
                finish_reason="stop",
                usage=Usage(input_tokens=20, output_tokens=8),
            ),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=4)

    state = await runtime.run("Summarize README.md", tmp_path, "fake-model")

    assert state.status == "completed"
    assert state.step == 2
    assert state.final_answer == "The repository README describes a runtime fixture."
    assert state.relevant_files == ["README.md"]
    assert state.total_input_tokens == 30
    assert state.total_output_tokens == 13
    assert len(client.requests) == 2
    tool_message = client.requests[1].messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call-1"
    result = json.loads(tool_message.content or "{}")
    assert result["status"] == "success"
    assert "Runtime fixture" in result["content"]


@pytest.mark.asyncio
async def test_tool_error_is_returned_to_model(tmp_path: Path) -> None:
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="read_file", arguments={"path": "../secret.txt"})
                ]
            ),
            LLMResponse(content="The requested path is outside the workspace."),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=3)

    state = await runtime.run("Read an invalid path", tmp_path, "fake-model")

    assert state.status == "completed"
    assert state.last_error is not None
    result = json.loads(client.requests[1].messages[-1].content or "{}")
    assert result["status"] == "error"
    assert "workspace" in result["content"]


@pytest.mark.asyncio
async def test_runtime_executes_multiple_tool_calls_in_order(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="call-a", name="read_file", arguments={"path": "a.txt"}),
                    ToolCall(id="call-b", name="read_file", arguments={"path": "b.txt"}),
                ]
            ),
            LLMResponse(content="Read both files."),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=3)

    state = await runtime.run("Read both files", tmp_path, "fake-model")

    assert state.status == "completed"
    assert state.relevant_files == ["a.txt", "b.txt"]
    tool_messages = client.requests[1].messages[-2:]
    assert [message.tool_call_id for message in tool_messages] == ["call-a", "call-b"]


@pytest.mark.asyncio
async def test_runtime_stops_at_step_limit(tmp_path: Path) -> None:
    (tmp_path / "loop.txt").write_text("loop", encoding="utf-8")
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="call-1", name="read_file", arguments={"path": "loop.txt"})]
            )
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=1)

    state = await runtime.run("Keep reading", tmp_path, "fake-model")

    assert state.status == "failed"
    assert state.step == 1
    assert state.last_error == "maximum step limit reached: 1"


@pytest.mark.asyncio
async def test_runtime_continues_existing_conversation(tmp_path: Path) -> None:
    client = ScriptedLLM(
        [
            LLMResponse(content="First answer"),
            LLMResponse(content="Second answer"),
        ]
    )
    runtime = AgentRuntime(client, build_p0_runtime_tools(), max_steps=2)
    checkpoints: list[SessionState] = []

    state = await runtime.run(
        "First question",
        tmp_path,
        "fake-model",
        checkpoint=lambda value: checkpoints.append(value.model_copy(deep=True)),
    )
    session_id = state.session_id
    state = await runtime.run("Follow-up question", tmp_path, "fake-model", state=state)

    assert state.session_id == session_id
    assert [message.role for message in client.requests[1].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert client.requests[1].messages[-1].content == "Follow-up question"
    assert checkpoints[-1].status == "completed"
