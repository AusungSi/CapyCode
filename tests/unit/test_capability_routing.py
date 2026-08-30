from __future__ import annotations

from pathlib import Path

import pytest

from capycode.app.runtime import load_profile_instruction
from capycode.capability import (
    Capability,
    CapabilityDetector,
    ContextBuilder,
    EscalationPolicy,
    Profile,
    ProfileRegistry,
    ProfileRouter,
    build_default_profile_registry,
)
from capycode.config import load_profiles
from capycode.core import AgentRuntime, SessionState
from capycode.llm import LLMResponse, Message, ScriptedLLM, ToolCall
from capycode.tools import build_p0_runtime_tools


def registry() -> ProfileRegistry:
    return ProfileRegistry.from_config(load_profiles("config/profiles.example.yaml"))


def test_profile_registry_resolves_configured_model_refs() -> None:
    config = load_profiles("config/profiles.example.yaml")
    resolved = ProfileRegistry.from_config(
        config,
        model_resolver=lambda model_ref: {
            "model-a": "provider/retrieval-model",
            "model-b": "provider/editing-model",
            "model-c": "provider/diagnosis-model",
        }[model_ref],
        instruction_resolver=lambda instruction: f"resolved: {instruction}",
    )
    assert resolved.get("retrieval_fast").model_ref == "provider/retrieval-model"
    assert resolved.get("diagnosis_deep").model_ref == "provider/diagnosis-model"
    assert resolved.get("retrieval_fast").instruction == "resolved: prompts/retrieval.md"


def test_default_profile_registry_keys_match_profile_ids() -> None:
    registry = build_default_profile_registry()
    for profile in registry.all():
        assert registry.get(profile.profile_id) is profile


def test_example_profile_instruction_paths_load_prompt_content() -> None:
    profile_path = Path("config/profiles.example.yaml")

    instruction = load_profile_instruction("prompts/verification.md", profile_path)

    assert "Validate the changed behavior" in instruction


def test_detector_uses_explainable_state_signals(tmp_path: Path) -> None:
    detector = CapabilityDetector()
    state = SessionState(workspace=str(tmp_path), task="Fix the bug")
    assert detector.detect(state).capability == Capability.RETRIEVAL

    state.relevant_files.append("module.py")
    understanding = detector.detect(state)
    assert understanding.capability == Capability.UNDERSTANDING

    state.last_tests_passed = False
    assert detector.detect(state).capability == Capability.DIAGNOSIS


def test_detector_enters_editing_after_context_for_coding_task(tmp_path: Path) -> None:
    detector = CapabilityDetector()
    state = SessionState(
        workspace=str(tmp_path),
        task="Fix the parser bug in the repository",
        current_capability=Capability.UNDERSTANDING.value,
        relevant_files=["parser.py"],
    )

    assert detector.detect(state).capability == Capability.EDITING


def test_router_rejects_missing_capability_profile(tmp_path: Path) -> None:
    state = SessionState(workspace=str(tmp_path), task="Fix the bug")
    decision = CapabilityDetector().detect(state)
    route = ProfileRouter(registry()).select(decision, state)
    assert route.capability == Capability.RETRIEVAL
    assert route.profile_id == "retrieval_fast"

    state.last_tests_passed = False
    retrieval = registry().get("retrieval_fast")
    assert retrieval is not None
    with pytest.raises(ValueError, match="no profile configured"):
        ProfileRouter(ProfileRegistry({"retrieval_fast": retrieval})).select(
            CapabilityDetector().detect(state), state
        )


def test_context_builder_filters_tools_and_history(tmp_path: Path) -> None:
    state = SessionState(
        workspace=str(tmp_path),
        task="Find the repository entry point",
        history=[],
    )
    profile = registry().get("retrieval_fast")
    assert profile is not None
    tools = build_p0_runtime_tools().definitions()
    messages, selected = ContextBuilder().build(state, profile, tools)
    assert messages[0].role == "system"
    assert {tool.name for tool in selected} == {"list_files", "read_file", "search_code"}


def test_context_builder_preserves_tool_pairs_when_compacting() -> None:
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="Fix the bug"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="old-call", name="read_file", arguments={"path": "old"})],
        ),
        Message(
            role="tool",
            name="read_file",
            tool_call_id="old-call",
            content="x" * 5_000,
        ),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="recent-call", name="read_file", arguments={"path": "recent"})],
        ),
        Message(
            role="tool",
            name="read_file",
            tool_call_id="recent-call",
            content="y" * 2_000,
        ),
    ]

    compacted = ContextBuilder.fit_to_budget(messages, [], max_input_tokens=1_000)

    ids = {message.tool_call_id for message in compacted if message.role == "tool"}
    assistant_ids = {
        call.id
        for message in compacted
        if message.role == "assistant"
        for call in message.tool_calls
    }
    assert ids == {"recent-call"}
    assert assistant_ids == ids
    assert any(message.role == "user" and message.content == "Fix the bug" for message in compacted)


def test_context_builder_profile_window_never_starts_with_orphaned_tool_result(
    tmp_path: Path,
) -> None:
    state = SessionState(
        workspace=str(tmp_path),
        task="Fix the bug",
        history=[
            Message(role="system", content="system"),
            Message(role="user", content="Fix the bug"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="old-a", name="read_file", arguments={"path": "a.py"}),
                    ToolCall(id="old-b", name="read_file", arguments={"path": "b.py"}),
                ],
            ),
            Message(role="tool", tool_call_id="old-a", name="read_file", content="a"),
            Message(role="tool", tool_call_id="old-b", name="read_file", content="b"),
            Message(role="assistant", content="inspection complete"),
            Message(role="user", content="continue"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="recent", name="git_diff", arguments={})],
            ),
            Message(role="tool", tool_call_id="recent", name="git_diff", content="diff"),
        ],
    )
    profile = registry().get("retrieval_fast")
    assert profile is not None

    messages, _ = ContextBuilder().build(state, profile, [])

    assistant_ids = {
        call.id
        for message in messages
        if message.role == "assistant"
        for call in message.tool_calls
    }
    tool_ids = {message.tool_call_id for message in messages if message.role == "tool"}
    assert assistant_ids == tool_ids
    assert messages[0].role == "system"
    assert any(message.role == "user" and message.content == "Fix the bug" for message in messages)


def test_context_builder_discards_incomplete_tool_call_group() -> None:
    messages = [
        Message(role="user", content="Fix the bug"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="a", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="b", name="read_file", arguments={"path": "b.py"}),
            ],
        ),
        Message(role="tool", tool_call_id="a", name="read_file", content="a"),
    ]

    compacted = ContextBuilder.fit_to_budget(messages, [], max_input_tokens=1)

    assert all(message.role != "tool" for message in compacted)
    assert all(not message.tool_calls for message in compacted)


@pytest.mark.asyncio
async def test_runtime_records_selected_capability_and_profile(tmp_path: Path) -> None:
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "a.txt"})]
            ),
            LLMResponse(content="done"),
        ]
    )
    (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
    runtime = AgentRuntime(
        client,
        build_p0_runtime_tools(),
        max_steps=2,
        capability_detector=CapabilityDetector(),
        profile_router=ProfileRouter(registry()),
    )
    state = await runtime.run("Read a.txt", tmp_path, "fake-model")
    assert state.current_capability == Capability.UNDERSTANDING.value
    assert state.current_profile == "understanding_fast"


@pytest.mark.asyncio
async def test_runtime_rejects_tool_outside_active_profile(tmp_path: Path) -> None:
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="write", name="write_file", arguments={"path": "a.txt", "content": "x"}
                    )
                ]
            ),
            LLMResponse(content="tool rejected"),
        ]
    )
    runtime = AgentRuntime(
        client,
        build_p0_runtime_tools(),
        max_steps=2,
        capability_detector=CapabilityDetector(),
        profile_router=ProfileRouter(registry()),
    )
    state = await runtime.run("Read a.txt", tmp_path, "fake-model")
    assert state.status == "completed"
    assert "not allowed by the active profile" in client.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_runtime_switches_profile_after_tool_failure(tmp_path: Path) -> None:
    primary = Profile(
        profile_id="retrieval_primary",
        capability=Capability.RETRIEVAL,
        model_ref="primary-model",
        instruction="Retrieve repository context.",
        tools=frozenset({"read_file"}),
        context_policy="retrieval",
        max_output_tokens=1024,
        max_steps=2,
    )
    backup = Profile(
        profile_id="retrieval_backup",
        capability=Capability.RETRIEVAL,
        model_ref="backup-model",
        instruction="Retrieve repository context with a fallback model.",
        tools=frozenset({"read_file"}),
        context_policy="retrieval",
        max_output_tokens=2048,
        max_steps=2,
    )
    client = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="missing",
                        name="read_file",
                        arguments={"path": "missing.txt"},
                    )
                ]
            ),
            LLMResponse(content="recovered"),
        ]
    )
    runtime = AgentRuntime(
        client,
        build_p0_runtime_tools(),
        max_steps=3,
        capability_detector=CapabilityDetector(),
        profile_router=ProfileRouter(
            ProfileRegistry({primary.profile_id: primary, backup.profile_id: backup})
        ),
        escalation_policy=EscalationPolicy(
            ProfileRegistry({primary.profile_id: primary, backup.profile_id: backup}),
            max_retries=0,
        ),
    )

    state = await runtime.run("Read a missing file", tmp_path, "selected-model")

    assert state.status == "completed"
    assert state.final_answer == "recovered"
    assert state.retry_count == 1
    assert [request.model for request in client.requests] == ["primary-model", "backup-model"]
    assert state.current_profile == "retrieval_backup"


@pytest.mark.asyncio
async def test_runtime_enforces_active_profile_step_budget(tmp_path: Path) -> None:
    profile = Profile(
        profile_id="retrieval_limited",
        capability=Capability.RETRIEVAL,
        model_ref="limited-model",
        instruction="Inspect one step only.",
        tools=frozenset({"list_files"}),
        context_policy="retrieval",
        max_output_tokens=1024,
        max_steps=1,
    )
    client = ScriptedLLM(
        [LLMResponse(tool_calls=[ToolCall(id="files", name="list_files", arguments={})])]
    )
    runtime = AgentRuntime(
        client,
        build_p0_runtime_tools(),
        max_steps=3,
        capability_detector=CapabilityDetector(),
        profile_router=ProfileRouter(ProfileRegistry({profile.profile_id: profile})),
    )

    state = await runtime.run("List the workspace", tmp_path, "selected-model")

    assert state.status == "failed"
    assert state.termination_reason == "profile_step_limit"
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_runtime_can_override_profile_step_budget_for_benchmark(tmp_path: Path) -> None:
    profile = Profile(
        profile_id="retrieval_limited",
        capability=Capability.RETRIEVAL,
        model_ref="limited-model",
        instruction="Inspect the workspace.",
        tools=frozenset({"list_files"}),
        context_policy="retrieval",
        max_output_tokens=1024,
        max_steps=1,
    )
    client = ScriptedLLM(
        [
            LLMResponse(tool_calls=[ToolCall(id="files", name="list_files", arguments={})]),
            LLMResponse(content="Workspace inspected."),
        ]
    )
    runtime = AgentRuntime(
        client,
        build_p0_runtime_tools(),
        max_steps=3,
        profile_step_limit=3,
        capability_detector=CapabilityDetector(),
        profile_router=ProfileRouter(ProfileRegistry({profile.profile_id: profile})),
    )

    state = await runtime.run("List the workspace", tmp_path, "selected-model")

    assert state.status == "completed"
    assert len(client.requests) == 2
