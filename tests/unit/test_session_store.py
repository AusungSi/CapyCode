from __future__ import annotations

from pathlib import Path

import pytest

from capycode.core import SessionState, SessionStore, recover_history
from capycode.llm import Message, ToolCall


def make_state(workspace: Path, task: str = "Inspect the project") -> SessionState:
    return SessionState(
        workspace=str(workspace),
        task=task,
        status="completed",
        final_answer="Done",
        history=[
            Message(role="system", content="system"),
            Message(role="user", content=task),
            Message(role="assistant", content="Done"),
        ],
    )


def test_session_store_round_trip_and_workspace_filter(tmp_path: Path) -> None:
    root = tmp_path / "local-sessions"
    workspace = tmp_path / "project"
    other_workspace = tmp_path / "other"
    workspace.mkdir()
    other_workspace.mkdir()
    store = SessionStore(root)
    state = make_state(workspace)

    saved = store.save(state, model_alias="small")

    assert saved.title == "Inspect the project"
    assert store.list(workspace)[0].session_id == state.session_id
    assert store.list(other_workspace) == []
    loaded = store.load(state.session_id, workspace)
    assert loaded.state.session_id == state.session_id
    assert loaded.state.status == "created"
    assert loaded.state.history[-1].role == "system"
    assert "resumed" in (loaded.state.history[-1].content or "")


def test_session_store_rejects_cross_workspace_resume(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    other_workspace = tmp_path / "other"
    workspace.mkdir()
    other_workspace.mkdir()
    store = SessionStore(tmp_path / "sessions")
    state = make_state(workspace)
    store.save(state, model_alias="small")

    with pytest.raises(ValueError, match="different workspace"):
        store.load(state.session_id, other_workspace)


def test_session_list_ignores_corrupt_records(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    root = tmp_path / "sessions"
    corrupt = root / ("a" * 32)
    corrupt.mkdir(parents=True)
    (corrupt / "session.json").write_text("not json", encoding="utf-8")

    assert SessionStore(root).list(workspace) == []


def test_recover_history_removes_incomplete_tool_groups() -> None:
    history = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="one", name="read_file", arguments={"path": "a.txt"}),
                ToolCall(id="two", name="read_file", arguments={"path": "b.txt"}),
            ],
        ),
        Message(role="tool", content="a", tool_call_id="one", name="read_file"),
        Message(role="assistant", content="An interrupted turn follows."),
        Message(role="tool", content="orphan", tool_call_id="orphan", name="read_file"),
    ]

    recovered = recover_history(history)

    assert [message.role for message in recovered] == ["system", "user", "assistant"]
    assert recovered[-1].content == "An interrupted turn follows."
