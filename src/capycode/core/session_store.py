from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from capycode.llm import Message

from .state import SessionState

SESSION_VERSION = 1
RESUME_NOTICE = (
    "This conversation was resumed from local history. Files and processes may have changed "
    "since the previous run. Re-read relevant files before modifying or relying on them. "
    "Background processes from the previous terminal are not running."
)


class SessionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = SESSION_VERSION
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    workspace: str
    model_alias: str
    title: str
    created_at: datetime
    updated_at: datetime
    state: SessionState


class SessionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    title: str
    workspace: str
    model_alias: str
    created_at: datetime
    updated_at: datetime
    status: str


class SessionStore:
    """Local, project-neutral persistence for resumable conversations."""

    def __init__(self, root: Path | None = None) -> None:
        capycode_home = os.getenv("CAPYCODE_HOME")
        self.root = root or (
            Path(capycode_home) / "sessions"
            if capycode_home
            else Path.home() / ".capycode" / "sessions"
        )

    def save(self, state: SessionState, *, model_alias: str) -> SessionRecord:
        now = datetime.now(UTC)
        path = self._record_path(state.session_id)
        existing = self._read_record(path, strict=False)
        created_at = existing.created_at if existing is not None else now
        title = existing.title if existing is not None else self._make_title(state.task)
        record = SessionRecord(
            session_id=state.session_id,
            workspace=str(Path(state.workspace).resolve()),
            model_alias=model_alias,
            title=title,
            created_at=created_at,
            updated_at=now,
            state=state,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="session-", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(record.model_dump_json(indent=2))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, path)
            path.chmod(0o600)
        finally:
            temporary_path.unlink(missing_ok=True)
        return record

    def list(self, workspace: Path) -> list[SessionSummary]:
        expected_workspace = workspace.resolve()
        if not self.root.exists():
            return []
        summaries: list[SessionSummary] = []
        for path in self.root.glob("*/session.json"):
            record = self._read_record(path, strict=False)
            if record is None or Path(record.workspace).resolve() != expected_workspace:
                continue
            summaries.append(
                SessionSummary(
                    session_id=record.session_id,
                    title=record.title,
                    workspace=record.workspace,
                    model_alias=record.model_alias,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    status=record.state.status,
                )
            )
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    def load(self, session_id: str, workspace: Path) -> SessionRecord:
        record = self._read_record(self._record_path(session_id), strict=True)
        assert record is not None
        if Path(record.workspace).resolve() != workspace.resolve():
            raise ValueError("session belongs to a different workspace")
        recovered = recover_history(record.state.history)
        recovered.append(Message(role="system", content=RESUME_NOTICE))
        record.state.history = recovered
        record.state.status = "created"
        record.state.termination_reason = None
        record.state.final_answer = None
        record.state.last_error = None
        return record

    def resolve(self, value: str, workspace: Path) -> SessionRecord:
        sessions = self.list(workspace)
        if not sessions:
            raise ValueError("当前工作区没有可恢复的会话")
        query = value.strip().lower()
        if query in {"", "latest"}:
            return self.load(sessions[0].session_id, workspace)
        matches = [
            item
            for item in sessions
            if item.session_id.startswith(query) or item.title.lower() == query
        ]
        if not matches:
            raise ValueError(f"未找到会话: {value}")
        if len(matches) > 1:
            raise ValueError(f"会话标识不唯一: {value}")
        return self.load(matches[0].session_id, workspace)

    def _record_path(self, session_id: str) -> Path:
        if len(session_id) != 32 or any(char not in "0123456789abcdef" for char in session_id):
            raise ValueError("invalid session id")
        return self.root / session_id / "session.json"

    @staticmethod
    def _read_record(path: Path, *, strict: bool) -> SessionRecord | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = SessionRecord.model_validate(payload)
            if record.version != SESSION_VERSION or record.session_id != path.parent.name:
                raise ValueError("unsupported or mismatched session record")
            return record
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            if strict:
                raise ValueError(f"无法读取会话记录 {path}: {exc}") from exc
            return None

    @staticmethod
    def _make_title(task: str) -> str:
        title = " ".join(task.strip().split())
        return title if len(title) <= 64 else title[:61] + "..."


def recover_history(history: list[Message]) -> list[Message]:
    """Remove orphaned tool messages and interrupted assistant tool-call groups."""

    recovered: list[Message] = []
    index = 0
    while index < len(history):
        message = history[index]
        if message.role == "tool":
            index += 1
            continue
        if message.role != "assistant" or not message.tool_calls:
            recovered.append(message)
            index += 1
            continue

        expected = {call.id for call in message.tool_calls}
        tool_messages: list[Message] = []
        cursor = index + 1
        while cursor < len(history) and history[cursor].role == "tool":
            tool_messages.append(history[cursor])
            cursor += 1
        actual = [item.tool_call_id for item in tool_messages]
        if len(actual) == len(expected) and set(actual) == expected:
            recovered.append(message)
            recovered.extend(tool_messages)
        index = cursor
    return recovered
