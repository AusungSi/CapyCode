from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .events import RUN_EVENT_ADAPTER, RunEvent, RunSummary, ToolRequestEvent, ToolResultEvent
from .redaction import TraceRedactor


class TraceRecorder:
    def __init__(
        self,
        workspace: Path,
        run_id: str,
        session_id: str,
        *,
        redactor: TraceRedactor | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.run_id = run_id
        self.session_id = session_id
        self.run_directory = self.workspace / ".capy" / "runs" / run_id
        self.trace_path = self.run_directory / "trace.jsonl"
        self.summary_path = self.run_directory / "summary.json"
        self.redactor = redactor or TraceRedactor()
        self._next_sequence = 1
        self._pending_tools: set[str] = set()
        self._result_tools: set[str] = set()
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self._stream = self.trace_path.open("x", encoding="utf-8", newline="\n")

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    @property
    def pending_tool_call_ids(self) -> frozenset[str]:
        return frozenset(self._pending_tools - self._result_tools)

    def append(self, event: RunEvent) -> None:
        if event.run_id != self.run_id or event.session_id != self.session_id:
            raise ValueError("event run or session does not match recorder")
        if event.sequence != self._next_sequence:
            raise ValueError(
                f"event sequence must be {self._next_sequence}, received {event.sequence}"
            )
        if isinstance(event, ToolRequestEvent):
            if event.tool_call_id in self._pending_tools:
                raise ValueError(f"duplicate tool request: {event.tool_call_id}")
            self._pending_tools.add(event.tool_call_id)
        elif isinstance(event, ToolResultEvent):
            if event.tool_call_id not in self._pending_tools:
                raise ValueError(f"tool result has no request: {event.tool_call_id}")
            if event.tool_call_id in self._result_tools:
                raise ValueError(f"duplicate tool result: {event.tool_call_id}")
            self._result_tools.add(event.tool_call_id)

        payload = event.model_dump(mode="json")
        redacted = self.redactor.redact(payload)
        validated = RUN_EVENT_ADAPTER.validate_python(redacted)
        self._stream.write(
            json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        )
        self._stream.write("\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._next_sequence += 1

    def write_summary(self, summary: RunSummary) -> None:
        if summary.run_id != self.run_id or summary.session_id != self.session_id:
            raise ValueError("summary run or session does not match recorder")
        self.run_directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="summary-", suffix=".tmp", dir=self.run_directory
        )
        temporary_path = Path(temporary_name)
        try:
            payload = self.redactor.redact(summary.model_dump(mode="json"))
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.summary_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()

    def __enter__(self) -> TraceRecorder:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
