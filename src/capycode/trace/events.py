from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from capycode.llm.types import ReasoningEffort


class TraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseRunEvent(TraceModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[0-9a-f]{32}$")
    event_type: str
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step: int | None = Field(default=None, ge=1)


class UserTaskEvent(BaseRunEvent):
    event_type: Literal["user_task"] = "user_task"
    task: str = Field(min_length=1)


class AssistantTextEvent(BaseRunEvent):
    event_type: Literal["assistant_text"] = "assistant_text"
    text: str | None = None
    finish_reason: str | None = None
    tool_call_count: int = Field(default=0, ge=0)


class ToolRequestEvent(BaseRunEvent):
    event_type: Literal["tool_request"] = "tool_request"
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(BaseRunEvent):
    event_type: Literal["tool_result"] = "tool_result"
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    status: Literal["success", "error"]
    content: str
    data: dict[str, Any] = Field(default_factory=dict)
    latency_seconds: float = Field(ge=0)


class ContextSnapshotEvent(BaseRunEvent):
    event_type: Literal["context_snapshot"] = "context_snapshot"
    compacted_start_sequence: int = Field(ge=1)
    compacted_end_sequence: int = Field(ge=1)
    estimated_tokens_before: int = Field(ge=0)
    estimated_tokens_after: int = Field(ge=0)
    content_hash: str = Field(min_length=1)


class ToolCallTrace(TraceModel):
    tool_call_id: str
    tool_name: str
    status: Literal["success", "error"]
    latency_seconds: float = Field(ge=0)
    output_characters: int = Field(ge=0)
    truncated: bool = False


class StepTraceEvent(BaseRunEvent):
    event_type: Literal["step_trace"] = "step_trace"
    step: int = Field(ge=1)
    provider: str
    model_id: str
    route_reason: str | None = None
    latency_seconds: float = Field(ge=0)
    first_token_latency_seconds: float | None = Field(default=None, ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(ge=0)
    cost: float = Field(ge=0)
    currency: str
    finish_reason: str | None = None
    retry_count: int = Field(default=0, ge=0)
    capability: str | None = None
    profile_id: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    escalation_level: int = Field(default=0, ge=0)
    tools: list[ToolCallTrace] = Field(default_factory=list)


class RunStatusEvent(BaseRunEvent):
    event_type: Literal["run_status"] = "run_status"
    status: Literal["started", "completed", "failed", "cancelled"]
    termination_reason: str | None = None
    error: str | None = None


RunEvent = Annotated[
    UserTaskEvent
    | AssistantTextEvent
    | ToolRequestEvent
    | ToolResultEvent
    | ContextSnapshotEvent
    | StepTraceEvent
    | RunStatusEvent,
    Field(discriminator="event_type"),
]
RUN_EVENT_ADAPTER: TypeAdapter[RunEvent] = TypeAdapter(RunEvent)


class RunSummary(TraceModel):
    schema_version: int = 2
    run_id: str
    session_id: str
    workspace: str
    task: str
    provider: str
    model_id: str
    status: Literal["completed", "failed", "cancelled"]
    termination_reason: str
    started_at: datetime
    finished_at: datetime
    latency_seconds: float = Field(ge=0)
    steps: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(ge=0)
    cost: float = Field(ge=0)
    currency: str
    retry_count: int = Field(ge=0)
    tool_requests: int = Field(ge=0)
    tool_successes: int = Field(ge=0)
    tool_failures: int = Field(ge=0)
    tests_passed: bool | None = None
    modified_files: list[str] = Field(default_factory=list)
    final_diff: str = ""
    trace_path: str
    pricing_snapshot_date: str
    pricing_configured: bool
    error: str | None = None
