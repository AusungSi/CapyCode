from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from capycode.llm.types import Message


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    workspace: str
    task: str
    status: Literal["created", "running", "completed", "failed"] = "created"
    termination_reason: str | None = None
    step: int = 0
    final_answer: str | None = None
    relevant_files: list[str] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)
    current_plan: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    current_diff: str = ""
    last_command: list[str] | None = None
    last_exit_code: int | None = None
    last_output: str = ""
    last_tests_passed: bool | None = None
    last_error: str | None = None
    current_capability: str | None = None
    current_profile: str | None = None
    current_model: str | None = None
    retry_count: int = 0
    current_run_id: str | None = None
    last_trace_path: str | None = None
    last_run_input_tokens: int = 0
    last_run_output_tokens: int = 0
    last_run_cost: float = 0.0
    last_run_currency: str = ""
    last_run_latency: float = 0.0
    capability_failures: dict[str, int] = Field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    history: list[Message] = Field(default_factory=list)
    compact_history: list[str] = Field(default_factory=list)
