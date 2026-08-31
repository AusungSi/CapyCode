from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from capycode.llm import LLMResponse
from capycode.tools import ToolResult

if TYPE_CHECKING:
    from capycode.core.state import SessionState

from .events import (
    AssistantTextEvent,
    RunEvent,
    RunStatusEvent,
    RunSummary,
    StepTraceEvent,
    ToolCallTrace,
    ToolRequestEvent,
    ToolResultEvent,
    UserTaskEvent,
)
from .recorder import TraceRecorder
from .redaction import TraceRedactor


@dataclass(frozen=True)
class RunTrackingConfig:
    provider: str
    model_id: str
    input_per_million: float
    output_per_million: float
    currency: str
    pricing_snapshot_date: str
    cached_input_per_million: float | None = None
    model_pricing: dict[str, tuple[float, float] | tuple[float, float, float | None]] = field(
        default_factory=dict
    )
    sensitive_values: tuple[str, ...] = ()
    event_sink: Callable[[RunEvent], None] | None = None


class RunTracker:
    def __init__(
        self,
        workspace: Path,
        session_id: str,
        config: RunTrackingConfig,
        *,
        run_id: str | None = None,
    ) -> None:
        self.run_id = run_id or uuid4().hex
        self.session_id = session_id
        self.workspace = workspace
        self.config = config
        self.recorder = TraceRecorder(
            workspace,
            self.run_id,
            session_id,
            redactor=TraceRedactor(config.sensitive_values),
        )
        self.started_at = datetime.now(UTC)
        self._started_counter = time.perf_counter()
        self.task = ""
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.retry_count = 0
        self.tool_requests = 0
        self.tool_successes = 0
        self.tool_failures = 0
        self._pending_tools: dict[str, tuple[int, str]] = {}
        self._summary: RunSummary | None = None

    @property
    def trace_path(self) -> Path:
        return self.recorder.trace_path

    def start(self, task: str) -> None:
        self.task = task
        self._append(
            RunStatusEvent(
                run_id=self.run_id,
                session_id=self.session_id,
                sequence=self.recorder.next_sequence,
                status="started",
            )
        )
        self._append(
            UserTaskEvent(
                run_id=self.run_id,
                session_id=self.session_id,
                sequence=self.recorder.next_sequence,
                task=task,
            )
        )

    def record_assistant(self, step: int, response: LLMResponse) -> None:
        self._append(
            AssistantTextEvent(
                run_id=self.run_id,
                session_id=self.session_id,
                sequence=self.recorder.next_sequence,
                step=step,
                text=response.content,
                finish_reason=response.finish_reason,
                tool_call_count=len(response.tool_calls),
            )
        )

    def record_tool_request(
        self,
        step: int,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> None:
        self._append(
            ToolRequestEvent(
                run_id=self.run_id,
                session_id=self.session_id,
                sequence=self.recorder.next_sequence,
                step=step,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
            )
        )
        self._pending_tools[tool_call_id] = (step, tool_name)
        self.tool_requests += 1

    def record_tool_result(
        self,
        step: int,
        tool_call_id: str,
        tool_name: str,
        result: ToolResult,
        latency_seconds: float,
    ) -> ToolCallTrace:
        self._append(
            ToolResultEvent(
                run_id=self.run_id,
                session_id=self.session_id,
                sequence=self.recorder.next_sequence,
                step=step,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status=result.status,
                content=result.content,
                data=result.data,
                latency_seconds=latency_seconds,
            )
        )
        self._pending_tools.pop(tool_call_id, None)
        if result.status == "success":
            self.tool_successes += 1
        else:
            self.tool_failures += 1
        truncated = bool(result.data.get("stdout_truncated")) or bool(
            result.data.get("stderr_truncated")
        )
        return ToolCallTrace(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=result.status,
            latency_seconds=latency_seconds,
            output_characters=len(result.content),
            truncated=truncated,
        )

    def record_step(
        self,
        step: int,
        response: LLMResponse,
        *,
        latency_seconds: float,
        first_token_latency_seconds: float | None,
        tools: list[ToolCallTrace],
        capability: str | None = None,
        profile_id: str | None = None,
        reasoning_effort: str | None = None,
        escalation_level: int = 0,
        model_id: str | None = None,
        route_reason: str | None = None,
    ) -> None:
        cost = self.calculate_cost(
            response.usage.input_tokens,
            response.usage.cached_input_tokens,
            response.usage.output_tokens,
            model_id=model_id,
        )
        self.input_tokens += response.usage.input_tokens
        self.cached_input_tokens += response.usage.cached_input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cost += cost
        self._append(
            StepTraceEvent(
                run_id=self.run_id,
                session_id=self.session_id,
                sequence=self.recorder.next_sequence,
                step=step,
                provider=self.config.provider,
                model_id=model_id or self.config.model_id,
                route_reason=route_reason,
                latency_seconds=latency_seconds,
                first_token_latency_seconds=first_token_latency_seconds,
                input_tokens=response.usage.input_tokens,
                cached_input_tokens=response.usage.cached_input_tokens,
                output_tokens=response.usage.output_tokens,
                cost=cost,
                currency=self.config.currency,
                finish_reason=response.finish_reason,
                retry_count=0,
                capability=capability,
                profile_id=profile_id,
                reasoning_effort=reasoning_effort,
                escalation_level=escalation_level,
                tools=tools,
            )
        )

    def synthesize_pending_results(self, reason: str) -> None:
        for tool_call_id, (step, tool_name) in list(self._pending_tools.items()):
            self.record_tool_result(
                step,
                tool_call_id,
                tool_name,
                ToolResult(status="error", content=reason, data={"synthetic": True}),
                0,
            )

    def finish(
        self,
        state: SessionState,
        *,
        status: str | None = None,
        termination_reason: str | None = None,
        error: str | None = None,
    ) -> RunSummary:
        if self._summary is not None:
            return self._summary
        resolved_status = cast(
            Literal["completed", "failed", "cancelled"],
            status or ("completed" if state.status == "completed" else "failed"),
        )
        if resolved_status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"invalid final run status: {resolved_status}")
        reason = termination_reason or state.termination_reason or resolved_status
        self._append(
            RunStatusEvent(
                run_id=self.run_id,
                session_id=self.session_id,
                sequence=self.recorder.next_sequence,
                step=state.step or None,
                status=resolved_status,
                termination_reason=reason,
                error=error or state.last_error,
            )
        )
        finished_at = datetime.now(UTC)
        summary = RunSummary(
            run_id=self.run_id,
            session_id=self.session_id,
            workspace=str(self.workspace),
            task=self.task,
            provider=self.config.provider,
            model_id=self.config.model_id,
            status=resolved_status,
            termination_reason=reason,
            started_at=self.started_at,
            finished_at=finished_at,
            latency_seconds=time.perf_counter() - self._started_counter,
            steps=state.step,
            input_tokens=self.input_tokens,
            cached_input_tokens=self.cached_input_tokens,
            output_tokens=self.output_tokens,
            cost=self.cost,
            currency=self.config.currency,
            retry_count=self.retry_count,
            tool_requests=self.tool_requests,
            tool_successes=self.tool_successes,
            tool_failures=self.tool_failures,
            tests_passed=state.last_tests_passed,
            modified_files=state.modified_files,
            final_diff=state.current_diff,
            trace_path=str(self.trace_path),
            pricing_snapshot_date=self.config.pricing_snapshot_date,
            pricing_configured=(
                self.config.input_per_million > 0 or self.config.output_per_million > 0
            ),
            error=error or state.last_error,
        )
        self.recorder.write_summary(summary)
        self.recorder.close()
        self._summary = summary
        return summary

    def calculate_cost(
        self,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        *,
        model_id: str | None = None,
    ) -> float:
        configured = self.config.model_pricing.get(model_id or self.config.model_id)
        if configured is None:
            input_price = self.config.input_per_million
            output_price = self.config.output_per_million
            cached_price = self.config.cached_input_per_million
        else:
            input_price, output_price, *cached_prices = configured
            cached_price = cached_prices[0] if cached_prices else None
        cached = min(max(cached_input_tokens, 0), input_tokens)
        uncached = input_tokens - cached
        effective_cached_price = input_price if cached_price is None else cached_price
        return (
            uncached * input_price + cached * effective_cached_price + output_tokens * output_price
        ) / 1_000_000

    def _append(self, event: RunEvent) -> None:
        persisted_event = self.recorder.append(event)
        if self.config.event_sink is not None:
            self.config.event_sink(persisted_event)
