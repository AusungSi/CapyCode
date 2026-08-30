from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from capycode.capability import (
    CapabilityDetector,
    ContextBuilder,
    EscalationAction,
    EscalationPolicy,
    ProfileRouter,
)
from capycode.llm import LLMClient, LLMError, LLMRequest, Message
from capycode.llm.types import DEFAULT_MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS_UPPER_LIMIT
from capycode.tools import ToolExecutor, ToolRegistry, ToolResult
from capycode.trace import RunTracker, RunTrackingConfig, ToolCallTrace
from capycode.workspace import LocalWorkspace

from .observer import NullRuntimeObserver, RuntimeObserver
from .state import SessionState

SYSTEM_PROMPT = """You are CapyCode, a local coding agent.
Use the available tools to inspect the workspace before answering questions about repository files.
Do not claim to have read a file unless a successful tool result is present in the conversation.
Existing files must be fully read before write_file or replace_text can modify them.
After changing code, run relevant tests and inspect git_diff before presenting the result.
For coding tasks, keep the loop action-oriented: inspect only what is needed, then edit the
implementation, run a focused test, and iterate on failures. Do not spend repeated turns
re-describing the problem without making progress. If a search is inconclusive, choose the
most likely implementation file and inspect it directly.
Use run_in_background=true for servers and other persistent commands. Use process_status to
inspect them and stop_process when finished. Use foreground commands only for work that exits.
run_command accepts an argv array and does not invoke a shell; do not use pipes, redirects,
&&, or other shell operators. Run separate commands or use the program's own filtering options.
Git diff may be unavailable in non-Git workspaces; do not retry it through run_command.
When enough evidence is available, answer the user's task directly.
"""


class AgentRuntime:
    def __init__(
        self,
        client: LLMClient,
        tools: ToolRegistry,
        *,
        max_steps: int = 10,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_tool_calls_per_step: int = 8,
        context_window: int | None = None,
        profile_step_limit: int | None = None,
        capability_detector: CapabilityDetector | None = None,
        profile_router: ProfileRouter | None = None,
        context_builder: ContextBuilder | None = None,
        escalation_policy: EscalationPolicy | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.client = client
        self.tools = tools
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        if max_tool_calls_per_step <= 0:
            raise ValueError("max_tool_calls_per_step must be positive")
        self.max_tool_calls_per_step = max_tool_calls_per_step
        if context_window is not None and context_window <= 0:
            raise ValueError("context_window must be positive")
        self.context_window = context_window
        if profile_step_limit is not None and profile_step_limit <= 0:
            raise ValueError("profile_step_limit must be positive")
        self.profile_step_limit = profile_step_limit
        self.capability_detector = capability_detector
        self.profile_router = profile_router
        self.context_builder = context_builder or ContextBuilder()
        self.escalation_policy = escalation_policy

    async def run(
        self,
        task: str,
        workspace_path: Path,
        model: str,
        observer: RuntimeObserver | None = None,
        state: SessionState | None = None,
        checkpoint: Callable[[SessionState], None] | None = None,
        tracking: RunTrackingConfig | None = None,
    ) -> SessionState:
        if not task.strip():
            raise ValueError("task cannot be empty")
        workspace = LocalWorkspace(workspace_path)
        executor = ToolExecutor(self.tools, workspace)
        runtime_observer = observer or NullRuntimeObserver()
        if state is None:
            state = SessionState(
                workspace=str(workspace.root),
                task=task,
                status="running",
                current_model=model,
                history=[
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=task),
                ],
            )
        else:
            saved_workspace = os.path.normcase(state.workspace)
            active_workspace = os.path.normcase(str(workspace.root))
            if saved_workspace != active_workspace:
                raise ValueError("session workspace does not match the active workspace")
            if state.history and state.history[0].role == "system":
                state.history[0] = Message(role="system", content=SYSTEM_PROMPT)
            else:
                state.history.insert(0, Message(role="system", content=SYSTEM_PROMPT))
            state.task = task
            state.status = "running"
            state.termination_reason = None
            state.final_answer = None
            state.last_error = None
            state.step = 0
            state.current_model = model
            state.history.append(Message(role="user", content=task))
        state.retry_count = 0
        state.capability_failures.clear()
        tracker = RunTracker(workspace.root, state.session_id, tracking) if tracking else None
        if tracker is not None:
            tracker.start(task)
            state.current_run_id = tracker.run_id
            state.last_trace_path = str(tracker.trace_path)
        self._checkpoint(state, checkpoint)
        previous_tool_signature: str | None = None
        repeated_tool_steps = 0
        loop_recoveries = 0
        max_output_tokens_retry: int | None = None
        reasoning_only_retries = 0
        stagnant_steps = 0
        progress_nudges = 0
        preferred_profile_id: str | None = None
        profile_step_counts: dict[str, int] = {}
        context_budget_override: int | None = None
        context_limit_retries = 0

        try:
            for step in range(1, self.max_steps + 1):
                state.step = step
                capability = None
                profile_id = None
                route_reason = None
                escalation_level = 0
                request_model = model
                request_max_output_tokens = self.max_output_tokens
                request_messages = state.history
                request_tools = self.tools.definitions()
                if self.capability_detector is not None and self.profile_router is not None:
                    capability_decision = self.capability_detector.detect(state)
                    route = self.profile_router.select(
                        capability_decision,
                        state,
                        preferred_profile_id=preferred_profile_id,
                    )
                    profile = self.profile_router.profile(route)
                    capability = route.capability.value
                    profile_id = route.profile_id
                    escalation_level = route.escalation_level
                    route_reason = route.reason
                    state.current_capability = capability
                    state.current_profile = profile_id
                    completed_profile_steps = profile_step_counts.get(profile_id, 0)
                    effective_profile_step_limit = self.profile_step_limit or profile.max_steps
                    if completed_profile_steps >= effective_profile_step_limit:
                        state.status = "failed"
                        state.termination_reason = "profile_step_limit"
                        state.last_error = (
                            "profile "
                            f"{profile_id} reached its step limit: {effective_profile_step_limit}"
                        )
                        self._finish_tracking(state, tracker)
                        self._checkpoint(state, checkpoint)
                        return state
                    profile_step_counts[profile_id] = completed_profile_steps + 1
                    request_model = model if profile.model_ref == "current" else profile.model_ref
                    request_max_output_tokens = min(
                        self.max_output_tokens, profile.max_output_tokens
                    )
                    request_messages, request_tools = self.context_builder.build(
                        state, profile, request_tools
                    )
                if max_output_tokens_retry is not None:
                    request_max_output_tokens = max_output_tokens_retry
                    max_output_tokens_retry = None
                configured_input_budget = (
                    max(4_000, self.context_window - request_max_output_tokens)
                    if self.context_window is not None
                    else None
                )
                effective_input_budget = context_budget_override or configured_input_budget
                request_messages = self.context_builder.fit_to_budget(
                    request_messages, request_tools, effective_input_budget
                )
                allowed_tool_names = {tool.name for tool in request_tools}
                await runtime_observer.on_model_start(step)
                model_started = time.perf_counter()
                first_token_at: float | None = None

                async def observe_text_delta(delta: str) -> None:
                    nonlocal first_token_at
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    await runtime_observer.on_text_delta(delta)

                try:
                    response = await self.client.stream(
                        LLMRequest(
                            model=request_model,
                            messages=request_messages,
                            tools=request_tools,
                            max_output_tokens=request_max_output_tokens,
                        ),
                        observe_text_delta,
                    )
                except LLMError as exc:
                    state.last_error = str(exc)
                    if self._is_context_limit_error(exc) and context_limit_retries < 2:
                        current_estimate = self.context_builder.estimate_tokens(
                            request_messages, request_tools
                        )
                        reduced_budget = max(4_000, int(current_estimate * 0.6))
                        if reduced_budget < current_estimate:
                            context_budget_override = reduced_budget
                            context_limit_retries += 1
                            state.retry_count += 1
                            continue
                    if self.escalation_policy is None or profile_id is None or not exc.retryable:
                        raise
                    state.retry_count += 1
                    escalation = self.escalation_policy.decide(state, profile_id, failed=True)
                    if escalation.action == EscalationAction.RETRY:
                        continue
                    if escalation.action == EscalationAction.SWITCH_PROFILE:
                        preferred_profile_id = escalation.next_profile_id
                        continue
                    raise RuntimeError(escalation.reason) from exc
                model_latency = time.perf_counter() - model_started
                state.total_input_tokens += response.usage.input_tokens
                state.total_cached_input_tokens += response.usage.cached_input_tokens
                state.total_output_tokens += response.usage.output_tokens
                state.total_latency += model_latency
                if tracker is not None:
                    state.total_cost += tracker.calculate_cost(
                        response.usage.input_tokens,
                        response.usage.cached_input_tokens,
                        response.usage.output_tokens,
                        model_id=request_model,
                    )
                    tracker.record_assistant(step, response)
                state.history.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                self._checkpoint(state, checkpoint)

                if response.tool_calls:
                    signature = json.dumps(
                        [(call.name, call.arguments) for call in response.tool_calls],
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    # Polling a background command is intentionally repetitive:
                    # the status can remain unchanged across several turns
                    # while the process is still running. Do not classify that
                    # normal wait pattern as an agent loop.
                    polling_only = all(
                        call.name == "process_status" for call in response.tool_calls
                    )
                    if polling_only:
                        repeated_tool_steps = 0
                    elif signature == previous_tool_signature:
                        repeated_tool_steps += 1
                    else:
                        previous_tool_signature = signature
                        repeated_tool_steps = 1

                    exceeded_tool_limit = len(response.tool_calls) > self.max_tool_calls_per_step
                    seen_tool_calls: set[str] = set()
                    tool_traces: list[ToolCallTrace] = []
                    tool_failed = False
                    for index, call in enumerate(response.tool_calls):
                        if tracker is not None:
                            tracker.record_tool_request(step, call.id, call.name, call.arguments)
                        await runtime_observer.on_tool_start(call.name, call.arguments)
                        tool_started = time.perf_counter()
                        call_signature = json.dumps(
                            (call.name, call.arguments),
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if index >= self.max_tool_calls_per_step:
                            result = self._tool_limit_result()
                        elif call_signature in seen_tool_calls:
                            result = ToolResult(
                                status="success",
                                content=(
                                    "Duplicate tool call skipped because the same tool and "
                                    "arguments already ran in this step."
                                ),
                                data={"duplicate_skipped": True},
                            )
                        elif call.name not in allowed_tool_names:
                            result = ToolResult(
                                status="error",
                                content=f"tool is not allowed by the active profile: {call.name}",
                            )
                        else:
                            seen_tool_calls.add(call_signature)
                            result = await executor.execute(call.name, call.arguments)
                        tool_latency = time.perf_counter() - tool_started
                        state.total_latency += tool_latency
                        if tracker is not None:
                            tool_traces.append(
                                tracker.record_tool_result(
                                    step,
                                    call.id,
                                    call.name,
                                    result,
                                    tool_latency,
                                )
                            )
                        await runtime_observer.on_tool_result(call.name, result)
                        tool_failed = tool_failed or result.status == "error"
                        state.history.append(
                            Message(
                                role="tool",
                                content=result.model_dump_json(),
                                tool_call_id=call.id,
                                name=call.name,
                            )
                        )
                        self._update_state_from_tool(state, call.name, result)
                        self._checkpoint(state, checkpoint)
                    if tracker is not None:
                        tracker.record_step(
                            step,
                            response,
                            latency_seconds=model_latency,
                            first_token_latency_seconds=(
                                first_token_at - model_started
                                if first_token_at is not None
                                else None
                            ),
                            tools=tool_traces,
                            capability=capability,
                            profile_id=profile_id,
                            escalation_level=escalation_level,
                            model_id=request_model,
                            route_reason=route_reason,
                        )
                    if exceeded_tool_limit:
                        state.status = "failed"
                        state.termination_reason = "tool_call_limit_exceeded"
                        state.last_error = (
                            f"model requested {len(response.tool_calls)} tools in one step; "
                            f"limit is {self.max_tool_calls_per_step}"
                        )
                        self._finish_tracking(state, tracker)
                        self._checkpoint(state, checkpoint)
                        return state
                    if repeated_tool_steps >= 3:
                        if loop_recoveries < 2:
                            loop_recoveries += 1
                            previous_tool_signature = None
                            repeated_tool_steps = 0
                            state.history.append(
                                Message(
                                    role="user",
                                    content=(
                                        "Loop recovery: the same tool call has already run three "
                                        "times without progress. Do not repeat it. Inspect the "
                                        "last result, choose a different tool or arguments, and "
                                        "with a concrete fix."
                                    ),
                                )
                            )
                            self._checkpoint(state, checkpoint)
                            continue
                        state.status = "failed"
                        state.termination_reason = "loop_detected"
                        state.last_error = "same tool calls repeated for 3 consecutive steps"
                        self._finish_tracking(state, tracker)
                        self._checkpoint(state, checkpoint)
                        return state
                    mutating_or_verifying = any(
                        call.name in {"write_file", "replace_text", "run_tests", "run_command"}
                        for call in response.tool_calls
                    )
                    stagnant_steps = 0 if mutating_or_verifying else stagnant_steps + 1
                    if stagnant_steps >= 8 and progress_nudges < 3:
                        # Claude Code keeps the main loop action-oriented when an
                        # agent spends too long exploring without changing or
                        # verifying anything. Add a short, explicit nudge to the
                        # existing conversation instead of burning the remaining
                        # step budget silently.
                        progress_nudges += 1
                        stagnant_steps = 0
                        state.history.append(
                            Message(
                                role="user",
                                content=(
                                    "Progress checkpoint: you have only inspected files so far. "
                                    "Choose the most likely implementation file, make the fix now, "
                                    "then run a focused test."
                                ),
                            )
                        )
                    if self.escalation_policy is not None and profile_id is not None:
                        escalation = self.escalation_policy.decide(
                            state, profile_id, failed=tool_failed
                        )
                        if tool_failed:
                            state.retry_count += 1
                            if escalation.action == EscalationAction.SWITCH_PROFILE:
                                preferred_profile_id = escalation.next_profile_id
                            elif escalation.action == EscalationAction.FAIL:
                                state.status = "failed"
                                state.termination_reason = "escalation_exhausted"
                                state.last_error = escalation.reason
                                self._finish_tracking(state, tracker)
                                self._checkpoint(state, checkpoint)
                                return state
                    continue

                if tracker is not None:
                    tracker.record_step(
                        step,
                        response,
                        latency_seconds=model_latency,
                        first_token_latency_seconds=(
                            first_token_at - model_started if first_token_at is not None else None
                        ),
                        tools=[],
                        capability=capability,
                        profile_id=profile_id,
                        escalation_level=escalation_level,
                        model_id=request_model,
                        route_reason=route_reason,
                    )
                if response.content:
                    state.final_answer = response.content
                    state.status = "completed"
                    state.termination_reason = "completed"
                    self._finish_tracking(state, tracker)
                    self._checkpoint(state, checkpoint)
                    return state

                # DeepSeek and several compatible gateways may return a
                # reasoning-only stream before producing the actual answer.
                # Ask for a continuation instead of failing immediately.
                if response.reasoning_content and reasoning_only_retries < 2:
                    reasoning_only_retries += 1
                    state.history.pop()
                    state.history.append(
                        Message(
                            role="user",
                            content=(
                                "Continue the coding task. Use the available tools now; "
                                "do not return reasoning only."
                            ),
                        )
                    )
                    continue

                if (
                    response.finish_reason == "length"
                    and request_max_output_tokens < MAX_OUTPUT_TOKENS_UPPER_LIMIT
                ):
                    # Match Claude Code's clean one-shot escalation for a response
                    # exhausted before it could produce a usable tool call or answer.
                    state.history.pop()
                    max_output_tokens_retry = MAX_OUTPUT_TOKENS_UPPER_LIMIT
                    continue

                # Gateways occasionally emit an empty 200 response while a
                # long tool-running conversation is still being assembled.
                # Claude Code continues the turn a few times instead of
                # terminating immediately; do the same with a bounded retry.
                if reasoning_only_retries < 3:
                    reasoning_only_retries += 1
                    state.history.pop()
                    state.history.append(
                        Message(
                            role="user",
                            content=(
                                "Continue from the last successful tool result. "
                                "Return a tool call or a concise final answer; "
                                "do not return empty content."
                            ),
                        )
                    )
                    await asyncio.sleep(0.2 * reasoning_only_retries)
                    continue

                state.status = "failed"
                state.termination_reason = "empty_model_response"
                state.last_error = "model returned neither text nor tool calls after retries"
                self._finish_tracking(state, tracker)
                self._checkpoint(state, checkpoint)
                return state

            state.status = "failed"
            state.termination_reason = "max_steps"
            state.last_error = f"maximum step limit reached: {self.max_steps}"
            self._finish_tracking(state, tracker)
            self._checkpoint(state, checkpoint)
            return state
        except asyncio.CancelledError:
            state.status = "failed"
            state.termination_reason = "cancelled"
            state.last_error = "run cancelled by user"
            if tracker is not None:
                tracker.synthesize_pending_results("tool call cancelled by user")
                self._finish_tracking(state, tracker, status="cancelled")
            self._checkpoint(state, checkpoint)
            raise
        except Exception as exc:
            state.status = "failed"
            state.termination_reason = "runtime_error"
            state.last_error = str(exc)
            if tracker is not None:
                tracker.synthesize_pending_results(f"tool call interrupted: {exc}")
                self._finish_tracking(state, tracker, error=str(exc))
            self._checkpoint(state, checkpoint)
            raise

    @staticmethod
    def _is_context_limit_error(exc: LLMError) -> bool:
        if exc.status_code != 400:
            return False
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "context length",
                "context window",
                "maximum context",
                "too many tokens",
                "prompt is too long",
                "input tokens exceed",
            )
        )

    @staticmethod
    def _checkpoint(
        state: SessionState,
        checkpoint: Callable[[SessionState], None] | None,
    ) -> None:
        if checkpoint is not None:
            checkpoint(state)

    @staticmethod
    def _finish_tracking(
        state: SessionState,
        tracker: RunTracker | None,
        *,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        if tracker is None:
            return
        summary = tracker.finish(state, status=status, error=error)
        state.last_run_input_tokens = summary.input_tokens
        state.last_run_cached_input_tokens = summary.cached_input_tokens
        state.last_run_output_tokens = summary.output_tokens
        state.last_run_cost = summary.cost
        state.last_run_currency = summary.currency
        state.last_run_latency = summary.latency_seconds
        state.last_trace_path = summary.trace_path

    def _tool_limit_result(self) -> ToolResult:
        return ToolResult(
            status="error",
            content=f"tool call skipped: per-step limit is {self.max_tool_calls_per_step}",
        )

    @staticmethod
    def _update_state_from_tool(
        state: SessionState,
        name: str,
        result: ToolResult,
    ) -> None:
        if result.status == "error":
            state.last_error = result.content
            if state.current_capability:
                key = state.current_capability
                state.capability_failures[key] = state.capability_failures.get(key, 0) + 1
        elif state.last_error and name in {"run_tests", "run_command", "read_file", "search_code"}:
            state.last_error = None
        path = result.data.get("path")
        if result.status == "success" and name == "read_file" and isinstance(path, str):
            if path not in state.relevant_files:
                state.relevant_files.append(path)
        if result.status == "success" and name in {"replace_text", "write_file"}:
            if isinstance(path, str) and path not in state.modified_files:
                state.modified_files.append(path)
        if name in {"run_command", "run_tests"}:
            argv = result.data.get("argv")
            state.last_command = [str(item) for item in argv] if isinstance(argv, list) else None
            exit_code = result.data.get("exit_code")
            state.last_exit_code = exit_code if isinstance(exit_code, int) else None
            state.last_output = result.content
        if name == "run_tests":
            tests_passed = result.data.get("tests_passed")
            state.last_tests_passed = tests_passed if isinstance(tests_passed, bool) else None
        if name == "git_diff" and result.status == "success":
            state.current_diff = result.content
