from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from capycode.llm import LLMClient, LLMRequest, Message
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
Use run_in_background=true for servers and other persistent commands. Use process_status to
inspect them and stop_process when finished. Use foreground commands only for work that exits.
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
        max_output_tokens: int = 4096,
        max_tool_calls_per_step: int = 8,
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
        tracker = RunTracker(workspace.root, state.session_id, tracking) if tracking else None
        if tracker is not None:
            tracker.start(task)
            state.current_run_id = tracker.run_id
            state.last_trace_path = str(tracker.trace_path)
        self._checkpoint(state, checkpoint)
        previous_tool_signature: str | None = None
        repeated_tool_steps = 0

        try:
            for step in range(1, self.max_steps + 1):
                state.step = step
                await runtime_observer.on_model_start(step)
                model_started = time.perf_counter()
                first_token_at: float | None = None

                async def observe_text_delta(delta: str) -> None:
                    nonlocal first_token_at
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    await runtime_observer.on_text_delta(delta)

                response = await self.client.stream(
                    LLMRequest(
                        model=model,
                        messages=state.history,
                        tools=self.tools.definitions(),
                        max_output_tokens=self.max_output_tokens,
                    ),
                    observe_text_delta,
                )
                model_latency = time.perf_counter() - model_started
                state.total_input_tokens += response.usage.input_tokens
                state.total_output_tokens += response.usage.output_tokens
                state.total_latency += model_latency
                if tracker is not None:
                    state.total_cost += tracker.calculate_cost(
                        response.usage.input_tokens,
                        response.usage.output_tokens,
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
                    if signature == previous_tool_signature:
                        repeated_tool_steps += 1
                    else:
                        previous_tool_signature = signature
                        repeated_tool_steps = 1

                    exceeded_tool_limit = len(response.tool_calls) > self.max_tool_calls_per_step
                    seen_tool_calls: set[str] = set()
                    tool_traces: list[ToolCallTrace] = []
                    for index, call in enumerate(response.tool_calls):
                        await runtime_observer.on_tool_start(call.name, call.arguments)
                        if tracker is not None:
                            tracker.record_tool_request(step, call.id, call.name, call.arguments)
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
                        state.status = "failed"
                        state.termination_reason = "loop_detected"
                        state.last_error = "same tool calls repeated for 3 consecutive steps"
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
                    )
                if response.content:
                    state.final_answer = response.content
                    state.status = "completed"
                    state.termination_reason = "completed"
                    self._finish_tracking(state, tracker)
                    self._checkpoint(state, checkpoint)
                    return state

                state.status = "failed"
                state.termination_reason = "empty_model_response"
                state.last_error = "model returned neither text nor tool calls"
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
