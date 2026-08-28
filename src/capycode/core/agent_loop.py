from __future__ import annotations

import json
from pathlib import Path

from capycode.llm import LLMClient, LLMRequest, Message
from capycode.tools import ToolExecutor, ToolRegistry, ToolResult
from capycode.workspace import LocalWorkspace

from .observer import NullRuntimeObserver, RuntimeObserver
from .state import SessionState

SYSTEM_PROMPT = """You are CapyCode, a local coding agent.
Use the available tools to inspect the workspace before answering questions about repository files.
Do not claim to have read a file unless a successful tool result is present in the conversation.
Existing files must be fully read before write_file or replace_text can modify them.
After changing code, run relevant tests and inspect git_diff before presenting the result.
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
    ) -> SessionState:
        if not task.strip():
            raise ValueError("task cannot be empty")
        workspace = LocalWorkspace(workspace_path)
        executor = ToolExecutor(self.tools, workspace)
        runtime_observer = observer or NullRuntimeObserver()
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
        previous_tool_signature: str | None = None
        repeated_tool_steps = 0

        for step in range(1, self.max_steps + 1):
            state.step = step
            await runtime_observer.on_model_start(step)
            response = await self.client.stream(
                LLMRequest(
                    model=model,
                    messages=state.history,
                    tools=self.tools.definitions(),
                    max_output_tokens=self.max_output_tokens,
                ),
                runtime_observer.on_text_delta,
            )
            state.total_input_tokens += response.usage.input_tokens
            state.total_output_tokens += response.usage.output_tokens
            state.history.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

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
                for index, call in enumerate(response.tool_calls):
                    await runtime_observer.on_tool_start(call.name, call.arguments)
                    if index >= self.max_tool_calls_per_step:
                        result = self._tool_limit_result()
                    else:
                        result = await executor.execute(call.name, call.arguments)
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
                if exceeded_tool_limit:
                    state.status = "failed"
                    state.termination_reason = "tool_call_limit_exceeded"
                    state.last_error = (
                        f"model requested {len(response.tool_calls)} tools in one step; "
                        f"limit is {self.max_tool_calls_per_step}"
                    )
                    return state
                if repeated_tool_steps >= 3:
                    state.status = "failed"
                    state.termination_reason = "loop_detected"
                    state.last_error = "same tool calls repeated for 3 consecutive steps"
                    return state
                continue

            if response.content:
                state.final_answer = response.content
                state.status = "completed"
                state.termination_reason = "completed"
                return state

            state.status = "failed"
            state.termination_reason = "empty_model_response"
            state.last_error = "model returned neither text nor tool calls"
            return state

        state.status = "failed"
        state.termination_reason = "max_steps"
        state.last_error = f"maximum step limit reached: {self.max_steps}"
        return state

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
