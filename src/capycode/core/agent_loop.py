from __future__ import annotations

from pathlib import Path

from capycode.llm import LLMClient, LLMRequest, Message
from capycode.tools import ToolExecutor, ToolRegistry
from capycode.workspace import LocalWorkspace

from .observer import NullRuntimeObserver, RuntimeObserver
from .state import SessionState

SYSTEM_PROMPT = """You are CapyCode, a local coding agent.
Use the available tools to inspect the workspace before answering questions about repository files.
Do not claim to have read a file unless a successful tool result is present in the conversation.
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
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.client = client
        self.tools = tools
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens

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
                for call in response.tool_calls:
                    await runtime_observer.on_tool_start(call.name, call.arguments)
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
                    if result.status == "success" and call.name == "read_file":
                        path = str(result.data["path"])
                        if path not in state.relevant_files:
                            state.relevant_files.append(path)
                    elif result.status == "error":
                        state.last_error = result.content
                continue

            if response.content:
                state.final_answer = response.content
                state.status = "completed"
                return state

            state.status = "failed"
            state.last_error = "model returned neither text nor tool calls"
            return state

        state.status = "failed"
        state.last_error = f"maximum step limit reached: {self.max_steps}"
        return state
