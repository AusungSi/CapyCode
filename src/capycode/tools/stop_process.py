from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult
from .process import CommandRunner
from .run_command import process_tool_result


class StopProcessInput(ToolInput):
    task_id: str = Field(min_length=1, max_length=64)


class StopProcessTool(Tool):
    name: ClassVar[str] = "stop_process"
    description: ClassVar[str] = "Terminate a background command and collect its final output."
    input_model: ClassVar[type[ToolInput]] = StopProcessInput

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, StopProcessInput):
            raise TypeError("stop_process received an unexpected argument model")
        result = await self.runner.stop_background(arguments.task_id)
        tool_result = process_tool_result(result, label="Background command")
        return tool_result.model_copy(
            update={
                "status": "success",
                "content": "Background command stopped.\n\n" + tool_result.content,
            }
        )

    async def aclose(self) -> None:
        await self.runner.aclose()
