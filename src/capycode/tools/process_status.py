from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult
from .process import CommandRunner
from .run_command import process_tool_result


class ProcessStatusInput(ToolInput):
    task_id: str = Field(min_length=1, max_length=64)


class ProcessStatusTool(Tool):
    name: ClassVar[str] = "process_status"
    description: ClassVar[str] = (
        "Check a background command by task ID and return output when complete."
    )
    input_model: ClassVar[type[ToolInput]] = ProcessStatusInput

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, ProcessStatusInput):
            raise TypeError("process_status received an unexpected argument model")
        result = await self.runner.inspect_background(arguments.task_id)
        return process_tool_result(result, label="Background command")

    async def aclose(self) -> None:
        await self.runner.aclose()
