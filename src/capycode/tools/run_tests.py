from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult
from .process import CommandRunner
from .run_command import process_tool_result


class RunTestsInput(ToolInput):
    argv: list[str] = Field(default_factory=lambda: ["pytest", "-q"], min_length=1, max_length=64)
    cwd: str = Field(default=".", min_length=1)
    timeout_seconds: int = Field(default=120, ge=1, le=600)


class RunTestsTool(Tool):
    name: ClassVar[str] = "run_tests"
    description: ClassVar[str] = "Run a test command inside the workspace without invoking a shell."
    input_model: ClassVar[type[ToolInput]] = RunTestsInput

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or CommandRunner()

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, RunTestsInput):
            raise TypeError("run_tests received an unexpected argument model")
        result = await self.runner.run(
            workspace,
            arguments.argv,
            cwd=arguments.cwd,
            timeout_seconds=arguments.timeout_seconds,
        )
        tool_result = process_tool_result(result, label="Tests")
        return tool_result.model_copy(
            update={"data": {**tool_result.data, "tests_passed": tool_result.status == "success"}}
        )

    async def aclose(self) -> None:
        await self.runner.aclose()
