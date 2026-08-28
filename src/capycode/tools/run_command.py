from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult
from .process import CommandRunner, ProcessResult


class RunCommandInput(ToolInput):
    argv: list[str] = Field(min_length=1, max_length=64)
    cwd: str = Field(default=".", min_length=1)
    timeout_seconds: int = Field(default=120, ge=1, le=600)


def process_tool_result(result: ProcessResult, *, label: str) -> ToolResult:
    sections = [f"{label} exited with code {result.exit_code}."]
    if result.stdout:
        sections.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        sections.append(f"STDERR:\n{result.stderr}")
    status: Literal["success", "error"] = (
        "error" if result.timed_out or result.exit_code != 0 else "success"
    )
    if result.timed_out:
        sections[0] = f"{label} timed out and was terminated."
    return ToolResult(
        status=status,
        content="\n\n".join(sections),
        data={
            "argv": list(result.argv),
            "cwd": str(result.cwd),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_seconds": result.duration_seconds,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
        },
    )


class RunCommandTool(Tool):
    name: ClassVar[str] = "run_command"
    description: ClassVar[str] = (
        "Run an allowlisted executable without a shell, using an argument array "
        "inside the workspace."
    )
    input_model: ClassVar[type[ToolInput]] = RunCommandInput

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or CommandRunner()

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, RunCommandInput):
            raise TypeError("run_command received an unexpected argument model")
        result = await self.runner.run(
            workspace,
            arguments.argv,
            cwd=arguments.cwd,
            timeout_seconds=arguments.timeout_seconds,
        )
        return process_tool_result(result, label="Command")
