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
    run_in_background: bool = False


def process_tool_result(result: ProcessResult, *, label: str) -> ToolResult:
    if result.running:
        return ToolResult(
            status="success",
            content=(
                f"{label} started in the background with task ID "
                f"{result.background_task_id}. Use process_status to inspect it "
                "and stop_process when it is no longer needed."
            ),
            data={
                "argv": list(result.argv),
                "cwd": str(result.cwd),
                "exit_code": None,
                "timed_out": False,
                "background_task_id": result.background_task_id,
                "running": True,
            },
        )
    sections = [f"{label} exited with code {result.exit_code}."]
    if result.stdout:
        sections.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        sections.append(f"STDERR:\n{result.stderr}")
    status: Literal["success", "error"] = (
        "error" if result.timed_out or result.exit_code != 0 else "success"
    )
    if result.timed_out:
        sections[0] = (
            f"{label} timed out and was terminated. For a server or persistent process, "
            "run it again with run_in_background=true."
        )
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
            "background_task_id": result.background_task_id,
            "running": False,
        },
    )


class RunCommandTool(Tool):
    name: ClassVar[str] = "run_command"
    description: ClassVar[str] = (
        "Run a configured development command without a shell. Set run_in_background=true "
        "for servers and other long-running processes."
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
            run_in_background=arguments.run_in_background,
        )
        return process_tool_result(result, label="Command")

    async def aclose(self) -> None:
        await self.runner.aclose()
