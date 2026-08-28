from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult
from .process import CommandRunner
from .run_command import process_tool_result


class GitDiffInput(ToolInput):
    path: str = Field(default=".", min_length=1, description="Workspace-relative path")
    staged: bool = False


class GitDiffTool(Tool):
    name: ClassVar[str] = "git_diff"
    description: ClassVar[str] = "Show a read-only Git diff for a path inside the workspace."
    input_model: ClassVar[type[ToolInput]] = GitDiffInput

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or CommandRunner(allowed_executables=frozenset({"git"}))

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, GitDiffInput):
            raise TypeError("git_diff received an unexpected argument model")
        argv = ["git", "diff", "--no-ext-diff", "--no-color"]
        if arguments.staged:
            argv.append("--cached")
        argv.extend(["--", arguments.path])
        result = await self.runner.run(workspace, argv, timeout_seconds=60)
        tool_result = process_tool_result(result, label="Git diff")
        if tool_result.status == "success" and not result.stdout:
            return tool_result.model_copy(update={"content": "Git diff is empty."})
        return tool_result
