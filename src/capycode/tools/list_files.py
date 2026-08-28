from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult


class ListFilesInput(ToolInput):
    path: str = Field(default=".", min_length=1, description="Workspace-relative directory")
    pattern: str = Field(
        default="*", min_length=1, description="Glob matched against relative paths"
    )
    max_results: int = Field(default=500, ge=1, le=2_000)


class ListFilesTool(Tool):
    name: ClassVar[str] = "list_files"
    description: ClassVar[str] = "List files inside the workspace, optionally filtered by a glob."
    input_model: ClassVar[type[ToolInput]] = ListFilesInput

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, ListFilesInput):
            raise TypeError("list_files received an unexpected argument model")
        files = workspace.iter_files(
            arguments.path,
            pattern=arguments.pattern,
            max_results=arguments.max_results,
        )
        return ToolResult(
            status="success",
            content="\n".join(files) if files else "No matching files found.",
            data={"path": arguments.path, "count": len(files)},
        )
