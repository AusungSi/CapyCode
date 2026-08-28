from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult


class ReadFileInput(ToolInput):
    path: str = Field(min_length=1, description="Workspace-relative file path")


class ReadFileTool(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Read a UTF-8 text file inside the current workspace."
    input_model: ClassVar[type[ToolInput]] = ReadFileInput

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, ReadFileInput):
            raise TypeError("read_file received an unexpected argument model")
        content = workspace.read_text(arguments.path)
        return ToolResult(
            status="success",
            content=content,
            data={"path": arguments.path, "characters": len(content)},
        )
