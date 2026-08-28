from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult


class WriteFileInput(ToolInput):
    path: str = Field(min_length=1, description="Workspace-relative file path")
    content: str = Field(description="Complete replacement content")


class WriteFileTool(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create a text file or fully replace a previously read file using an atomic write."
    )
    input_model: ClassVar[type[ToolInput]] = WriteFileInput

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, WriteFileInput):
            raise TypeError("write_file received an unexpected argument model")
        existed = workspace.resolve_new_file(arguments.path).exists()
        record = workspace.write_text(arguments.path, arguments.content)
        return ToolResult(
            status="success",
            content=f"{'Updated' if existed else 'Created'} {arguments.path}",
            data={
                "path": arguments.path,
                "created": not existed,
                "bytes": record.size_bytes,
                "digest": record.digest,
                "newline": record.newline,
            },
        )
