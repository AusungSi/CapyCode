from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult


class ReadFileInput(ToolInput):
    path: str = Field(min_length=1, description="Workspace-relative file path")
    start_line: int | None = Field(default=None, ge=1, description="First 1-based line to return")
    end_line: int | None = Field(
        default=None, ge=1, description="Last 1-based line to return (inclusive)"
    )
    offset: int | None = Field(
        default=None, ge=0, description="Zero-based character offset (compatibility alias)"
    )
    limit: int | None = Field(
        default=None, gt=0, description="Maximum characters after offset (compatibility alias)"
    )

    @model_validator(mode="after")
    def validate_range(self) -> ReadFileInput:
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ReadFileTool(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a UTF-8 text file inside the current workspace. Use start_line/end_line "
        "for a focused range; offset/limit are accepted as compatibility aliases."
    )
    input_model: ClassVar[type[ToolInput]] = ReadFileInput

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, ReadFileInput):
            raise TypeError("read_file received an unexpected argument model")
        content = workspace.read_text(arguments.path)
        original_characters = len(content)
        range_applied = False
        if arguments.start_line is not None or arguments.end_line is not None:
            lines = content.splitlines(keepends=True)
            first = (arguments.start_line or 1) - 1
            last = arguments.end_line if arguments.end_line is not None else len(lines)
            content = "".join(lines[first:last])
            range_applied = True
        if arguments.offset is not None:
            content = content[arguments.offset :]
            range_applied = True
        if arguments.limit is not None:
            content = content[: arguments.limit]
            range_applied = True
        record = workspace.read_ledger.get(workspace.resolve_file(arguments.path))
        if record is None:
            raise RuntimeError("read ledger was not updated")
        return ToolResult(
            status="success",
            content=content,
            data={
                "path": arguments.path,
                "characters": len(content),
                "original_characters": original_characters,
                "range_applied": range_applied,
                "digest": record.digest,
                "encoding": record.encoding,
                "newline": record.newline,
            },
        )
