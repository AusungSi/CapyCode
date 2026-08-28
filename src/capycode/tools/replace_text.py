from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace

from .base import Tool, ToolInput, ToolResult


class ReplaceTextInput(ToolInput):
    path: str = Field(min_length=1, description="Workspace-relative file path")
    old_text: str = Field(min_length=1, description="Exact text to replace")
    new_text: str = Field(description="Replacement text")
    replace_all: bool = False


class ReplaceTextTool(Tool):
    name: ClassVar[str] = "replace_text"
    description: ClassVar[str] = (
        "Replace exact text in a previously read file; multiple matches require replace_all=true."
    )
    input_model: ClassVar[type[ToolInput]] = ReplaceTextInput

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, ReplaceTextInput):
            raise TypeError("replace_text received an unexpected argument model")
        record, replacements = workspace.replace_text(
            arguments.path,
            arguments.old_text,
            arguments.new_text,
            replace_all=arguments.replace_all,
        )
        return ToolResult(
            status="success",
            content=f"Updated {arguments.path}: {replacements} replacement(s)",
            data={
                "path": arguments.path,
                "replacements": replacements,
                "bytes": record.size_bytes,
                "digest": record.digest,
                "newline": record.newline,
            },
        )
