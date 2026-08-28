from __future__ import annotations

import re
from typing import ClassVar

from pydantic import Field

from capycode.workspace import LocalWorkspace, WorkspaceError

from .base import Tool, ToolInput, ToolResult


class SearchCodeInput(ToolInput):
    query: str = Field(min_length=1, max_length=500)
    path: str = Field(default=".", min_length=1, description="Workspace-relative file or directory")
    pattern: str = Field(default="*", min_length=1, description="File glob")
    regex: bool = False
    case_sensitive: bool = False
    max_results: int = Field(default=200, ge=1, le=1_000)


class SearchCodeTool(Tool):
    name: ClassVar[str] = "search_code"
    description: ClassVar[str] = (
        "Search text in a workspace file or directory and return path, line, and preview."
    )
    input_model: ClassVar[type[ToolInput]] = SearchCodeInput

    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        if not isinstance(arguments, SearchCodeInput):
            raise TypeError("search_code received an unexpected argument model")
        flags = 0 if arguments.case_sensitive else re.IGNORECASE
        expression = arguments.query if arguments.regex else re.escape(arguments.query)
        try:
            matcher = re.compile(expression, flags)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc

        matches: list[str] = []
        skipped = 0
        files = workspace.iter_files(arguments.path, pattern=arguments.pattern, max_results=5_000)
        for relative_path in files:
            try:
                content = workspace.read_text_for_search(relative_path)
            except WorkspaceError:
                skipped += 1
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if matcher.search(line):
                    preview = line.strip()
                    if len(preview) > 300:
                        preview = preview[:297] + "..."
                    matches.append(f"{relative_path}:{line_number}: {preview}")
                    if len(matches) >= arguments.max_results:
                        break
            if len(matches) >= arguments.max_results:
                break

        return ToolResult(
            status="success",
            content="\n".join(matches) if matches else "No matches found.",
            data={"matches": len(matches), "files_scanned": len(files), "files_skipped": skipped},
        )
