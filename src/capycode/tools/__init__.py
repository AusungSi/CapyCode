"""Model-callable tool contracts and implementations."""

from .base import ToolExecutor, ToolRegistry, ToolResult
from .list_files import ListFilesTool
from .read_file import ReadFileTool
from .replace_text import ReplaceTextTool
from .search_code import SearchCodeTool
from .write_file import WriteFileTool


def build_p0_runtime_tools() -> ToolRegistry:
    return ToolRegistry(
        [ListFilesTool(), ReadFileTool(), ReplaceTextTool(), SearchCodeTool(), WriteFileTool()]
    )


__all__ = [
    "ListFilesTool",
    "ReadFileTool",
    "ReplaceTextTool",
    "SearchCodeTool",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "WriteFileTool",
    "build_p0_runtime_tools",
]
