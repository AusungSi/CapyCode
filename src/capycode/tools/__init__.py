"""Model-callable tool contracts and implementations."""

from .base import ToolExecutor, ToolRegistry, ToolResult
from .list_files import ListFilesTool
from .read_file import ReadFileTool
from .search_code import SearchCodeTool


def build_p0_runtime_tools() -> ToolRegistry:
    return ToolRegistry([ListFilesTool(), ReadFileTool(), SearchCodeTool()])


__all__ = [
    "ListFilesTool",
    "ReadFileTool",
    "SearchCodeTool",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "build_p0_runtime_tools",
]
