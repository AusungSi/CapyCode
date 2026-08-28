"""Model-callable tool contracts and implementations."""

from .base import ToolExecutor, ToolRegistry, ToolResult
from .read_file import ReadFileTool


def build_p0_runtime_tools() -> ToolRegistry:
    return ToolRegistry([ReadFileTool()])


__all__ = [
    "ReadFileTool",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "build_p0_runtime_tools",
]
