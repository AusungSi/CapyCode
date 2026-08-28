"""Model-callable tool contracts and implementations."""

from .base import ToolExecutor, ToolRegistry, ToolResult
from .git_diff import GitDiffTool
from .list_files import ListFilesTool
from .process import CommandRunner, ProcessResult
from .read_file import ReadFileTool
from .replace_text import ReplaceTextTool
from .run_command import RunCommandTool
from .run_tests import RunTestsTool
from .search_code import SearchCodeTool
from .write_file import WriteFileTool


def build_p0_runtime_tools() -> ToolRegistry:
    return ToolRegistry(
        [
            GitDiffTool(),
            ListFilesTool(),
            ReadFileTool(),
            ReplaceTextTool(),
            RunCommandTool(),
            RunTestsTool(),
            SearchCodeTool(),
            WriteFileTool(),
        ]
    )


__all__ = [
    "CommandRunner",
    "GitDiffTool",
    "ListFilesTool",
    "ProcessResult",
    "ReadFileTool",
    "ReplaceTextTool",
    "RunCommandTool",
    "RunTestsTool",
    "SearchCodeTool",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "WriteFileTool",
    "build_p0_runtime_tools",
]
