"""Model-callable tool contracts and implementations."""

from .base import ToolExecutor, ToolRegistry, ToolResult
from .git_diff import GitDiffTool
from .list_files import ListFilesTool
from .process import CommandRunner, ProcessResult
from .process_status import ProcessStatusTool
from .read_file import ReadFileTool
from .replace_text import ReplaceTextTool
from .run_command import RunCommandTool
from .run_tests import RunTestsTool
from .search_code import SearchCodeTool
from .stop_process import StopProcessTool
from .write_file import WriteFileTool


def build_p0_runtime_tools() -> ToolRegistry:
    command_runner = CommandRunner()
    git_runner = CommandRunner(allowed_executables=frozenset({"git"}))
    return ToolRegistry(
        [
            GitDiffTool(git_runner),
            ListFilesTool(),
            ProcessStatusTool(command_runner),
            ReadFileTool(),
            ReplaceTextTool(),
            RunCommandTool(command_runner),
            RunTestsTool(command_runner),
            SearchCodeTool(),
            StopProcessTool(command_runner),
            WriteFileTool(),
        ]
    )


__all__ = [
    "CommandRunner",
    "GitDiffTool",
    "ListFilesTool",
    "ProcessResult",
    "ProcessStatusTool",
    "ReadFileTool",
    "ReplaceTextTool",
    "RunCommandTool",
    "RunTestsTool",
    "SearchCodeTool",
    "StopProcessTool",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "WriteFileTool",
    "build_p0_runtime_tools",
]
