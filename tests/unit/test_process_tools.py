from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from capycode.tools import (
    CommandRunner,
    GitDiffTool,
    RunCommandTool,
    RunTestsTool,
    ToolExecutor,
    ToolRegistry,
)
from capycode.workspace import LocalWorkspace


def python_name() -> str:
    return "python"


def initialize_git_fixture(directory: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
    path = directory / "tracked.txt"
    path.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=directory, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CapyCode Test",
            "-c",
            "user.email=capycode@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=directory,
        check=True,
    )
    path.write_text("after\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_run_command_executes_argument_array_without_shell(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("print('hello from child')\n", encoding="utf-8")
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("run_command", {"argv": [python_name(), "hello.py"]})

    assert result.status == "success"
    assert "hello from child" in result.content
    assert result.data["exit_code"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("executable", ["bash", "cmd", "git", "powershell", "pwsh"])
async def test_run_command_rejects_non_allowlisted_programs(
    tmp_path: Path, executable: str
) -> None:
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("run_command", {"argv": [executable, "--version"]})

    assert result.status == "error"
    assert "not allowed" in result.content


@pytest.mark.asyncio
async def test_run_command_rejects_path_argument_outside_workspace(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute(
        "run_command", {"argv": [python_name(), "../outside/script.py"]}
    )

    assert result.status == "error"
    assert "escapes the workspace" in result.content


@pytest.mark.asyncio
async def test_run_command_rejects_inline_python(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute(
        "run_command", {"argv": [python_name(), "-c", "print('unsafe')"]}
    )

    assert result.status == "error"
    assert "inline Python" in result.content


@pytest.mark.asyncio
async def test_run_command_rejects_cwd_outside_workspace(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute(
        "run_command",
        {"argv": [python_name(), "--version"], "cwd": ".."},
    )

    assert result.status == "error"
    assert "escapes the workspace" in result.content


@pytest.mark.asyncio
async def test_child_process_does_not_receive_sensitive_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "environment.py"
    script.write_text(
        "import os\nprint(os.getenv('CAPYCODE_TEST_API_KEY', 'missing'))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPYCODE_TEST_API_KEY", "must-not-leak")
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("run_command", {"argv": [python_name(), "environment.py"]})

    assert result.status == "success"
    assert "missing" in result.content
    assert "must-not-leak" not in result.content


@pytest.mark.asyncio
async def test_command_timeout_terminates_process(tmp_path: Path) -> None:
    (tmp_path / "sleep.py").write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute(
        "run_command",
        {"argv": [python_name(), "sleep.py"], "timeout_seconds": 1},
    )

    assert result.status == "error"
    assert result.data["timed_out"] is True
    assert "timed out" in result.content


@pytest.mark.asyncio
async def test_command_output_preserves_head_and_tail(tmp_path: Path) -> None:
    (tmp_path / "output.py").write_text("print('A' * 300 + 'B' * 300)\n", encoding="utf-8")
    runner = CommandRunner(max_stream_characters=128)
    executor = ToolExecutor(ToolRegistry([RunCommandTool(runner)]), LocalWorkspace(tmp_path))

    result = await executor.execute("run_command", {"argv": [python_name(), "output.py"]})

    assert result.status == "success"
    assert "output truncated" in result.content
    assert "AAAA" in result.content
    assert "BBBB" in result.content
    assert result.data["stdout_truncated"] is True


@pytest.mark.asyncio
async def test_run_tests_reports_failure_then_success(tmp_path: Path) -> None:
    test_file = tmp_path / "test_demo.py"
    test_file.write_text("def test_demo():\n    assert False\n", encoding="utf-8")
    executor = ToolExecutor(ToolRegistry([RunTestsTool()]), LocalWorkspace(tmp_path))

    failed = await executor.execute("run_tests", {"argv": [python_name(), "-m", "pytest", "-q"]})
    test_file.write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    passed = await executor.execute("run_tests", {"argv": [python_name(), "-m", "pytest", "-q"]})

    assert failed.status == "error"
    assert failed.data["tests_passed"] is False
    assert passed.status == "success"
    assert passed.data["tests_passed"] is True


@pytest.mark.asyncio
async def test_git_diff_is_read_only_and_returns_workspace_change(tmp_path: Path) -> None:
    initialize_git_fixture(tmp_path)
    executor = ToolExecutor(ToolRegistry([GitDiffTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("git_diff", {"path": "tracked.txt"})

    assert result.status == "success"
    assert "-before" in result.content
    assert "+after" in result.content
