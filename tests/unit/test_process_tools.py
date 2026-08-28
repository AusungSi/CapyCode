from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from capycode.tools import (
    CommandRunner,
    GitDiffTool,
    ProcessStatusTool,
    RunCommandTool,
    RunTestsTool,
    StopProcessTool,
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
@pytest.mark.parametrize("executable", ["bash", "cmd", "powershell", "pwsh"])
async def test_run_command_rejects_non_allowlisted_programs(
    tmp_path: Path, executable: str
) -> None:
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("run_command", {"argv": [executable, "--version"]})

    assert result.status == "error"
    assert "blocked by command policy" in result.content


@pytest.mark.asyncio
async def test_run_command_rejects_path_argument_outside_workspace(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute(
        "run_command", {"argv": [python_name(), "../outside/script.py"]}
    )

    assert result.status == "error"
    assert "escapes the workspace" in result.content


@pytest.mark.asyncio
async def test_run_command_allows_inline_python_for_one_shot_checks(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute(
        "run_command", {"argv": [python_name(), "-c", "print('inline-ok')"]}
    )

    assert result.status == "success"
    assert "inline-ok" in result.content


@pytest.mark.asyncio
async def test_run_command_allows_read_only_git_and_rejects_mutating_git(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([RunCommandTool()]), LocalWorkspace(tmp_path))

    version = await executor.execute("run_command", {"argv": ["git", "--version"]})
    mutation = await executor.execute("run_command", {"argv": ["git", "reset", "--hard"]})

    assert version.status == "success"
    assert "git version" in version.content
    assert mutation.status == "error"
    assert "read-only subcommands" in mutation.content


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
async def test_background_server_can_be_checked_and_stopped(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("background-ok", encoding="utf-8")
    (tmp_path / "server.py").write_text(
        "from http.server import HTTPServer, SimpleHTTPRequestHandler\n"
        "from pathlib import Path\n"
        "server = HTTPServer(('127.0.0.1', 0), SimpleHTTPRequestHandler)\n"
        "Path('port.txt').write_text(str(server.server_port), encoding='utf-8')\n"
        "server.serve_forever()\n",
        encoding="utf-8",
    )
    runner = CommandRunner()
    registry = ToolRegistry(
        [RunCommandTool(runner), ProcessStatusTool(runner), StopProcessTool(runner)]
    )
    executor = ToolExecutor(registry, LocalWorkspace(tmp_path))

    started = await executor.execute(
        "run_command",
        {"argv": [python_name(), "server.py"], "run_in_background": True},
    )
    task_id = str(started.data["background_task_id"])
    for _ in range(40):
        if (tmp_path / "port.txt").exists():
            break
        await asyncio.sleep(0.05)
    port = (tmp_path / "port.txt").read_text(encoding="utf-8")

    status = await executor.execute("process_status", {"task_id": task_id})
    checked = await executor.execute(
        "run_command",
        {
            "argv": [
                python_name(),
                "-c",
                (
                    "import urllib.request; "
                    f"print(urllib.request.urlopen('http://127.0.0.1:{port}/index.html').read().decode())"
                ),
            ]
        },
    )
    stopped = await executor.execute("stop_process", {"task_id": task_id})
    await registry.aclose()

    assert started.status == "success"
    assert started.data["running"] is True
    assert status.status == "success"
    assert status.data["running"] is True
    assert checked.status == "success"
    assert "background-ok" in checked.content
    assert stopped.status == "success"
    assert stopped.data["running"] is False


@pytest.mark.asyncio
async def test_command_output_preserves_head_and_tail(tmp_path: Path) -> None:
    (tmp_path / "output.py").write_text(
        "print('A' * 1_000_000 + 'B' * 1_000_000)\n", encoding="utf-8"
    )
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


@pytest.mark.asyncio
async def test_git_diff_is_gracefully_unavailable_outside_repository(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([GitDiffTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("git_diff", {})

    assert result.status == "success"
    assert result.data["available"] is False
    assert "not a Git repository" in result.content
