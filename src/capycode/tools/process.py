from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from capycode.workspace import LocalWorkspace, WorkspaceError

BLOCKED_EXECUTABLES = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "diskpart",
        "fish",
        "format",
        "powershell",
        "powershell.exe",
        "pwsh",
        "reboot",
        "reg",
        "rm",
        "sh",
        "shutdown",
        "wsl",
        "zsh",
    }
)
READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {"--version", "diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}
)
SENSITIVE_ENV_MARKERS = ("API_KEY", "AUTHORIZATION", "PASSWORD", "SECRET", "TOKEN")
SHELL_OPERATOR_ARGUMENTS = frozenset({"|", "||", "&&", ";", "<", ">", ">>", "2>", "2>&1"})


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    cwd: Path
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool
    duration_seconds: float
    background_task_id: str | None = None
    running: bool = False


@dataclass
class BackgroundProcess:
    task_id: str
    process: asyncio.subprocess.Process
    argv: tuple[str, ...]
    cwd: Path
    started: float
    temporary_directory: tempfile.TemporaryDirectory[str]
    stdout_task: asyncio.Task[tuple[str, bool]]
    stderr_task: asyncio.Task[tuple[str, bool]]


class CommandRunner:
    def __init__(
        self,
        *,
        allowed_executables: frozenset[str] | None = None,
        max_stream_characters: int = 20_000,
        container_image: str | None = None,
    ) -> None:
        if allowed_executables is not None and not allowed_executables:
            raise ValueError("allowed_executables must not be empty")
        if max_stream_characters < 128:
            raise ValueError("max_stream_characters must be at least 128")
        self.allowed_executables = (
            frozenset(name.casefold() for name in allowed_executables)
            if allowed_executables is not None
            else None
        )
        self.max_stream_characters = max_stream_characters
        self.container_image = container_image
        self._background: dict[str, BackgroundProcess] = {}
        self._completed: dict[str, ProcessResult] = {}

    async def run(
        self,
        workspace: LocalWorkspace,
        argv: list[str],
        *,
        cwd: str = ".",
        timeout_seconds: float = 120,
        run_in_background: bool = False,
    ) -> ProcessResult:
        if not argv or any(not item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        executable = argv[0] if self.container_image else self._resolve_executable(argv[0])
        if self.container_image:
            # Models sometimes echo the container-visible cwd. Translate it
            # back to the mounted instance workspace before boundary checks.
            normalized_cwd = cwd.replace("\\", "/")
            if normalized_cwd == "/workspace" or normalized_cwd.startswith("/workspace/"):
                normalized_cwd = normalized_cwd[len("/workspace") :].lstrip("/") or "."
                if normalized_cwd == workspace.root.name or normalized_cwd.startswith(
                    workspace.root.name + "/"
                ):
                    normalized_cwd = normalized_cwd[len(workspace.root.name) :].lstrip("/") or "."
                cwd = normalized_cwd
        working_directory = workspace.resolve_directory(cwd)
        self._validate_arguments(
            argv, workspace, working_directory, allow_container_paths=bool(self.container_image)
        )
        if self.container_image:
            docker = shutil.which("docker")
            if docker is None:
                raise WorkspaceError(
                    "Docker is required for this benchmark environment but was not found on PATH"
                )
            relative_cwd = working_directory.relative_to(workspace.root).as_posix()
            container_cwd = "/workspace" if relative_cwd == "." else f"/workspace/{relative_cwd}"
            command = [
                docker,
                "run",
                "--rm",
                "--init",
                "--pull=never",
                "--network=bridge",
                "-v",
                f"{workspace.root}:/workspace",
                "-w",
                container_cwd,
                self.container_image,
                *argv,
            ]
        else:
            command = [executable, *argv[1:]]
        temporary_directory = tempfile.TemporaryDirectory(prefix="capycode-process-")
        environment = self._sanitized_environment()
        environment["PYTHONPYCACHEPREFIX"] = (
            "/tmp/capycode-pycache" if self.container_image else temporary_directory.name
        )
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=working_directory,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except BaseException:
            temporary_directory.cleanup()
            raise
        stdout_task = asyncio.create_task(self._read_stream(process.stdout))
        stderr_task = asyncio.create_task(self._read_stream(process.stderr))

        if run_in_background:
            task_id = self._new_task_id()
            self._background[task_id] = BackgroundProcess(
                task_id=task_id,
                process=process,
                argv=tuple(argv),
                cwd=working_directory,
                started=started,
                temporary_directory=temporary_directory,
                stdout_task=stdout_task,
                stderr_task=stderr_task,
            )
            return ProcessResult(
                argv=tuple(argv),
                cwd=working_directory,
                exit_code=None,
                stdout="",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                timed_out=False,
                duration_seconds=time.monotonic() - started,
                background_task_id=task_id,
                running=True,
            )

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except TimeoutError:
            timed_out = True
            process.kill()
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            stdout_task.cancel()
            stderr_task.cancel()
            temporary_directory.cleanup()
            raise
        stdout, stdout_truncated = await stdout_task
        stderr, stderr_truncated = await stderr_task
        temporary_directory.cleanup()
        return ProcessResult(
            argv=tuple(argv),
            cwd=working_directory,
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
        )

    async def inspect_background(self, task_id: str) -> ProcessResult:
        completed = self._completed.get(task_id)
        if completed is not None:
            return completed
        task = self._background.get(task_id)
        if task is None:
            raise ValueError(f"unknown background task: {task_id}")
        if task.process.returncode is None:
            return self._running_result(task)
        return await self._finalize_background(task)

    async def stop_background(self, task_id: str) -> ProcessResult:
        completed = self._completed.get(task_id)
        if completed is not None:
            return completed
        task = self._background.get(task_id)
        if task is None:
            raise ValueError(f"unknown background task: {task_id}")
        if task.process.returncode is None:
            task.process.kill()
            await task.process.wait()
        return await self._finalize_background(task)

    async def aclose(self) -> None:
        for task_id in list(self._background):
            await self.stop_background(task_id)

    def _resolve_executable(self, requested: str) -> str:
        if Path(requested).name != requested or PureWindowsPath(requested).name != requested:
            raise WorkspaceError("executable must be a configured program name, not a path")
        normalized = requested.casefold()
        if normalized in BLOCKED_EXECUTABLES:
            raise WorkspaceError(
                f"executable is blocked by command policy because it starts a secondary shell "
                f"or high-risk system operation: {requested}"
            )
        if self.allowed_executables is not None and normalized not in self.allowed_executables:
            choices = ", ".join(sorted(self.allowed_executables))
            raise WorkspaceError(f"executable is not allowed: {requested}; allowed: {choices}")
        # Test and development commands must use the interpreter running CapyCode.
        # Resolving ``python`` through PATH can silently select a different Conda install.
        if normalized in {"py", "python", "python3", "python.exe", "python3.exe"}:
            return sys.executable
        resolved = shutil.which(requested)
        if resolved is None:
            raise WorkspaceError(f"allowed executable was not found on PATH: {requested}")
        return resolved

    @staticmethod
    def _validate_arguments(
        argv: list[str],
        workspace: LocalWorkspace,
        working_directory: Path,
        *,
        allow_container_paths: bool = False,
    ) -> None:
        executable = argv[0].casefold()
        if executable == "git":
            subcommand = argv[1].casefold() if len(argv) > 1 else ""
            if subcommand not in READ_ONLY_GIT_SUBCOMMANDS:
                raise WorkspaceError(
                    "generic git execution is limited to read-only subcommands; "
                    "use dedicated tools for repository changes"
                )
        skip_argument_indexes: set[int] = set()
        if executable in {"py", "python", "python3"}:
            for index, argument in enumerate(argv[:-1]):
                if argument == "-c":
                    skip_argument_indexes.add(index + 1)
        if executable in {"node", "deno", "bun"}:
            for index, argument in enumerate(argv[:-1]):
                if argument in {"-e", "--eval"}:
                    skip_argument_indexes.add(index + 1)
        for index, argument in enumerate(argv[1:], start=1):
            if index in skip_argument_indexes:
                continue
            if argument in SHELL_OPERATOR_ARGUMENTS:
                raise WorkspaceError(
                    "run_command executes an argv array without a shell; shell operators such as "
                    f"{argument!r} are not supported. Run separate commands or use a safe "
                    "program option instead."
                )
            candidate = (
                argument.split("=", 1)[1]
                if argument.startswith("-") and "=" in argument
                else argument
            )
            if not CommandRunner._looks_like_path(candidate):
                continue
            if allow_container_paths and candidate.replace("\\", "/").startswith("/workspace/"):
                continue
            windows_path = PureWindowsPath(candidate)
            path = Path(candidate)
            if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
                raise WorkspaceError(f"command path arguments must be relative: {candidate}")
            resolved = (working_directory / path).resolve(strict=False)
            if not resolved.is_relative_to(workspace.root):
                raise WorkspaceError(f"command path argument escapes the workspace: {candidate}")

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return (
            value in {".", ".."}
            or value.startswith(("./", "../", ".\\", "..\\", "/", "\\"))
            or "\\" in value
            or (
                "/" in value
                and not value.startswith("http://")
                and not value.startswith("https://")
            )
            or bool(PureWindowsPath(value).drive)
        )

    @staticmethod
    def _sanitized_environment() -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not any(marker in key.upper() for marker in SENSITIVE_ENV_MARKERS)
        }
        environment["NO_COLOR"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        return environment

    async def _read_stream(self, reader: asyncio.StreamReader | None) -> tuple[str, bool]:
        if reader is None:
            return "", False
        marker = b"\n... output truncated ...\n"
        byte_limit = self.max_stream_characters
        available = byte_limit - len(marker)
        head_limit = available // 2
        tail_limit = available - head_limit
        complete = bytearray()
        head = bytearray()
        tail = bytearray()
        truncated = False
        while chunk := await reader.read(8_192):
            if not truncated:
                complete.extend(chunk)
                if len(complete) <= byte_limit:
                    continue
                truncated = True
                head.extend(complete[:head_limit])
                tail.extend(complete[-tail_limit:])
                complete.clear()
            else:
                tail.extend(chunk)
                if len(tail) > tail_limit:
                    del tail[:-tail_limit]
        raw = bytes(head + marker + tail) if truncated else bytes(complete)
        return raw.decode("utf-8", errors="replace"), truncated

    async def _finalize_background(self, task: BackgroundProcess) -> ProcessResult:
        stdout, stdout_truncated = await task.stdout_task
        stderr, stderr_truncated = await task.stderr_task
        result = ProcessResult(
            argv=task.argv,
            cwd=task.cwd,
            exit_code=task.process.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=False,
            duration_seconds=time.monotonic() - task.started,
            background_task_id=task.task_id,
            running=False,
        )
        task.temporary_directory.cleanup()
        self._background.pop(task.task_id, None)
        self._completed[task.task_id] = result
        return result

    @staticmethod
    def _running_result(task: BackgroundProcess) -> ProcessResult:
        return ProcessResult(
            argv=task.argv,
            cwd=task.cwd,
            exit_code=None,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            timed_out=False,
            duration_seconds=time.monotonic() - task.started,
            background_task_id=task.task_id,
            running=True,
        )

    def _new_task_id(self) -> str:
        while True:
            task_id = secrets.token_hex(6)
            if task_id not in self._background and task_id not in self._completed:
                return task_id
