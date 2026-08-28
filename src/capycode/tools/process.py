from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from capycode.workspace import LocalWorkspace, WorkspaceError

DEFAULT_ALLOWED_EXECUTABLES = frozenset({"py", "pytest", "python", "python3", "rg", "uv"})
SENSITIVE_ENV_MARKERS = ("API_KEY", "AUTHORIZATION", "PASSWORD", "SECRET", "TOKEN")


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    cwd: Path
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool
    duration_seconds: float


class CommandRunner:
    def __init__(
        self,
        *,
        allowed_executables: frozenset[str] = DEFAULT_ALLOWED_EXECUTABLES,
        max_stream_characters: int = 20_000,
    ) -> None:
        if not allowed_executables:
            raise ValueError("allowed_executables must not be empty")
        if max_stream_characters < 128:
            raise ValueError("max_stream_characters must be at least 128")
        self.allowed_executables = frozenset(name.casefold() for name in allowed_executables)
        self.max_stream_characters = max_stream_characters

    async def run(
        self,
        workspace: LocalWorkspace,
        argv: list[str],
        *,
        cwd: str = ".",
        timeout_seconds: float = 120,
    ) -> ProcessResult:
        if not argv or any(not item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        executable = self._resolve_executable(argv[0])
        working_directory = workspace.resolve_directory(cwd)
        self._validate_arguments(argv, workspace, working_directory)
        with tempfile.TemporaryDirectory(prefix="capycode-pycache-") as pycache_directory:
            environment = self._sanitized_environment()
            environment["PYTHONPYCACHEPREFIX"] = pycache_directory

            started = time.monotonic()
            process = await asyncio.create_subprocess_exec(
                executable,
                *argv[1:],
                cwd=working_directory,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            timed_out = False
            stdout_task = asyncio.create_task(self._read_stream(process.stdout))
            stderr_task = asyncio.create_task(self._read_stream(process.stderr))
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
                raise
            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task

        return ProcessResult(
            argv=tuple(argv),
            cwd=working_directory,
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
        )

    def _resolve_executable(self, requested: str) -> str:
        if Path(requested).name != requested or PureWindowsPath(requested).name != requested:
            raise WorkspaceError("executable must be a name from the allowlist, not a path")
        if requested.casefold() not in self.allowed_executables:
            choices = ", ".join(sorted(self.allowed_executables))
            raise WorkspaceError(f"executable is not allowed: {requested}; allowed: {choices}")
        resolved = shutil.which(requested)
        if resolved is None:
            raise WorkspaceError(f"allowed executable was not found on PATH: {requested}")
        return resolved

    @staticmethod
    def _validate_arguments(
        argv: list[str], workspace: LocalWorkspace, working_directory: Path
    ) -> None:
        if argv[0].casefold() in {"py", "python", "python3"} and "-c" in argv[1:]:
            raise WorkspaceError("inline Python with -c is not allowed")
        for argument in argv[1:]:
            candidate = (
                argument.split("=", 1)[1]
                if argument.startswith("-") and "=" in argument
                else argument
            )
            if not CommandRunner._looks_like_path(candidate):
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
