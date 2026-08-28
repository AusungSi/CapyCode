from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath
from typing import Literal

EncodingName = Literal["utf-8", "utf-8-sig"]
NewlineStyle = Literal["lf", "crlf", "cr", "mixed", "none"]


class WorkspaceError(ValueError):
    pass


@dataclass(frozen=True)
class FileReadRecord:
    path: Path
    digest: str
    modified_ns: int
    size_bytes: int
    complete: bool
    encoding: EncodingName
    newline: NewlineStyle


class FileReadLedger:
    def __init__(self) -> None:
        self._records: dict[Path, FileReadRecord] = {}

    def record(self, entry: FileReadRecord) -> None:
        self._records[entry.path] = entry

    def get(self, path: Path) -> FileReadRecord | None:
        return self._records.get(path)

    def paths(self) -> tuple[Path, ...]:
        return tuple(self._records)


class LocalWorkspace:
    DEFAULT_IGNORED_DIRECTORIES = frozenset(
        {
            ".capy",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "node_modules",
        }
    )

    def __init__(self, root: Path, *, max_read_bytes: int = 1_000_000) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace is not a directory: {self.root}")
        if max_read_bytes <= 0:
            raise ValueError("max_read_bytes must be positive")
        self.max_read_bytes = max_read_bytes
        self.read_ledger = FileReadLedger()

    def resolve_file(self, relative_path: str) -> Path:
        resolved = self._resolve(relative_path, must_exist=True)
        if not resolved.is_file():
            raise WorkspaceError(f"path is not a file: {relative_path}")
        return resolved

    def resolve_directory(self, relative_path: str = ".") -> Path:
        resolved = self._resolve(relative_path, must_exist=True)
        if not resolved.is_dir():
            raise WorkspaceError(f"path is not a directory: {relative_path}")
        return resolved

    def resolve_new_file(self, relative_path: str) -> Path:
        resolved = self._resolve(relative_path, must_exist=False)
        if resolved == self.root:
            raise WorkspaceError("file path cannot be the workspace root")
        if resolved.exists() and not resolved.is_file():
            raise WorkspaceError(f"path is not a file: {relative_path}")
        if not resolved.parent.exists():
            raise WorkspaceError(f"parent directory not found: {relative_path}")
        if not resolved.parent.is_dir():
            raise WorkspaceError(f"parent path is not a directory: {relative_path}")
        return resolved

    def read_text(self, relative_path: str) -> str:
        path = self.resolve_file(relative_path)
        raw = self._read_bytes(path, relative_path)
        content, encoding = self._decode_text(raw, relative_path)
        stat = path.stat()
        self.read_ledger.record(
            FileReadRecord(
                path=path,
                digest=self._digest(raw),
                modified_ns=stat.st_mtime_ns,
                size_bytes=len(raw),
                complete=True,
                encoding=encoding,
                newline=self._detect_newline(content),
            )
        )
        return content

    def read_text_for_search(self, relative_path: str) -> str:
        path = self.resolve_file(relative_path)
        raw = self._read_bytes(path, relative_path)
        content, _ = self._decode_text(raw, relative_path)
        return content

    def iter_files(
        self,
        relative_path: str = ".",
        *,
        pattern: str = "*",
        max_results: int = 1_000,
    ) -> list[str]:
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        directory = self.resolve_directory(relative_path)
        matches: list[str] = []
        candidates = sorted(directory.rglob("*"), key=lambda path: path.as_posix())
        for candidate in candidates:
            relative_candidate = candidate.relative_to(self.root)
            if any(part in self.DEFAULT_IGNORED_DIRECTORIES for part in relative_candidate.parts):
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if not resolved.is_relative_to(self.root) or not resolved.is_file():
                continue
            relative = resolved.relative_to(self.root).as_posix()
            if not PurePath(relative).match(pattern):
                continue
            matches.append(relative)
            if len(matches) >= max_results:
                break
        return sorted(set(matches))

    def require_fresh_read(self, relative_path: str) -> FileReadRecord:
        path = self.resolve_file(relative_path)
        record = self.read_ledger.get(path)
        if record is None or not record.complete:
            raise WorkspaceError(f"file must be fully read before modification: {relative_path}")

        stat = path.stat()
        if stat.st_mtime_ns != record.modified_ns or stat.st_size != record.size_bytes:
            raise WorkspaceError(f"file changed since it was read; read it again: {relative_path}")
        current = self._read_bytes(path, relative_path)
        if self._digest(current) != record.digest:
            raise WorkspaceError(f"file changed since it was read; read it again: {relative_path}")
        return record

    def _resolve(self, relative_path: str, *, must_exist: bool) -> Path:
        requested = self._validate_relative_path(relative_path)
        candidate = (self.root / requested).resolve(strict=False)
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError("path escapes the workspace")
        if must_exist and not candidate.exists():
            raise WorkspaceError(f"path not found: {relative_path}")
        return candidate

    @staticmethod
    def _validate_relative_path(relative_path: str) -> Path:
        if not relative_path or "\x00" in relative_path:
            raise WorkspaceError("workspace path must not be empty")
        windows_path = PureWindowsPath(relative_path)
        if windows_path.drive.startswith("\\\\"):
            raise WorkspaceError("UNC network paths are not allowed")
        requested = Path(relative_path)
        if requested.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise WorkspaceError("workspace paths must be relative")
        return requested

    def _read_bytes(self, path: Path, display_path: str) -> bytes:
        size = path.stat().st_size
        if size > self.max_read_bytes:
            raise WorkspaceError(
                f"file exceeds read limit: {size} bytes > {self.max_read_bytes} bytes"
            )
        try:
            return path.read_bytes()
        except OSError as exc:
            raise WorkspaceError(f"unable to read file: {display_path}: {exc}") from exc

    @staticmethod
    def _decode_text(raw: bytes, display_path: str) -> tuple[str, EncodingName]:
        encoding: EncodingName = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"file is not valid UTF-8 text: {display_path}") from exc

    @staticmethod
    def _digest(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _detect_newline(content: str) -> NewlineStyle:
        crlf_count = content.count("\r\n")
        without_crlf = content.replace("\r\n", "")
        lf_count = without_crlf.count("\n")
        cr_count = without_crlf.count("\r")
        present = sum(count > 0 for count in (crlf_count, lf_count, cr_count))
        if present > 1:
            return "mixed"
        if crlf_count:
            return "crlf"
        if lf_count:
            return "lf"
        if cr_count:
            return "cr"
        return "none"
