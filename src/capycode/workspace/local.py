from __future__ import annotations

import hashlib
import os
import tempfile
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

    def __init__(
        self,
        root: Path,
        *,
        max_read_bytes: int = 1_000_000,
        max_write_bytes: int = 1_000_000,
    ) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace is not a directory: {self.root}")
        if max_read_bytes <= 0:
            raise ValueError("max_read_bytes must be positive")
        if max_write_bytes <= 0:
            raise ValueError("max_write_bytes must be positive")
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes
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
        directory_parts = directory.relative_to(self.root).parts
        if any(part in self.DEFAULT_IGNORED_DIRECTORIES for part in directory_parts):
            raise WorkspaceError(f"directory is excluded from discovery: {relative_path}")
        matches: list[str] = []
        for current, directories, filenames in os.walk(directory, followlinks=False):
            directories[:] = sorted(
                name for name in directories if name not in self.DEFAULT_IGNORED_DIRECTORIES
            )
            for filename in sorted(filenames):
                candidate = Path(current) / filename
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
                    return sorted(set(matches))
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

    def write_text(self, relative_path: str, content: str) -> FileReadRecord:
        path = self.resolve_new_file(relative_path)
        previous = self.require_fresh_read(relative_path) if path.exists() else None
        prepared = self._preserve_newlines(content, previous.newline if previous else None)
        encoding: EncodingName = previous.encoding if previous else "utf-8"
        raw = prepared.encode(encoding)
        if len(raw) > self.max_write_bytes:
            raise WorkspaceError(
                f"file exceeds write limit: {len(raw)} bytes > {self.max_write_bytes} bytes"
            )
        if path.exists() and raw == path.read_bytes():
            raise WorkspaceError(f"new content is identical to the current file: {relative_path}")
        self._atomic_write(path, raw)
        return self._record_read(path, raw, encoding)

    def replace_text(
        self,
        relative_path: str,
        old_text: str,
        new_text: str,
        *,
        replace_all: bool = False,
    ) -> tuple[FileReadRecord, int]:
        if old_text == new_text:
            raise WorkspaceError("old_text and new_text must be different")
        self.require_fresh_read(relative_path)
        content = self.read_text_for_search(relative_path)
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise WorkspaceError(f"old_text was not found in file: {relative_path}")
        if occurrences > 1 and not replace_all:
            raise WorkspaceError(
                f"old_text matches {occurrences} locations; set replace_all=true to replace all"
            )
        count = occurrences if replace_all else 1
        updated = content.replace(old_text, new_text, -1 if replace_all else 1)
        return self.write_text(relative_path, updated), count

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
            raw = path.read_bytes()
        except OSError as exc:
            raise WorkspaceError(f"unable to read file: {display_path}: {exc}") from exc
        if len(raw) > self.max_read_bytes:
            raise WorkspaceError(
                f"file exceeds read limit: {len(raw)} bytes > {self.max_read_bytes} bytes"
            )
        return raw

    def _record_read(
        self,
        path: Path,
        raw: bytes,
        encoding: EncodingName,
    ) -> FileReadRecord:
        content, _ = self._decode_text(raw, str(path.relative_to(self.root)))
        stat = path.stat()
        record = FileReadRecord(
            path=path,
            digest=self._digest(raw),
            modified_ns=stat.st_mtime_ns,
            size_bytes=len(raw),
            complete=True,
            encoding=encoding,
            newline=self._detect_newline(content),
        )
        self.read_ledger.record(record)
        return record

    @staticmethod
    def _preserve_newlines(content: str, newline: NewlineStyle | None) -> str:
        if newline not in {"lf", "crlf", "cr"}:
            return content
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        separator = {"lf": "\n", "crlf": "\r\n", "cr": "\r"}[newline]
        return normalized.replace("\n", separator)

    @staticmethod
    def _atomic_write(path: Path, raw: bytes) -> None:
        original_mode = path.stat().st_mode if path.exists() else None
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            if original_mode is not None:
                os.chmod(temporary, original_mode)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

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
