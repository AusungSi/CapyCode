from __future__ import annotations

from pathlib import Path


class WorkspaceError(ValueError):
    pass


class LocalWorkspace:
    def __init__(self, root: Path, *, max_read_bytes: int = 1_000_000) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace is not a directory: {self.root}")
        self.max_read_bytes = max_read_bytes

    def resolve_file(self, relative_path: str) -> Path:
        requested = Path(relative_path)
        if requested.is_absolute():
            raise WorkspaceError("workspace paths must be relative")

        candidate = (self.root / requested).resolve(strict=False)
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError("path escapes the workspace")

        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkspaceError(f"file not found: {relative_path}") from exc

        if not resolved.is_relative_to(self.root):
            raise WorkspaceError("path escapes the workspace")
        if not resolved.is_file():
            raise WorkspaceError(f"path is not a file: {relative_path}")
        return resolved

    def read_text(self, relative_path: str) -> str:
        path = self.resolve_file(relative_path)
        size = path.stat().st_size
        if size > self.max_read_bytes:
            raise WorkspaceError(
                f"file exceeds read limit: {size} bytes > {self.max_read_bytes} bytes"
            )
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"file is not valid UTF-8 text: {relative_path}") from exc
