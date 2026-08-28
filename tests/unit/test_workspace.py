from __future__ import annotations

from pathlib import Path

import pytest

from capycode.workspace import LocalWorkspace, WorkspaceError


def test_workspace_reads_utf8_file(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)

    assert workspace.read_text("hello.py") == "print('hello')\n"


def test_workspace_rejects_path_escape(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(WorkspaceError, match="escapes the workspace"):
        workspace.read_text("../outside.txt")


def test_workspace_rejects_absolute_path(tmp_path: Path) -> None:
    path = tmp_path / "inside.txt"
    path.write_text("content", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(WorkspaceError, match="must be relative"):
        workspace.read_text(str(path.resolve()))


def test_workspace_reports_missing_file_inside_workspace(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(WorkspaceError, match=r"file not found: missing\.txt"):
        workspace.read_text("missing.txt")


def test_workspace_rejects_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("12345", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path, max_read_bytes=4)

    with pytest.raises(WorkspaceError, match="exceeds read limit"):
        workspace.read_text("large.txt")
