from __future__ import annotations

import os
from pathlib import Path

import pytest

from capycode.workspace import LocalWorkspace, WorkspaceError


def test_workspace_reads_utf8_file(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_bytes(b"print('hello')\n")
    workspace = LocalWorkspace(tmp_path)

    assert workspace.read_text("hello.py") == "print('hello')\n"
    record = workspace.read_ledger.get((tmp_path / "hello.py").resolve())
    assert record is not None
    assert record.complete
    assert record.encoding == "utf-8"
    assert record.newline == "lf"
    assert len(record.digest) == 64


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

    with pytest.raises(WorkspaceError, match=r"path not found: missing\.txt"):
        workspace.read_text("missing.txt")


def test_workspace_rejects_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("12345", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path, max_read_bytes=4)

    with pytest.raises(WorkspaceError, match="exceeds read limit"):
        workspace.read_text("large.txt")


def test_workspace_records_utf8_bom_and_crlf_without_normalizing(tmp_path: Path) -> None:
    path = tmp_path / "windows.txt"
    path.write_bytes(b"\xef\xbb\xbffirst\r\nsecond\r\n")
    workspace = LocalWorkspace(tmp_path)

    assert workspace.read_text("windows.txt") == "first\r\nsecond\r\n"
    record = workspace.read_ledger.get(path.resolve())
    assert record is not None
    assert record.encoding == "utf-8-sig"
    assert record.newline == "crlf"


def test_workspace_requires_read_before_modification(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("before", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(WorkspaceError, match="must be fully read"):
        workspace.require_fresh_read("target.txt")


def test_workspace_detects_external_change_after_read(tmp_path: Path) -> None:
    path = tmp_path / "target.txt"
    path.write_text("before", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)
    workspace.read_text("target.txt")
    path.write_text("changed-content", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="changed since it was read"):
        workspace.require_fresh_read("target.txt")


def test_workspace_detects_changed_digest_when_metadata_matches(tmp_path: Path) -> None:
    path = tmp_path / "target.txt"
    path.write_bytes(b"before")
    workspace = LocalWorkspace(tmp_path)
    workspace.read_text("target.txt")
    record = workspace.read_ledger.get(path.resolve())
    assert record is not None

    path.write_bytes(b"change")
    os.utime(path, ns=(record.modified_ns, record.modified_ns))

    with pytest.raises(WorkspaceError, match="changed since it was read"):
        workspace.require_fresh_read("target.txt")


@pytest.mark.parametrize("path", [r"C:\outside.txt", r"\\server\share\file.txt"])
def test_workspace_rejects_windows_absolute_and_unc_paths(tmp_path: Path, path: str) -> None:
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(WorkspaceError, match=r"relative|UNC"):
        workspace.read_text(path)


def test_workspace_rejects_symlink_that_escapes_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "outside-link.txt"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this host")
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(WorkspaceError, match="escapes the workspace"):
        workspace.read_text("outside-link.txt")
