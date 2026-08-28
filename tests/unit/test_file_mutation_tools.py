from __future__ import annotations

import os
from pathlib import Path

import pytest

from capycode.tools import ReadFileTool, ReplaceTextTool, ToolExecutor, ToolRegistry, WriteFileTool
from capycode.workspace import LocalWorkspace


def temporary_artifacts(directory: Path, filename: str) -> list[Path]:
    return list(directory.glob(f".{filename}.*.tmp"))


def build_executor(workspace: LocalWorkspace) -> ToolExecutor:
    return ToolExecutor(
        ToolRegistry([ReadFileTool(), ReplaceTextTool(), WriteFileTool()]),
        workspace,
    )


@pytest.mark.asyncio
async def test_write_file_creates_new_file_atomically(tmp_path: Path) -> None:
    executor = build_executor(LocalWorkspace(tmp_path))

    result = await executor.execute("write_file", {"path": "new.txt", "content": "hello\n"})

    assert result.status == "success"
    assert result.data["created"] is True
    assert (tmp_path / "new.txt").read_bytes() == b"hello\n"
    assert temporary_artifacts(tmp_path, "new.txt") == []


@pytest.mark.asyncio
async def test_write_existing_file_requires_fresh_read(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("before", encoding="utf-8")
    executor = build_executor(LocalWorkspace(tmp_path))

    result = await executor.execute("write_file", {"path": "target.txt", "content": "after"})

    assert result.status == "error"
    assert "must be fully read" in result.content
    assert (tmp_path / "target.txt").read_text(encoding="utf-8") == "before"


@pytest.mark.asyncio
async def test_write_existing_file_detects_stale_read(tmp_path: Path) -> None:
    path = tmp_path / "target.txt"
    path.write_text("before", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)
    executor = build_executor(workspace)
    await executor.execute("read_file", {"path": "target.txt"})
    path.write_text("external", encoding="utf-8")

    result = await executor.execute("write_file", {"path": "target.txt", "content": "agent"})

    assert result.status == "error"
    assert "changed since it was read" in result.content
    assert path.read_text(encoding="utf-8") == "external"


@pytest.mark.asyncio
async def test_write_preserves_bom_and_crlf(tmp_path: Path) -> None:
    path = tmp_path / "windows.txt"
    path.write_bytes(b"\xef\xbb\xbfbefore\r\n")
    workspace = LocalWorkspace(tmp_path)
    executor = build_executor(workspace)
    await executor.execute("read_file", {"path": "windows.txt"})

    result = await executor.execute(
        "write_file", {"path": "windows.txt", "content": "after\nsecond\n"}
    )

    assert result.status == "success"
    assert path.read_bytes() == b"\xef\xbb\xbfafter\r\nsecond\r\n"
    assert result.data["newline"] == "crlf"


@pytest.mark.asyncio
async def test_write_rejects_identical_content(tmp_path: Path) -> None:
    path = tmp_path / "target.txt"
    path.write_text("same", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)
    executor = build_executor(workspace)
    await executor.execute("read_file", {"path": "target.txt"})

    result = await executor.execute("write_file", {"path": "target.txt", "content": "same"})

    assert result.status == "error"
    assert "identical" in result.content


@pytest.mark.asyncio
async def test_write_rejects_content_over_byte_limit(tmp_path: Path) -> None:
    executor = build_executor(LocalWorkspace(tmp_path, max_write_bytes=4))

    result = await executor.execute("write_file", {"path": "large.txt", "content": "12345"})

    assert result.status == "error"
    assert "exceeds write limit" in result.content
    assert not (tmp_path / "large.txt").exists()


@pytest.mark.asyncio
async def test_replace_text_requires_unique_match_by_default(tmp_path: Path) -> None:
    path = tmp_path / "target.txt"
    path.write_text("same same", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)
    executor = build_executor(workspace)
    await executor.execute("read_file", {"path": "target.txt"})

    result = await executor.execute(
        "replace_text", {"path": "target.txt", "old_text": "same", "new_text": "new"}
    )

    assert result.status == "error"
    assert "matches 2 locations" in result.content
    assert path.read_text(encoding="utf-8") == "same same"


@pytest.mark.asyncio
async def test_replace_text_can_explicitly_replace_all(tmp_path: Path) -> None:
    path = tmp_path / "target.txt"
    path.write_text("same same", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)
    executor = build_executor(workspace)
    await executor.execute("read_file", {"path": "target.txt"})

    result = await executor.execute(
        "replace_text",
        {"path": "target.txt", "old_text": "same", "new_text": "new", "replace_all": True},
    )

    assert result.status == "success"
    assert result.data["replacements"] == 2
    assert path.read_text(encoding="utf-8") == "new new"


def test_atomic_write_failure_keeps_original_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "target.txt"
    path.write_text("before", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)
    workspace.read_text("target.txt")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        workspace.write_text("target.txt", "after")
    assert path.read_text(encoding="utf-8") == "before"
    assert temporary_artifacts(tmp_path, "target.txt") == []
