from __future__ import annotations

from pathlib import Path

import pytest

from capycode.tools import ListFilesTool, ReadFileTool, SearchCodeTool, ToolExecutor, ToolRegistry
from capycode.workspace import LocalWorkspace


@pytest.mark.asyncio
async def test_list_files_filters_and_ignores_generated_directories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('app')", encoding="utf-8")
    (tmp_path / "src" / "notes.txt").write_text("notes", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("hidden", encoding="utf-8")
    executor = ToolExecutor(ToolRegistry([ListFilesTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("list_files", {"pattern": "*.py"})

    assert result.status == "success"
    assert result.content == "src/app.py"
    assert result.data["count"] == 1


@pytest.mark.asyncio
async def test_search_code_returns_lines_without_marking_files_as_read(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("Alpha\nneedle here\nOmega\n", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)
    executor = ToolExecutor(ToolRegistry([SearchCodeTool()]), workspace)

    result = await executor.execute("search_code", {"query": "NEEDLE", "pattern": "*.py"})

    assert result.status == "success"
    assert result.content == "app.py:2: needle here"
    assert workspace.read_ledger.paths() == ()


@pytest.mark.asyncio
async def test_search_code_reports_invalid_regex(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([SearchCodeTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("search_code", {"query": "[", "regex": True})

    assert result.status == "error"
    assert "invalid regular expression" in result.content


@pytest.mark.asyncio
async def test_read_file_reports_ledger_metadata(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_bytes(b"hello\r\n")
    executor = ToolExecutor(ToolRegistry([ReadFileTool()]), LocalWorkspace(tmp_path))

    result = await executor.execute("read_file", {"path": "file.txt"})

    assert result.status == "success"
    assert result.data["newline"] == "crlf"
    assert len(result.data["digest"]) == 64


@pytest.mark.asyncio
async def test_tool_executor_keeps_head_and_tail_when_truncating(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("A" * 300 + "B" * 300, encoding="utf-8")
    executor = ToolExecutor(
        ToolRegistry([ReadFileTool()]),
        LocalWorkspace(tmp_path),
        max_result_characters=256,
    )

    result = await executor.execute("read_file", {"path": "large.txt"})

    assert len(result.content) == 256
    assert result.content.startswith("A")
    assert result.content.endswith("B")
    assert result.data["truncated"] is True
    assert result.data["original_characters"] == 600
