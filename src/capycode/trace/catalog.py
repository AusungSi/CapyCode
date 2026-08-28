from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

from pydantic import ValidationError

from .events import RUN_EVENT_ADAPTER, RunEvent, RunSummary

RunEventList: TypeAlias = list[RunEvent]


class RunCatalog:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.root = self.workspace / ".capy" / "runs"

    def list(self) -> list[RunSummary]:
        if not self.root.exists():
            return []
        summaries: list[RunSummary] = []
        for path in self.root.glob("*/summary.json"):
            summary = self._read(path, strict=False)
            if summary is not None and Path(summary.workspace).resolve() == self.workspace:
                summaries.append(summary)
        return sorted(summaries, key=lambda item: item.finished_at, reverse=True)

    def resolve(self, value: str) -> RunSummary:
        summaries = self.list()
        if not summaries:
            raise ValueError("当前工作区没有运行记录")
        query = value.strip().lower()
        if query in {"", "latest"}:
            return summaries[0]
        matches = [item for item in summaries if item.run_id.startswith(query)]
        if not matches:
            raise ValueError(f"未找到运行记录: {value}")
        if len(matches) > 1:
            raise ValueError(f"运行记录标识不唯一: {value}")
        return matches[0]

    def events(self, value: str) -> RunEventList:
        summary = self.resolve(value)
        trace_path = self.root / summary.run_id / "trace.jsonl"
        try:
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            return [RUN_EVENT_ADAPTER.validate_json(line) for line in lines if line.strip()]
        except (OSError, ValidationError, ValueError) as exc:
            raise ValueError(f"无法读取运行事件 {trace_path}: {exc}") from exc

    @staticmethod
    def _read(path: Path, *, strict: bool) -> RunSummary | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("schema_version") == 1:
                payload.pop("model_alias", None)
                payload["currency"] = payload.get("currency") or "UNSPECIFIED"
                payload["schema_version"] = 2
            return RunSummary.model_validate(payload)
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            if strict:
                raise ValueError(f"无法读取运行记录 {path}: {exc}") from exc
            return None
