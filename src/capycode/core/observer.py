from __future__ import annotations

from typing import Protocol

from capycode.tools import ToolResult


class RuntimeObserver(Protocol):
    async def on_model_start(self, step: int) -> None: ...

    async def on_text_delta(self, delta: str) -> None: ...

    async def on_tool_start(self, name: str, arguments: dict[str, object]) -> None: ...

    async def on_tool_result(self, name: str, result: ToolResult) -> None: ...


class NullRuntimeObserver:
    async def on_model_start(self, step: int) -> None:
        pass

    async def on_text_delta(self, delta: str) -> None:
        pass

    async def on_tool_start(self, name: str, arguments: dict[str, object]) -> None:
        pass

    async def on_tool_result(self, name: str, result: ToolResult) -> None:
        pass
