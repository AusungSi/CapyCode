from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from capycode.tools import ToolResult

if TYPE_CHECKING:
    from capycode.trace import RunEvent


class RuntimeObserver(Protocol):
    def on_run_event(self, event: RunEvent) -> None: ...

    async def on_model_start(self, step: int) -> None: ...

    async def on_text_delta(self, delta: str) -> None: ...

    async def on_tool_start(self, name: str, arguments: dict[str, object]) -> None: ...

    async def on_tool_result(self, name: str, result: ToolResult) -> None: ...


class NullRuntimeObserver:
    def on_run_event(self, event: RunEvent) -> None:
        pass

    async def on_model_start(self, step: int) -> None:
        pass

    async def on_text_delta(self, delta: str) -> None:
        pass

    async def on_tool_start(self, name: str, arguments: dict[str, object]) -> None:
        pass

    async def on_tool_result(self, name: str, result: ToolResult) -> None:
        pass
