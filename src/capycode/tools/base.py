from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from capycode.llm.types import ToolDefinition
from capycode.workspace import LocalWorkspace


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolResult(BaseModel):
    status: Literal["success", "error"]
    content: str
    data: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[ToolInput]]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )

    @abstractmethod
    async def execute(self, arguments: ToolInput, workspace: LocalWorkspace) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [self._tools[name].definition() for name in sorted(self._tools)]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        workspace: LocalWorkspace,
        *,
        max_result_characters: int = 40_000,
    ) -> None:
        if max_result_characters < 256:
            raise ValueError("max_result_characters must be at least 256")
        self.registry = registry
        self.workspace = workspace
        self.max_result_characters = max_result_characters

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(status="error", content=f"unknown tool: {name}")
        try:
            parsed = tool.input_model.model_validate(arguments)
            result = await tool.execute(parsed, self.workspace)
            return self._limit_result(result)
        except ValidationError as exc:
            return ToolResult(status="error", content=f"invalid tool arguments: {exc}")
        except (OSError, ValueError) as exc:
            return ToolResult(status="error", content=f"{type(exc).__name__}: {exc}")

    def _limit_result(self, result: ToolResult) -> ToolResult:
        if len(result.content) <= self.max_result_characters:
            return result
        marker = "\n\n... tool result truncated ...\n\n"
        available = self.max_result_characters - len(marker)
        head_size = available // 2
        tail_size = available - head_size
        data = dict(result.data)
        data.update(
            {
                "truncated": True,
                "original_characters": len(result.content),
            }
        )
        return result.model_copy(
            update={
                "content": result.content[:head_size] + marker + result.content[-tail_size:],
                "data": data,
            }
        )
