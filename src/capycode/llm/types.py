from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCall(RuntimeModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(RuntimeModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> Message:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool calls")
        return self


class ToolDefinition(RuntimeModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any]


class LLMRequest(RuntimeModel):
    model: str = Field(min_length=1)
    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    temperature: float = 0.0
    max_output_tokens: int = Field(default=4096, gt=0)


class Usage(RuntimeModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class LLMResponse(RuntimeModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
