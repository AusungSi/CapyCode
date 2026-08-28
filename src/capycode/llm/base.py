from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Protocol

from .types import LLMRequest, LLMResponse

TextDeltaHandler = Callable[[str], Awaitable[None]]


class LLMErrorKind(StrEnum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SERVICE = "service"
    NETWORK = "network"
    INVALID_RESPONSE = "invalid_response"


class LLMError(RuntimeError):
    def __init__(self, kind: LLMErrorKind, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class LLMClient(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    async def stream(self, request: LLMRequest, on_text_delta: TextDeltaHandler) -> LLMResponse: ...
