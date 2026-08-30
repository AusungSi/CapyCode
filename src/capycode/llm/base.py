from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Protocol

from .types import LLMRequest, LLMResponse

TextDeltaHandler = Callable[[str], Awaitable[None]]


class LLMErrorKind(StrEnum):
    AUTHENTICATION = "authentication"
    BAD_REQUEST = "bad_request"
    RATE_LIMIT = "rate_limit"
    SERVICE = "service"
    NETWORK = "network"
    INVALID_RESPONSE = "invalid_response"


class LLMError(RuntimeError):
    def __init__(
        self,
        kind: LLMErrorKind,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class LLMClient(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    async def stream(self, request: LLMRequest, on_text_delta: TextDeltaHandler) -> LLMResponse: ...
