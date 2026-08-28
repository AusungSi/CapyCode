from __future__ import annotations

from collections import deque

from .base import TextDeltaHandler
from .types import LLMRequest, LLMResponse


class ScriptedLLM:
    """Deterministic LLM used by runtime and integration tests."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request.model_copy(deep=True))
        if not self._responses:
            raise RuntimeError("scripted LLM has no response remaining")
        return self._responses.popleft()

    async def stream(self, request: LLMRequest, on_text_delta: TextDeltaHandler) -> LLMResponse:
        response = await self.generate(request)
        if response.content:
            await on_text_delta(response.content)
        return response
