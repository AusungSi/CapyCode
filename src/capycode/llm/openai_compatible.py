from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import httpx

from .base import LLMError, LLMErrorKind, TextDeltaHandler
from .types import LLMRequest, LLMResponse, Message, ToolCall, Usage


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.name:
        payload["name"] = message.name
    return payload


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 120,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
        retry_base_seconds: float = 0.75,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds must not be negative")
        normalized_url = base_url.rstrip("/") + "/"
        self._client = httpx.AsyncClient(
            base_url=normalized_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds

    async def generate(self, request: LLMRequest) -> LLMResponse:
        effective_request = request
        try:
            payload = self._request_payload(effective_request)
            response = await self._request("POST", "chat/completions", json=payload)
        except LLMError as exc:
            if not self._requires_temperature_one(exc) or request.temperature == 1:
                raise
            effective_request = request.model_copy(update={"temperature": 1.0})
            payload = self._request_payload(effective_request)
            response = await self._request("POST", "chat/completions", json=payload)

        try:
            return self._parse_response(response.json())
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError(
                LLMErrorKind.INVALID_RESPONSE,
                f"invalid OpenAI-compatible response: {exc}",
                retryable=False,
            ) from exc

    async def stream(self, request: LLMRequest, on_text_delta: TextDeltaHandler) -> LLMResponse:
        return await self._stream_with_retries(request, on_text_delta, attempt=0)

    async def _stream_with_retries(
        self,
        request: LLMRequest,
        on_text_delta: TextDeltaHandler,
        *,
        attempt: int,
    ) -> LLMResponse:
        payload = self._request_payload(request)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage = Usage()

        try:
            async with self._client.stream("POST", "chat/completions", json=payload) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    chunk_usage = chunk.get("usage") or {}
                    if chunk_usage:
                        usage = self._parse_usage(chunk_usage)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        text_parts.append(content)
                        await on_text_delta(content)
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                    for item in delta.get("tool_calls") or []:
                        index = item["index"]
                        part = tool_parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if item.get("id"):
                            part["id"] = item["id"]
                        function = item.get("function") or {}
                        if function.get("name"):
                            part["name"] += function["name"]
                        if function.get("arguments"):
                            part["arguments"] += function["arguments"]
        except LLMError as exc:
            if self._requires_temperature_one(exc) and request.temperature != 1:
                adjusted = request.model_copy(update={"temperature": 1.0})
                return await self._stream_with_retries(adjusted, on_text_delta, attempt=attempt)
            # Some OpenAI-compatible gateways implement Chat Completions but
            # reject SSE or stream_options. Retry once through the regular
            # JSON Chat Completions request only when the response explicitly
            # identifies streaming as the unsupported feature.
            if exc.status_code == 400 and self._is_stream_compatibility_error(exc):
                return await self.generate(request)
            if exc.retryable and attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay(exc, attempt))
                return await self._stream_with_retries(request, on_text_delta, attempt=attempt + 1)
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError(
                LLMErrorKind.INVALID_RESPONSE,
                f"invalid OpenAI-compatible stream: {exc}",
                retryable=False,
            ) from exc
        except httpx.TimeoutException as exc:
            error = LLMError(
                LLMErrorKind.NETWORK,
                "model endpoint request timed out; check the endpoint, proxy, and model support",
                retryable=True,
            )
            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay(error, attempt))
                return await self._stream_with_retries(request, on_text_delta, attempt=attempt + 1)
            raise error from exc
        except httpx.HTTPError as exc:
            error = LLMError(LLMErrorKind.NETWORK, str(exc) or type(exc).__name__, retryable=True)
            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay(error, attempt))
                return await self._stream_with_retries(request, on_text_delta, attempt=attempt + 1)
            raise error from exc

        try:
            tool_calls = [
                ToolCall(
                    id=part["id"],
                    name=part["name"],
                    arguments=json.loads(part["arguments"] or "{}"),
                )
                for _, part in sorted(tool_parts.items())
            ]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            # Gateways can occasionally truncate a streamed function-call
            # argument while their regular Chat Completions response remains
            # valid. Reissue once without streaming before rejecting it.
            try:
                return await self.generate(request)
            except LLMError as fallback_exc:
                raise LLMError(
                    LLMErrorKind.INVALID_RESPONSE,
                    f"invalid streamed tool call: {exc}; non-stream fallback failed: "
                    f"{fallback_exc}",
                    retryable=False,
                ) from fallback_exc

        # A few compatible gateways close a streamed response with an empty
        # choice even though the same request works through the regular JSON
        # endpoint.  Retry once without streaming before surfacing an empty
        # model response to the agent loop.
        if not text_parts and not reasoning_parts and not tool_calls:
            try:
                return await self.generate(request)
            except LLMError:
                # Preserve the normal agent-loop retry path for gateways that
                # return an empty/non-JSON body on both transports.
                return LLMResponse(finish_reason="empty")
        return LLMResponse(
            content="".join(text_parts) or None,
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _request_payload(request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [_message_payload(message) for message in request.messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if not request.tools:
            payload.pop("tools")
        return payload

    async def list_models(self) -> list[str]:
        response = await self._request("GET", "models")
        try:
            payload = response.json()
            items = payload["data"]
            if not isinstance(items, list):
                raise TypeError("data must be a list")
            model_ids = sorted(
                {
                    item["id"]
                    for item in items
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and item["id"].strip()
                }
            )
            if not model_ids:
                raise ValueError("response contains no model IDs")
            return model_ids
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError(
                LLMErrorKind.INVALID_RESPONSE,
                f"invalid model-list response: {exc}",
                retryable=False,
            ) from exc

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method, path, **kwargs)
                await self._raise_for_status(response)
                return response
            except LLMError as exc:
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                await asyncio.sleep(self._retry_delay(exc, attempt))
            except httpx.TimeoutException as exc:
                error = LLMError(
                    LLMErrorKind.NETWORK,
                    "model endpoint request timed out; check the endpoint, proxy, "
                    "and model support",
                    retryable=True,
                )
                if attempt >= self._max_retries:
                    raise error from exc
                await asyncio.sleep(self._retry_delay(error, attempt))
            except httpx.HTTPError as exc:
                error = LLMError(
                    LLMErrorKind.NETWORK,
                    str(exc) or type(exc).__name__,
                    retryable=True,
                )
                if attempt >= self._max_retries:
                    raise error from exc
                await asyncio.sleep(self._retry_delay(error, attempt))
        raise AssertionError("unreachable retry loop")

    def _retry_delay(self, exc: LLMError, attempt: int) -> float:
        if exc.retry_after_seconds is not None:
            return float(min(max(exc.retry_after_seconds, 0), 30))
        base = float(self._retry_base_seconds) * (2**attempt)
        jitter = random.random() * float(self._retry_base_seconds) * 0.35
        return float(min(base + jitter, 15))

    @staticmethod
    def _requires_temperature_one(exc: LLMError) -> bool:
        message = str(exc).lower()
        return (
            exc.status_code == 400
            and "temperature" in message
            and any(
                marker in message
                for marker in ("only 1", "must be 1", "temperature=1", "temperature 1")
            )
        )

    @staticmethod
    def _is_stream_compatibility_error(exc: LLMError) -> bool:
        message = str(exc).lower()
        stream_terms = ("stream", "sse", "stream_options", "server-sent")
        unsupported_terms = ("unsupported", "not support", "not allowed", "invalid")
        return any(term in message for term in stream_terms) and any(
            term in message for term in unsupported_terms
        )

    @staticmethod
    async def _raise_for_status(response: httpx.Response) -> None:
        if not response.is_error:
            return
        status = response.status_code
        # Streaming responses cannot expose ``.text`` until their body has
        # been consumed.  Keep status diagnostics safe for both transports.
        detail = ""
        try:
            if not response.is_stream_consumed:
                await response.aread()
            detail = response.text.strip().replace("\r", " ").replace("\n", " ")
        except httpx.HTTPError:
            detail = ""
        normalized_detail = detail.lower()
        transient_auth_failure = status in {401, 403} and any(
            marker in normalized_detail
            for marker in (
                "timeout",
                "timed out",
                "temporarily unavailable",
                "upstream unavailable",
                "鉴权服务连接失败",
            )
        )
        if transient_auth_failure:
            kind, retryable = LLMErrorKind.SERVICE, True
        elif status in {401, 403}:
            kind, retryable = LLMErrorKind.AUTHENTICATION, False
        elif 400 <= status < 500 and status != 429:
            kind, retryable = LLMErrorKind.BAD_REQUEST, False
        elif status == 429:
            kind, retryable = LLMErrorKind.RATE_LIMIT, True
        else:
            kind, retryable = LLMErrorKind.SERVICE, status >= 500
        if len(detail) > 400:
            detail = detail[:400] + "..."
        suffix = f": {detail}" if detail else ""
        retry_after_seconds = None
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                retry_after_seconds = float(retry_after)
            except ValueError:
                retry_after_seconds = None
        raise LLMError(
            kind,
            f"model endpoint returned HTTP {status}{suffix}",
            retryable=retryable,
            status_code=status,
            retry_after_seconds=retry_after_seconds,
        )

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> LLMResponse:
        choice = payload["choices"][0]
        message = choice["message"]
        calls: list[ToolCall] = []
        for item in message.get("tool_calls", []):
            function = item["function"]
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must decode to an object")
            calls.append(ToolCall(id=item["id"], name=function["name"], arguments=arguments))
        usage = payload.get("usage") or {}
        return LLMResponse(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            usage=OpenAICompatibleLLM._parse_usage(usage),
        )

    @staticmethod
    def _parse_usage(payload: dict[str, Any]) -> Usage:
        details = payload.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}
        cached = details.get("cached_tokens", payload.get("prompt_cache_hit_tokens", 0))
        input_tokens = payload.get("prompt_tokens", 0)
        if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
            input_tokens = 0
        if not isinstance(cached, int) or isinstance(cached, bool):
            cached = 0
        return Usage(
            input_tokens=input_tokens,
            cached_input_tokens=min(max(cached, 0), input_tokens),
            output_tokens=payload.get("completion_tokens", 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
