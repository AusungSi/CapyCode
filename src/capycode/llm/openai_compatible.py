from __future__ import annotations

import json
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
    ) -> None:
        normalized_url = base_url.rstrip("/") + "/"
        self._client = httpx.AsyncClient(
            base_url=normalized_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = self._request_payload(request)
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
        payload = self._request_payload(request)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage = Usage()

        try:
            async with self._client.stream("POST", "chat/completions", json=payload) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    chunk_usage = chunk.get("usage") or {}
                    if chunk_usage:
                        usage = Usage(
                            input_tokens=chunk_usage.get("prompt_tokens", 0),
                            output_tokens=chunk_usage.get("completion_tokens", 0),
                        )
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
        except LLMError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError(
                LLMErrorKind.INVALID_RESPONSE,
                f"invalid OpenAI-compatible stream: {exc}",
                retryable=False,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(LLMErrorKind.NETWORK, str(exc), retryable=True) from exc

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
            raise LLMError(
                LLMErrorKind.INVALID_RESPONSE,
                f"invalid streamed tool call: {exc}",
                retryable=False,
            ) from exc
        return LLMResponse(
            content="".join(text_parts) or None,
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
        try:
            response = await self._client.request(method, path, **kwargs)
            self._raise_for_status(response)
            return response
        except LLMError:
            raise
        except httpx.HTTPError as exc:
            raise LLMError(LLMErrorKind.NETWORK, str(exc), retryable=True) from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if not response.is_error:
            return
        status = response.status_code
        if status in {401, 403}:
            kind, retryable = LLMErrorKind.AUTHENTICATION, False
        elif status == 429:
            kind, retryable = LLMErrorKind.RATE_LIMIT, True
        else:
            kind, retryable = LLMErrorKind.SERVICE, status >= 500
        raise LLMError(kind, f"model endpoint returned HTTP {status}", retryable=retryable)

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
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
