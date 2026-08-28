from __future__ import annotations

import json

import httpx
import pytest

from capycode.llm import (
    LLMError,
    LLMErrorKind,
    LLMRequest,
    Message,
    OpenAICompatibleLLM,
    ToolDefinition,
)


@pytest.mark.asyncio
async def test_openai_compatible_tool_call_mapping() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
        )

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.generate(
            LLMRequest(
                model="test-model",
                messages=[Message(role="user", content="Read the README")],
                tools=[
                    ToolDefinition(
                        name="read_file",
                        description="Read a file",
                        parameters={"type": "object"},
                    )
                ],
            )
        )
    finally:
        await client.aclose()

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "test-model"
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "kind", "retryable"),
    [
        (401, LLMErrorKind.AUTHENTICATION, False),
        (429, LLMErrorKind.RATE_LIMIT, True),
        (503, LLMErrorKind.SERVICE, True),
    ],
)
async def test_openai_compatible_maps_http_errors(
    status_code: int, kind: LLMErrorKind, retryable: bool
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request)

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(LLMError) as raised:
            await client.generate(LLMRequest(model="test-model", messages=[]))
    finally:
        await client.aclose()

    assert raised.value.kind == kind
    assert raised.value.retryable is retryable


@pytest.mark.asyncio
async def test_openai_compatible_rejects_non_object_tool_arguments() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "[]",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(LLMError) as raised:
            await client.generate(LLMRequest(model="test-model", messages=[]))
    finally:
        await client.aclose()

    assert raised.value.kind == LLMErrorKind.INVALID_RESPONSE
    assert raised.value.retryable is False


@pytest.mark.asyncio
async def test_openai_compatible_discovers_models() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            request=request,
            json={
                "object": "list",
                "data": [
                    {"id": "model-z", "object": "model"},
                    {"id": "model-a", "object": "model"},
                    {"id": "model-a", "object": "model"},
                ],
            },
        )

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        models = await client.list_models()
    finally:
        await client.aclose()

    assert captured["url"] == "https://example.test/v1/models"
    assert captured["authorization"] == "Bearer test-key"
    assert models == ["model-a", "model-z"]


@pytest.mark.asyncio
async def test_openai_compatible_streams_text_and_usage() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        content = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2}}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, request=request, text=content)

    deltas: list[str] = []

    async def on_delta(delta: str) -> None:
        deltas.append(delta)

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.stream(
            LLMRequest(model="test-model", messages=[Message(role="user", content="Hi")]),
            on_delta,
        )
    finally:
        await client.aclose()

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert deltas == ["Hello", " world"]
    assert response.content == "Hello world"
    assert response.usage.input_tokens == 9
    assert response.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_openai_compatible_assembles_streamed_tool_call() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        content = "\n\n".join(
            [
                "data: "
                '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
                '"function":{"name":"read_file","arguments":"{\\"path\\":"}}]},'
                '"finish_reason":null}]}',
                "data: "
                '{"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"\\"README.md\\"}"}}]},'
                '"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, request=request, text=content)

    async def ignore_delta(delta: str) -> None:
        pass

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.stream(LLMRequest(model="test-model", messages=[]), ignore_delta)
    finally:
        await client.aclose()

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}
