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
                "usage": {
                    "prompt_tokens": 12,
                    "prompt_cache_hit_tokens": 5,
                    "completion_tokens": 7,
                },
            },
        )

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
        max_retries=0,
    )
    try:
        response = await client.generate(
            LLMRequest(
                model="test-model",
                messages=[Message(role="user", content="Read the README")],
                reasoning_effort="high",
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
    assert body["max_tokens"] == 32_000
    assert body["reasoning_effort"] == "high"
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert response.usage.input_tokens == 12
    assert response.usage.cached_input_tokens == 5
    assert response.usage.output_tokens == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "kind", "retryable"),
    [
        (401, LLMErrorKind.AUTHENTICATION, False),
        (400, LLMErrorKind.BAD_REQUEST, False),
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
        max_retries=0,
    )
    try:
        with pytest.raises(LLMError) as raised:
            await client.generate(LLMRequest(model="test-model", messages=[]))
    finally:
        await client.aclose()

    assert raised.value.kind == kind
    assert raised.value.retryable is retryable


@pytest.mark.asyncio
async def test_openai_compatible_includes_bad_request_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            request=request,
            json={"error": {"message": "tools are not supported for this model"}},
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

    assert raised.value.kind == LLMErrorKind.BAD_REQUEST
    assert raised.value.status_code == 400
    assert "tools are not supported" in str(raised.value)


@pytest.mark.asyncio
async def test_openai_compatible_falls_back_when_streaming_is_unsupported() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        if payload.get("stream") is True:
            return httpx.Response(
                400,
                request=request,
                json={"error": {"message": "streaming is not supported"}},
            )
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.stream(
            LLMRequest(model="test-model", messages=[Message(role="user", content="Hi")]),
            lambda _delta: None,
        )
    finally:
        await client.aclose()

    assert calls == 2
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_openai_compatible_adjusts_required_temperature_once() -> None:
    temperatures: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        temperatures.append(payload["temperature"])
        if payload["temperature"] != 1:
            return httpx.Response(
                400,
                request=request,
                json={
                    "error": {"message": "invalid temperature: only 1 is allowed for this model"}
                },
            )
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.generate(
            LLMRequest(
                model="temperature-one-model",
                messages=[Message(role="user", content="Hi")],
            )
        )
    finally:
        await client.aclose()

    assert temperatures == [0.0, 1.0]
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_openai_compatible_maps_timeout_to_actionable_network_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream did not respond", request=request)

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
        max_retries=0,
    )
    try:
        with pytest.raises(LLMError) as raised:
            await client.generate(LLMRequest(model="test-model", messages=[]))
    finally:
        await client.aclose()

    assert raised.value.kind == LLMErrorKind.NETWORK
    assert raised.value.retryable is True
    assert "timed out" in str(raised.value)


@pytest.mark.asyncio
async def test_openai_compatible_retries_rate_limit_before_succeeding() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, request=request, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
        max_retries=2,
        retry_base_seconds=0,
    )
    try:
        response = await client.generate(LLMRequest(model="test-model", messages=[]))
    finally:
        await client.aclose()

    assert calls == 3
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_openai_compatible_retries_transient_auth_service_timeout() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                401,
                request=request,
                json={"code": 401, "msg": "鉴权服务连接失败:timeout"},
            )
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
        max_retries=1,
        retry_base_seconds=0,
    )
    try:
        response = await client.generate(LLMRequest(model="test-model", messages=[]))
    finally:
        await client.aclose()

    assert calls == 2
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_openai_compatible_retries_rate_limited_stream() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, request=request, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            request=request,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n",
        )

    async def ignore_delta(_delta: str) -> None:
        pass

    client = OpenAICompatibleLLM(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(handler),
        max_retries=1,
        retry_base_seconds=0,
    )
    try:
        response = await client.stream(LLMRequest(model="test-model", messages=[]), ignore_delta)
    finally:
        await client.aclose()

    assert calls == 2
    assert response.content == "ok"


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
                'data: {"choices":[],"usage":{"prompt_tokens":9,'
                '"prompt_tokens_details":{"cached_tokens":3},"completion_tokens":2}}',
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
    assert response.usage.cached_input_tokens == 3
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


@pytest.mark.asyncio
async def test_openai_compatible_falls_back_for_malformed_streamed_tool_call() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        if payload.get("stream") is True:
            return httpx.Response(
                200,
                request=request,
                text=(
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                    '"id":"call-1","function":{"name":"read_file",'
                    '"arguments":"{bad"}}]},"finish_reason":"tool_calls"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
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
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    async def ignore_delta(_delta: str) -> None:
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

    assert calls == 2
    assert response.tool_calls[0].arguments == {"path": "README.md"}
