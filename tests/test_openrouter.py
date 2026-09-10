"""The OpenRouter client: JSON tolerance, pricing, cost and retries."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from vex import openrouter

MODELS_PAYLOAD: dict[str, Any] = {
    "data": [
        {
            "id": "test/text",
            "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
            "supported_parameters": ["response_format", "structured_outputs"],
            "architecture": {"input_modalities": ["text"]},
        },
        {
            "id": "test/vision",
            "pricing": {"prompt": "0.0000002", "completion": "0.0000008", "image": "0.001"},
            "supported_parameters": ["response_format"],
            "architecture": {"input_modalities": ["text", "image"]},
        },
    ]
}


def completion_payload(
    content: str,
    *,
    model: str = "test/text",
    prompt_tokens: int = 1000,
    completion_tokens: int = 100,
) -> dict[str, Any]:
    """An OpenAI-shaped chat completion carrying this message content."""
    return {
        "id": "gen-1",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_client(handler: Any, **kwargs: Any) -> openrouter.OpenRouterClient:
    """A client wired to a mocked transport, with retries that do not sleep."""
    return openrouter.OpenRouterClient(
        api_key="test-key",
        require_key=False,
        backoff=0.0,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def models_and_completion(content: str, **kwargs: Any) -> Any:
    """A handler that serves the model list and always replies with this content."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        return httpx.Response(200, json=completion_payload(content, **kwargs))

    return handler


@pytest.mark.parametrize(
    "raw",
    [
        '{"category": "recipe"}',
        '```json\n{"category": "recipe"}\n```',
        '```\n{"category": "recipe"}\n```',
        'Here you go:\n{"category": "recipe"}\nHope that helps.',
        '   {"category": "recipe"}   ',
    ],
)
def test_parse_json_object_is_tolerant(raw: str) -> None:
    assert openrouter.parse_json_object(raw) == {"category": "recipe"}


def test_parse_json_object_handles_braces_inside_strings() -> None:
    raw = 'prose {"caption": "a } brace", "category": "article"} trailing'
    assert openrouter.parse_json_object(raw)["caption"] == "a } brace"


@pytest.mark.parametrize("raw", ["", "   ", "no json at all", "{not json}"])
def test_parse_json_object_rejects_rubbish(raw: str) -> None:
    with pytest.raises(openrouter.ResponseFormatError):
        openrouter.parse_json_object(raw)


def test_cost_is_computed_from_usage_and_pricing() -> None:
    client = make_client(models_and_completion('{"category": "article"}'))
    result = client.complete("test/text", [openrouter.text_message("user", "hi")])
    # 1000 input at $0.0000001 plus 100 output at $0.0000004.
    assert result.input_tokens == 1000
    assert result.output_tokens == 100
    assert result.cost_usd == pytest.approx(0.0001 + 0.00004)
    client.close()


def test_reported_cost_wins_over_computed_cost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        payload = completion_payload('{"category": "article"}')
        payload["usage"]["cost"] = 0.00025
        return httpx.Response(200, json=payload)

    client = make_client(handler)
    assert client.complete("test/text", []).cost_usd == pytest.approx(0.00025)
    client.close()


def test_image_pricing_is_added_per_image() -> None:
    client = make_client(models_and_completion('{"category": "article"}'))
    result = client.complete("test/vision", [], images=1)
    assert result.cost_usd == pytest.approx(0.0002 + 0.00008 + 0.001)
    client.close()


def test_pricing_is_fetched_once_per_client() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        return httpx.Response(200, json=completion_payload('{"category": "article"}'))

    client = make_client(handler)
    client.complete("test/text", [])
    client.complete("test/text", [])
    assert calls.count("/api/v1/models") == 1
    client.close()


def test_json_schema_is_requested_when_the_model_supports_it() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion_payload('{"category": "article"}'))

    client = make_client(handler)
    schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    client.complete("test/text", [], json_schema=schema)
    client.complete("test/vision", [], json_schema=schema)
    assert bodies[0]["response_format"] == schema
    # test/vision advertises response_format but not structured outputs.
    assert bodies[1]["response_format"] == {"type": "json_object"}
    client.close()


def test_retries_on_429_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=completion_payload('{"category": "article"}'))

    client = make_client(handler)
    result = client.complete("test/text", [])
    assert attempts["count"] == 3
    assert result.attempts == 3
    client.close()


def test_gives_up_after_the_attempt_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        return httpx.Response(503, text="upstream down")

    client = make_client(handler, max_attempts=2)
    with pytest.raises(openrouter.OpenRouterError):
        client.complete("test/text", [])
    client.close()


def test_client_errors_are_not_retried() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        attempts["count"] += 1
        return httpx.Response(401, text="no key")

    client = make_client(handler)
    with pytest.raises(openrouter.OpenRouterError):
        client.complete("test/text", [])
    assert attempts["count"] == 1
    client.close()


def test_missing_pricing_degrades_gracefully() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(500, text="nope")
        return httpx.Response(200, json=completion_payload('{"category": "article"}'))

    client = make_client(handler)
    result = client.complete("unknown/model", [])
    assert result.cost_usd == 0.0
    assert result.json_payload() == {"category": "article"}
    client.close()


def test_missing_api_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(openrouter.ENV_API_KEY, raising=False)
    monkeypatch.setattr(openrouter.config, "load_env", lambda: None)
    with pytest.raises(openrouter.MissingApiKeyError):
        openrouter.get_api_key()


@pytest.mark.parametrize(
    ("choice", "expected", "truncated"),
    [
        ({"finish_reason": "stop"}, "stop", False),
        ({"finish_reason": "length"}, "length", True),
        ({"finish_reason": "LENGTH"}, "length", True),
        ({"native_finish_reason": "MAX_TOKENS"}, "max_tokens", False),
        ({}, None, False),
    ],
)
def test_finish_reason_is_surfaced(
    choice: dict[str, str], expected: str | None, truncated: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        payload = completion_payload('{"category": "article"}')
        payload["choices"][0].update(choice)
        return httpx.Response(200, json=payload)

    client = make_client(handler)
    result = client.complete("test/text", [])
    assert result.finish_reason == expected
    assert result.truncated is truncated
    client.close()


def test_image_message_carries_a_data_url() -> None:
    message = openrouter.image_message("look", "QUJD")
    assert message["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,QUJD")
