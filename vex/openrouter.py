"""A small OpenRouter client: chat completions, JSON coaxing, pricing and cost."""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

from vex import config

BASE_URL = "https://openrouter.ai/api/v1"
ENV_API_KEY = "OPENROUTER_API_KEY"

REFERER = "https://github.com/teejayen/vex"
TITLE = "vex"

DEFAULT_TIMEOUT = 90.0
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF = 1.5
MAX_BACKOFF = 20.0

RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 520, 522, 524})

#: The finish reason a model reports when it ran out of output tokens mid-reply.
FINISH_LENGTH = "length"

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


class OpenRouterError(RuntimeError):
    """Anything that went wrong talking to OpenRouter."""


class MissingApiKeyError(OpenRouterError):
    """No API key is available in the environment or a .env file."""


class ResponseFormatError(OpenRouterError):
    """The model replied with something that is not the JSON object we asked for."""


def get_api_key(explicit: str | None = None) -> str:
    """Read the key from the argument, the environment, or a .env in the working tree."""
    if explicit:
        return explicit
    config.load_env()
    key = os.environ.get(ENV_API_KEY, "").strip()
    if not key:
        raise MissingApiKeyError(
            f"No {ENV_API_KEY} found. Put it in the environment or in a .env at the repo root."
        )
    return key


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-token prices in US dollars, as OpenRouter reports them."""

    model: str
    prompt: float = 0.0
    completion: float = 0.0
    image: float = 0.0
    request: float = 0.0
    supports_response_format: bool = False
    supports_json_schema: bool = False
    supports_image_input: bool = False

    def cost(self, input_tokens: int, output_tokens: int, images: int = 0) -> float:
        """Dollar cost of one call."""
        return (
            input_tokens * self.prompt
            + output_tokens * self.completion
            + images * self.image
            + self.request
        )

    @property
    def prompt_per_million(self) -> float:
        """Input price per million tokens, the way OpenRouter's site quotes it."""
        return self.prompt * 1_000_000

    @property
    def completion_per_million(self) -> float:
        """Output price per million tokens."""
        return self.completion * 1_000_000


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_model_entry(entry: dict[str, Any]) -> ModelPricing:
    """Turn one row of ``GET /models`` into pricing plus capability flags."""
    pricing = entry.get("pricing") or {}
    supported = set(entry.get("supported_parameters") or ())
    modalities = set((entry.get("architecture") or {}).get("input_modalities") or ())
    return ModelPricing(
        model=str(entry.get("id", "")),
        prompt=_to_float(pricing.get("prompt")),
        completion=_to_float(pricing.get("completion")),
        image=_to_float(pricing.get("image")),
        request=_to_float(pricing.get("request")),
        supports_response_format="response_format" in supported,
        supports_json_schema="structured_outputs" in supported,
        supports_image_input="image" in modalities,
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Completion:
    """One chat completion, with what it cost and how long it took."""

    model: str
    content: str
    raw: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
    attempts: int = 1
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """True when the reply stopped because it ran out of output tokens."""
        return self.finish_reason == FINISH_LENGTH

    def json_payload(self) -> dict[str, Any]:
        """The message content parsed as a JSON object."""
        return parse_json_object(self.content)


def strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped).removesuffix("```")
    return stripped.strip()


def _skip_string(text: str, index: int) -> int:
    """The index just past the JSON string literal starting at ``index``."""
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    return len(text)


def _balanced_object(text: str) -> str | None:
    """The first balanced ``{...}`` run in the text, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == '"':
            index = _skip_string(text, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
        index += 1
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply, tolerating fences and stray prose."""
    if not text or not text.strip():
        raise ResponseFormatError("The model returned an empty reply.")
    candidates = [strip_code_fences(text)]
    extracted = _balanced_object(candidates[0])
    if extracted and extracted != candidates[0]:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    preview = text.strip().replace("\n", " ")[:160]
    raise ResponseFormatError(f"Could not read a JSON object from the reply: {preview}")


def finish_reason(payload: dict[str, Any]) -> str | None:
    """Why the model stopped, normalised across providers."""
    choices = payload.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    reason = choice.get("finish_reason") or choice.get("native_finish_reason")
    return str(reason).lower() if reason else None


def message_content(payload: dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-shaped completion payload."""
    choices = payload.get("choices") or []
    if not choices:
        raise ResponseFormatError("The response carried no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # Some providers return content parts rather than a plain string.
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "".join(parts)
    if not isinstance(content, str):
        raise ResponseFormatError("The response carried no message content.")
    return content


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def text_message(role: str, text: str) -> dict[str, Any]:
    """A plain text chat message."""
    return {"role": role, "content": text}


def image_message(text: str, image_base64: str, media_type: str = "image/jpeg") -> dict[str, Any]:
    """A user message carrying a prompt and one inline image."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_base64}"},
            },
        ],
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class OpenRouterClient:
    """Thread-safe wrapper over the OpenRouter chat completions endpoint."""

    api_key: str | None = None
    base_url: str = BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff: float = DEFAULT_BACKOFF
    transport: httpx.BaseTransport | None = None
    require_key: bool = True

    _client: httpx.Client = field(init=False)
    _pricing: dict[str, ModelPricing] | None = field(init=False, default=None)
    _pricing_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _calls: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": REFERER,
            "X-Title": TITLE,
        }
        key = self.api_key
        if self.require_key:
            key = get_api_key(self.api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            transport=self.transport,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def calls(self) -> int:
        """How many completion requests this client has actually sent."""
        return self._calls

    # -- pricing -----------------------------------------------------------

    def pricing_table(self) -> dict[str, ModelPricing]:
        """The whole model list, fetched once per run and kept in memory."""
        with self._pricing_lock:
            if self._pricing is None:
                self._pricing = self._fetch_pricing()
            return self._pricing

    def _fetch_pricing(self) -> dict[str, ModelPricing]:
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            entries = response.json().get("data") or []
        except (httpx.HTTPError, ValueError):
            # Pricing is a nicety. A failed lookup must not stop classification.
            return {}
        table: dict[str, ModelPricing] = {}
        for entry in entries:
            if isinstance(entry, dict):
                parsed = parse_model_entry(entry)
                if parsed.model:
                    table[parsed.model] = parsed
        return table

    def pricing_for(self, model: str) -> ModelPricing:
        """Pricing for one model, falling back to zeroes when it is not listed."""
        table = self.pricing_table()
        if model in table:
            return table[model]
        # Variants such as ":free" or ":nitro" share the base model's pricing shape.
        base = model.split(":", 1)[0]
        return table.get(base, ModelPricing(model=model))

    # -- completions -------------------------------------------------------

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        images: int = 0,
    ) -> Completion:
        """One chat completion, asking for JSON back, with retry and cost accounting."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        response_format = self._response_format(model, json_schema)
        if response_format:
            body["response_format"] = response_format

        started = time.monotonic()
        payload, attempts = self._post_with_retry("/chat/completions", body)
        latency = time.monotonic() - started

        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        reported = usage.get("cost")
        if isinstance(reported, (int, float)):
            cost = float(reported)
        else:
            cost = self.pricing_for(model).cost(input_tokens, output_tokens, images)

        return Completion(
            model=model,
            content=message_content(payload),
            raw=payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_s=latency,
            attempts=attempts,
            finish_reason=finish_reason(payload),
        )

    def _response_format(
        self, model: str, json_schema: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Ask for the strictest JSON mode this model advertises."""
        pricing = self.pricing_for(model)
        table = self.pricing_table()
        if json_schema and pricing.supports_json_schema:
            return json_schema
        if pricing.supports_response_format or not table:
            # With no model list we cannot tell, so try the widely supported json_object mode.
            return {"type": "json_object"}
        return None

    def _post_with_retry(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                self._calls += 1
                response = self._client.post(path, json=body)
            except httpx.HTTPError as exc:
                last_error = exc
                self._sleep_before_retry(attempt, None)
                continue

            if response.status_code in RETRY_STATUS and attempt < self.max_attempts:
                self._sleep_before_retry(attempt, response.headers.get("retry-after"))
                continue
            if response.status_code >= httpx.codes.BAD_REQUEST:
                raise OpenRouterError(
                    f"OpenRouter returned {response.status_code}: {response.text[:300]}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise OpenRouterError("OpenRouter returned a non-JSON body.") from exc
            error = payload.get("error")
            if error:
                message = error.get("message") if isinstance(error, dict) else str(error)
                raise OpenRouterError(f"OpenRouter reported an error: {message}")
            return payload, attempt

        raise OpenRouterError(
            f"Gave up after {self.max_attempts} attempts talking to OpenRouter: {last_error}"
        )

    def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), MAX_BACKOFF))
            except ValueError:
                pass
            else:
                return
        delay = min(self.backoff * (2 ** (attempt - 1)), MAX_BACKOFF)
        time.sleep(delay * (0.5 + random.random() / 2))  # noqa: S311
