from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelGatewayError(RuntimeError):
    """Base error for model gateway failures."""


class ModelOutputValidationError(ModelGatewayError):
    """Raised when a model response cannot be parsed into the expected schema."""


class ModelGatewayTimeoutError(ModelGatewayError):
    """Raised when a model call exceeds an explicit deadline.

    Used by the orchestrator to enforce a tight glasses-surface budget so
    the wearable does not present a stale popup that arrived too late.
    """


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    instructions: str = Field(min_length=1)
    input_text: str = Field(min_length=1)
    response_schema_name: str = Field(min_length=1, max_length=64)
    response_schema: dict[str, Any]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    store: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parsed: dict[str, Any]
    provider: str
    model: str
    latency_ms: int
    raw_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached: bool = False


class TextModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    instructions: str = Field(min_length=1)
    input_text: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class TextModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    provider: str
    model: str
    latency_ms: int
    raw_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached: bool = False


class ModelClient(Protocol):
    def generate_structured(self, request: ModelRequest) -> ModelResponse:
        """Generate structured JSON according to request.response_schema."""

    def generate_text(self, request: TextModelRequest) -> TextModelResponse:
        """Generate plain text for low-latency narrow routes."""


class OpenAIResponsesClient:
    """OpenAI Responses API client using JSON Schema structured outputs."""

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    def generate_structured(self, request: ModelRequest) -> ModelResponse:
        client = self._client or self._build_client()
        started = time.perf_counter()
        try:
            response = client.responses.create(
                model=request.model,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": request.instructions,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": request.input_text,
                            }
                        ],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": request.response_schema_name,
                        "schema": request.response_schema,
                        "strict": True,
                    }
                },
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                store=request.store,
                metadata=request.metadata,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise ModelGatewayError(f"OpenAI Responses request failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        parsed = _parse_response_json(response)
        usage = getattr(response, "usage", None)
        return ModelResponse(
            parsed=parsed,
            provider="openai",
            model=request.model,
            latency_ms=latency_ms,
            raw_response_id=getattr(response, "id", None),
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
        )

    def generate_text(self, request: TextModelRequest) -> TextModelResponse:
        client = self._client or self._build_client()
        started = time.perf_counter()
        try:
            response = client.responses.create(
                model=request.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": request.instructions}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": request.input_text}],
                    },
                ],
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                metadata=request.metadata,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise ModelGatewayError(f"OpenAI Responses text request failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        return TextModelResponse(
            text=_parse_response_text(response),
            provider="openai",
            model=request.model,
            latency_ms=latency_ms,
            raw_response_id=getattr(response, "id", None),
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
        )

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is installed in normal envs
            raise ModelGatewayError("openai package is not installed") from exc

        kwargs: dict[str, Any] = {"timeout": self._timeout_seconds, "max_retries": self._max_retries}
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._api_key is not None:
            kwargs["api_key"] = (
                self._api_key.get_secret_value()
                if isinstance(self._api_key, SecretStr)
                else self._api_key
            )
        return OpenAI(**kwargs)


class OpenAIChatCompletionsClient:
    """OpenAI-compatible Chat Completions client using the OpenAI SDK.

    This fits providers that expose an OpenAI-compatible endpoint such as:

    OpenAI(base_url="http://{addr}:58081", api_key=...).chat.completions.create(...)
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
        response_format: str = "json_schema",
        max_tokens_param: str = "max_tokens",
        provider: str = "openai_compatible_chat",
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._response_format = response_format
        self._max_tokens_param = max_tokens_param
        self._provider = provider
        # SDK default is 2 retries; each retry multiplies latency on a flaky
        # endpoint (503/timeout). For a realtime assistant prefer to fail fast
        # and fall back to rules rather than stall the wearable.
        self._max_retries = max_retries

    def generate_structured(self, request: ModelRequest) -> ModelResponse:
        client = self._client or self._build_client()
        started = time.perf_counter()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": request.instructions},
        ]
        # Endpoints that only support json_object (or no response_format) do
        # not natively enforce the JSON schema, so the model is free to
        # invent enum values and drop required fields. Inline the schema as
        # explicit guidance so non-strict endpoints still honor the contract.
        if self._response_format != "json_schema":
            messages.append({"role": "system", "content": _schema_guidance_message(request)})
        messages.append({"role": "user", "content": request.input_text})
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }
        response_format = _chat_response_format(request, self._response_format)
        if response_format is not None:
            kwargs["response_format"] = response_format
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            _set_chat_max_tokens(kwargs, self._max_tokens_param, request.max_output_tokens)

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ModelGatewayError(f"OpenAI Chat Completions request failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        parsed = _parse_chat_completion_json(response)
        usage = getattr(response, "usage", None)
        return ModelResponse(
            parsed=parsed,
            provider=self._provider,
            model=request.model,
            latency_ms=latency_ms,
            raw_response_id=getattr(response, "id", None),
            input_tokens=_usage_first(usage, "input_tokens", "prompt_tokens"),
            output_tokens=_usage_first(usage, "output_tokens", "completion_tokens"),
        )

    def generate_text(self, request: TextModelRequest) -> TextModelResponse:
        client = self._client or self._build_client()
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": request.input_text},
            ],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            _set_chat_max_tokens(kwargs, self._max_tokens_param, request.max_output_tokens)
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ModelGatewayError(f"OpenAI Chat Completions text request failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        return TextModelResponse(
            text=_parse_chat_completion_text(response),
            provider=self._provider,
            model=request.model,
            latency_ms=latency_ms,
            raw_response_id=getattr(response, "id", None),
            input_tokens=_usage_first(usage, "input_tokens", "prompt_tokens"),
            output_tokens=_usage_first(usage, "output_tokens", "completion_tokens"),
        )

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is installed in normal envs
            raise ModelGatewayError("openai package is not installed") from exc

        kwargs: dict[str, Any] = {"timeout": self._timeout_seconds, "max_retries": self._max_retries}
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._api_key is not None:
            kwargs["api_key"] = (
                self._api_key.get_secret_value()
                if isinstance(self._api_key, SecretStr)
                else self._api_key
            )
        return OpenAI(**kwargs)


class FakeModelClient:
    """Deterministic model client for tests and offline development."""

    def __init__(
        self,
        response: Mapping[str, Any] | Callable[[ModelRequest], Mapping[str, Any]],
        *,
        text_response: str | Callable[[TextModelRequest], str] | None = None,
        provider: str = "fake",
        latency_ms: int = 0,
    ) -> None:
        self._response = response
        self._text_response = text_response
        self.provider = provider
        self.latency_ms = latency_ms
        self.requests: list[ModelRequest] = []
        self.text_requests: list[TextModelRequest] = []

    def generate_structured(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        payload = self._response(request) if callable(self._response) else self._response
        return ModelResponse(
            parsed=dict(payload),
            provider=self.provider,
            model=request.model,
            latency_ms=self.latency_ms,
            cached=True,
        )

    def generate_text(self, request: TextModelRequest) -> TextModelResponse:
        self.text_requests.append(request)
        if callable(self._text_response):
            text = self._text_response(request)
        elif self._text_response is not None:
            text = self._text_response
        elif isinstance(self._response, Mapping):
            text = str(self._response.get("text") or self._response.get("answer") or "")
        else:
            text = ""
        return TextModelResponse(
            text=text,
            provider=self.provider,
            model=request.model,
            latency_ms=self.latency_ms,
            cached=True,
        )


def _parse_response_json(response: Any) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return _loads_json_object(output_text)

    output = getattr(response, "output", None) or []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                return _loads_json_object(text)

    raise ModelOutputValidationError("OpenAI response did not contain JSON output text")


def _parse_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    if parts:
        return "\n".join(parts)
    raise ModelOutputValidationError("OpenAI response did not contain text output")


def _parse_chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ModelOutputValidationError("chat completion response did not contain choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise ModelOutputValidationError("chat completion response did not contain a message")
    text = _message_text(message).strip()
    if not text:
        raise ModelOutputValidationError(
            "chat completion message did not contain content"
            f" ({_empty_chat_message_detail(choice, message)})"
        )
    return text


def _parse_chat_completion_json(response: Any) -> dict[str, Any]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ModelOutputValidationError("chat completion response did not contain choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise ModelOutputValidationError("chat completion response did not contain a message")

    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    content = _message_text(message)
    if not content.strip():
        raise ModelOutputValidationError(
            "chat completion message did not contain content"
            f" ({_empty_chat_message_detail(choice, message)})"
        )
    return _loads_json_object_maybe_embedded(content)


def _message_text(message: Any) -> str:
    values = [
        getattr(message, "content", None),
        getattr(message, "text", None),
        getattr(message, "output_text", None),
        getattr(message, "reasoning_content", None),
    ]
    if isinstance(message, dict):
        values.extend(
            [
                message.get("content"),
                message.get("text"),
                message.get("output_text"),
                message.get("reasoning_content"),
            ]
        )
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        values.extend(
            [
                model_extra.get("content"),
                model_extra.get("text"),
                model_extra.get("output_text"),
                model_extra.get("reasoning_content"),
            ]
        )
    return "".join(_message_content_text(value) for value in values if value is not None)


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _empty_chat_message_detail(choice: Any, message: Any) -> str:
    finish_reason = getattr(choice, "finish_reason", None)
    if isinstance(choice, dict):
        finish_reason = choice.get("finish_reason", finish_reason)
    fields = sorted(_message_field_names(message))
    field_text = ",".join(fields) if fields else "none"
    return f"finish_reason={finish_reason or 'unknown'}, message_fields={field_text}"


def _message_field_names(message: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(message, dict):
        fields.update(str(key) for key in message)
    else:
        fields.update(str(key) for key in vars(message).keys())
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        fields.update(str(key) for key in model_extra)
    return fields


def _schema_guidance_message(request: ModelRequest) -> str:
    """Inline JSON-schema guidance for endpoints lacking strict enforcement.

    Compact but explicit: the model must use only allowed enum values and
    include every required field. Mirrors what strict json_schema mode would
    enforce natively on OpenAI's first-party API.
    """

    schema_json = json.dumps(request.response_schema, ensure_ascii=False, separators=(",", ":"))
    return (
        "You must return ONLY a single JSON object that strictly conforms to the "
        "following JSON Schema. Use only the allowed enum values. Include every "
        "property listed under \"required\". Do not add properties that are not in "
        f"the schema.\n\nJSON Schema for \"{request.response_schema_name}\":\n{schema_json}"
    )


def _chat_response_format(request: ModelRequest, mode: str) -> dict[str, Any] | None:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_schema_name,
                "schema": request.response_schema,
                "strict": True,
            },
        }
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "none":
        return None
    raise ModelGatewayError(f"unsupported chat response format: {mode}")


def _set_chat_max_tokens(kwargs: dict[str, Any], param_name: str, value: int) -> None:
    if param_name == "none":
        return
    if param_name not in {"max_tokens", "max_completion_tokens"}:
        raise ModelGatewayError(f"unsupported chat max token parameter: {param_name}")
    kwargs[param_name] = value


def _loads_json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ModelOutputValidationError("model output was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ModelOutputValidationError("model output JSON must be an object")
    return parsed


def _loads_json_object_maybe_embedded(value: str) -> dict[str, Any]:
    stripped = _strip_json_fence(value.strip())
    try:
        return _loads_json_object(stripped)
    except ModelOutputValidationError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ModelOutputValidationError("model output was not valid JSON")


def _strip_json_fence(value: str) -> str:
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if not lines:
        return value
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _usage_value(usage: Any, key: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, key, None)
    return value if isinstance(value, int) else None


def _usage_first(usage: Any, *keys: str) -> int | None:
    for key in keys:
        value = _usage_value(usage, key)
        if value is not None:
            return value
    return None
