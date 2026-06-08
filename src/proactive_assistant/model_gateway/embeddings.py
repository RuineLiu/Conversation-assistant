from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from proactive_assistant.model_gateway.clients import ModelGatewayError


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embeddings: list[list[float]]
    provider: str
    model: str
    latency_ms: int
    raw_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    cached: bool = False


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str], *, model: str) -> EmbeddingResponse:
        """Return one embedding vector for each input text."""


class OpenAIEmbeddingClient:
    """OpenAI-compatible embedding API client."""

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
        provider: str = "openai_embeddings",
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._provider = provider

    def embed_texts(self, texts: Sequence[str], *, model: str) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(embeddings=[], provider=self._provider, model=model, latency_ms=0, cached=True)

        client = self._client or self._build_client()
        started = time.perf_counter()
        response = client.embeddings.create(model=model, input=list(texts), timeout=self._timeout_seconds)
        latency_ms = int((time.perf_counter() - started) * 1000)
        data = getattr(response, "data", None) or []
        embeddings = [list(getattr(item, "embedding")) for item in sorted(data, key=lambda item: getattr(item, "index", 0))]
        usage = getattr(response, "usage", None)
        if len(embeddings) != len(texts):
            raise ModelGatewayError("embedding response count did not match input count")
        return EmbeddingResponse(
            embeddings=embeddings,
            provider=self._provider,
            model=model,
            latency_ms=latency_ms,
            raw_response_id=getattr(response, "id", None),
            input_tokens=_usage_value(usage, "prompt_tokens"),
        )

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is installed in normal envs
            raise ModelGatewayError("openai package is not installed") from exc

        kwargs: dict[str, Any] = {"timeout": self._timeout_seconds}
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._api_key is not None:
            kwargs["api_key"] = (
                self._api_key.get_secret_value()
                if isinstance(self._api_key, SecretStr)
                else self._api_key
            )
        return OpenAI(**kwargs)


class FakeEmbeddingClient:
    """Deterministic embedding client for tests and offline retrieval development."""

    def __init__(
        self,
        embeddings: Mapping[str, Sequence[float]] | Callable[[str], Sequence[float]] | None = None,
        *,
        dimensions: int = 64,
        provider: str = "fake_embeddings",
    ) -> None:
        self._embeddings = embeddings
        self._dimensions = dimensions
        self.provider = provider
        self.requests: list[tuple[str, list[str]]] = []

    def embed_texts(self, texts: Sequence[str], *, model: str) -> EmbeddingResponse:
        self.requests.append((model, list(texts)))
        return EmbeddingResponse(
            embeddings=[self._embed_text(text) for text in texts],
            provider=self.provider,
            model=model,
            latency_ms=0,
            cached=True,
        )

    def _embed_text(self, text: str) -> list[float]:
        if callable(self._embeddings):
            return _normalized(list(float(value) for value in self._embeddings(text)))
        if isinstance(self._embeddings, Mapping) and text in self._embeddings:
            return _normalized(list(float(value) for value in self._embeddings[text]))
        return _hashed_text_embedding(text, dimensions=self._dimensions)


def _hashed_text_embedding(text: str, *, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in _embedding_tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return _normalized(vector)


def _embedding_tokens(text: str) -> list[str]:
    normalized = text.lower()
    tokens: list[str] = re.findall(r"[a-z0-9_]+", normalized)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        tokens.append(chunk)
        for size in (2, 3):
            if len(chunk) >= size:
                tokens.extend(chunk[index : index + size] for index in range(0, len(chunk) - size + 1))
    return tokens


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _usage_value(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, name, None)
    if isinstance(usage, dict):
        value = usage.get(name, value)
    return int(value) if isinstance(value, int | float) else None
