"""LLM-based query understanding for memory retrieval.

Replaces the hard-coded ``_classify_intent``, ``_resolve_target_entity``,
and ``_normalize_dates`` paths in ``MemoryRetriever`` for the cases where
the rules underfit. A single LLM call returns a structured
``QueryUnderstandingResult`` containing intent, target entity, anaphora
resolution, time window, and referenced speaker.

Design notes:
- ``QueryUnderstandingService.understand`` is the only public entry. It
  caches by (query_text, prompt_category) within a short TTL so calling
  it on the same transcript chunk multiple times does not multiply LLM
  cost.
- ``MemoryRetriever.__init__`` accepts an optional service; if confidence
  is below threshold or any error occurs, the legacy rule path runs as
  fallback.
- Time-window extraction emits ISO date strings, plus a relative flag so
  the caller can decide whether to anchor against meeting-now or the
  user's wall clock.
- Reasoning surfaces in ``StructuredMemoryQuery`` as additional fields
  so the downstream feature-score functions can use them.

The system instruction is intentionally short so the fast model can
respond in <500ms.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from proactive_assistant.memory.contracts import MemoryRetrievalIntent
from proactive_assistant.model_gateway import (
    ModelClient,
    ModelGatewayError,
    ModelOutputValidationError,
    ModelRequest,
)
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import (
    PromptCategory,
    openai_strict_json_schema,
)


QUERY_UNDERSTANDING_SYSTEM_INSTRUCTIONS = """You classify a memory retrieval query.

Read the user query text plus optional recent transcript context. Return JSON only.

Intent codes:
- lookup_deadline: user is asking when something is due
- lookup_owner: user is asking who is responsible
- lookup_status: user is asking how it is going / current state
- lookup_rationale: user is asking why a decision was made / background
- lookup_task_list: user is asking for "what do I need to do" / outstanding items
- lookup_schedule: user is asking what is on their schedule / upcoming events
- open_recall: general retrieval, no specific intent

Fields:
- target_entity: the topic the user is asking about (e.g. "客户报价确认"). Empty if unclear.
- target_entity_confidence: 0-1 how sure you are the entity name is correct.
- anaphora_resolved: true when the query had "this/that/它" that you resolved to an entity in context.
- time_window_start: ISO date "YYYY-MM-DD" when query implies a window start. Empty if none.
- time_window_end: ISO date or empty.
- time_is_relative: true when the time expression was relative ("上周", "明天") rather than absolute.
- referenced_speaker: name of a speaker the query refers to, empty if none.
- confidence: 0-1 how confident you are in the overall classification.
- rationale: one sentence explaining the choice. Plain text.

Resolve anaphora using recent transcript context only. Do not invent dates."""


class QueryUnderstandingModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    intent: MemoryRetrievalIntent
    target_entity: str = Field(default="", max_length=240)
    target_entity_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    anaphora_resolved: bool = False
    time_window_start: str = Field(default="", max_length=32)
    time_window_end: str = Field(default="", max_length=32)
    time_is_relative: bool = False
    referenced_speaker: str = Field(default="", max_length=120)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=400)


class QueryUnderstandingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    query_text: str
    prompt_category: PromptCategory | None = None
    recent_transcript_text: str = ""
    active_entities: list[str] = Field(default_factory=list)
    reference_date_iso: str = ""


class QueryUnderstandingResult(BaseModel):
    """Structured understanding of a memory retrieval query.

    Note: ``use_enum_values=True`` means ``intent`` is stored as its str
    value, not the enum member. Callers that need enum semantics must
    re-wrap: ``MemoryRetrievalIntent(result.intent)`` (as
    ``build_structured_query`` does).
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    intent: MemoryRetrievalIntent
    target_entity: str | None
    target_entity_confidence: float
    anaphora_resolved: bool
    time_window_start: str | None
    time_window_end: str | None
    time_is_relative: bool
    referenced_speaker: str | None
    confidence: float
    rationale: str
    cached: bool = False


class _LRUCache:
    """Tiny TTL-aware LRU. Not thread-safe; for single-process demos."""

    def __init__(self, *, max_size: int = 128, ttl_seconds: float = 30.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._items: OrderedDict[str, tuple[float, QueryUnderstandingResult]] = OrderedDict()

    def get(self, key: str) -> QueryUnderstandingResult | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        stored_at, result = entry
        if time.monotonic() - stored_at > self._ttl:
            self._items.pop(key, None)
            return None
        # Refresh recency
        self._items.move_to_end(key)
        return result

    def set(self, key: str, value: QueryUnderstandingResult) -> None:
        self._items[key] = (time.monotonic(), value)
        self._items.move_to_end(key)
        while len(self._items) > self._max_size:
            self._items.popitem(last=False)


class QueryUnderstandingService:
    """LLM-backed query understanding with caching and rules fallback.

    The retriever owns the rules fallback; this service does one job:
    issue exactly one LLM call per uncached query and return a structured
    result. Errors (network, validation, low confidence) propagate as
    ``None`` so the retriever picks up the rule path.
    """

    def __init__(
        self,
        *,
        model_client: ModelClient,
        settings: ModelGatewaySettings | None = None,
        confidence_threshold: float = 0.6,
        cache: _LRUCache | None = None,
    ) -> None:
        self._model_client = model_client
        self._settings = settings or ModelGatewaySettings()
        self._confidence_threshold = confidence_threshold
        self._cache = cache if cache is not None else _LRUCache()

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    def understand(
        self,
        request: QueryUnderstandingRequest,
        *,
        model: str | None = None,
    ) -> QueryUnderstandingResult | None:
        if not request.query_text.strip():
            return None
        cache_key = _cache_key(request)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"cached": True})

        resolved_model = model or self._settings.fast_model or self._settings.default_model
        model_request = ModelRequest(
            model=resolved_model,
            instructions=QUERY_UNDERSTANDING_SYSTEM_INSTRUCTIONS,
            input_text=_build_input(request),
            response_schema_name="QueryUnderstandingResult",
            response_schema=openai_strict_json_schema(QueryUnderstandingModelOutput),
            max_output_tokens=min(self._settings.max_output_tokens, 400),
            store=self._settings.store_model_responses,
            metadata={"contract": "query_understanding_v1"},
        )
        try:
            response = self._model_client.generate_structured(model_request)
        except ModelGatewayError:
            return None
        try:
            output = QueryUnderstandingModelOutput.model_validate(response.parsed)
        except (ValidationError, ModelOutputValidationError):
            return None
        if output.confidence < self._confidence_threshold:
            return None
        result = QueryUnderstandingResult(
            intent=output.intent,
            target_entity=output.target_entity.strip() or None,
            target_entity_confidence=output.target_entity_confidence,
            anaphora_resolved=output.anaphora_resolved,
            time_window_start=_validated_iso_date(output.time_window_start),
            time_window_end=_validated_iso_date(output.time_window_end),
            time_is_relative=output.time_is_relative,
            referenced_speaker=output.referenced_speaker.strip() or None,
            confidence=output.confidence,
            rationale=output.rationale.strip(),
        )
        self._cache.set(cache_key, result)
        return result


def _build_input(request: QueryUnderstandingRequest) -> str:
    payload = {
        "query_text": request.query_text,
        "prompt_category": (
            request.prompt_category.value
            if hasattr(request.prompt_category, "value")
            else request.prompt_category
        ),
        "recent_transcript_text": request.recent_transcript_text[:600],
        "active_entities": list(request.active_entities)[:12],
        "reference_date_iso": request.reference_date_iso,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _validated_iso_date(value: str) -> str | None:
    """Return the trimmed value only when it parses as an ISO date.

    The LLM is instructed to emit "YYYY-MM-DD" but may disobey (e.g.
    "下周"). A non-ISO date string flowing into the store-level time
    window filter would empty the whole candidate pool (the filter treats
    an unparseable query window as matching nothing), so invalid values
    are dropped here and the regex-based inference in
    ``build_structured_query`` gets a chance to fill the gap instead.
    """

    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        date.fromisoformat(cleaned[:10])
    except ValueError:
        return None
    return cleaned[:10]


def _cache_key(request: QueryUnderstandingRequest) -> str:
    payload = {
        "query_text": request.query_text.strip().lower(),
        "prompt_category": (
            request.prompt_category.value
            if hasattr(request.prompt_category, "value")
            else request.prompt_category
        ),
        "recent_transcript_text": request.recent_transcript_text.strip().lower()[:600],
        "active_entities": [entity.strip().lower() for entity in request.active_entities[:12]],
        "reference_date_iso": request.reference_date_iso or "",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
