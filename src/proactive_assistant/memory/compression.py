from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from proactive_assistant.memory.extraction import (
    MemoryExtractionModelCandidate,
    MemoryExtractionRequest,
    _memory_candidate_from_model_candidate,
    _normalize_extraction_candidate,
    _normalize_source_refs,
)
from proactive_assistant.model_gateway import ModelClient, ModelOutputValidationError, ModelRequest, ModelResponse
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import ModelUsageMetadata, TranscriptWindowItem, openai_strict_json_schema
from proactive_assistant.runtime import MemoryCandidate


MEMORY_COMPRESSION_SYSTEM_INSTRUCTIONS = """You compress meeting transcript chunks into durable memory-ready summaries.

Return a JSON object only.
Your job is not to rewrite the whole transcript. Preserve only information that is useful for later proactive assistance.
Every summary item and candidate memory must be grounded in source_refs, usually transcript:{transcript_id}.
Do not invent people, deadlines, project facts, decisions, or user preferences.

Compression goals:
- chunk_summary: concise faithful summary of the transcript chunk.
- key_points: important facts, decisions, commitments, and context.
- open_questions: unresolved gaps, missing owner/deadline, risks, or follow-ups.
- candidate_memories: durable atomic memories using the same candidate_type policy as memory extraction.
- compression_quality: estimate how much useful information was retained.
- loss_risk_score: estimate risk that important details were omitted or ambiguous.

Prefer structured atomic memories over a long prose summary. Keep summaries short and retrieval-ready.
If the transcript is low-value small talk, return a short summary and no candidate_memories.
"""


class MemoryCompressionModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_summary: str = Field(min_length=1, max_length=1200)
    key_points: list[str] = Field(default_factory=list, max_length=12)
    open_questions: list[str] = Field(default_factory=list, max_length=12)
    candidate_memories: list[MemoryExtractionModelCandidate] = Field(default_factory=list, max_length=20)
    source_refs: list[str] = Field(min_length=1, max_length=24)
    compression_quality: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    loss_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    compression_notes: str = Field(default="", max_length=800)
    safety_flags: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def quality_and_source_refs_must_be_consistent(self) -> "MemoryCompressionModelOutput":
        if not self.source_refs:
            raise ValueError("source_refs are required")
        if self.candidate_memories:
            candidate_refs = {ref for candidate in self.candidate_memories for ref in candidate.source_refs}
            if not candidate_refs.issubset(set(self.source_refs)):
                raise ValueError("candidate memory source_refs must be included in top-level source_refs")
        return self


class MemoryCompressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    scenario_id: str = "meeting_business_v1"
    locale: str = "zh-CN"
    transcript_window: list[TranscriptWindowItem] = Field(min_length=1)
    session_context: dict[str, Any] = Field(default_factory=dict)
    meeting_state: dict[str, Any] = Field(default_factory=dict)
    memory_context: list[str] = Field(default_factory=list)
    privacy_constraints: list[str] = Field(default_factory=list)
    max_candidate_memories: int = Field(default=8, ge=0, le=20)
    decision_id: str | None = None


class MemoryCompressionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    chunk_id: str
    chunk_summary: str
    key_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    candidate_memories: list[MemoryCandidate] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    compression_quality: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    loss_risk_score: float = Field(ge=0.0, le=1.0)
    compression_notes: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    model_usage: ModelUsageMetadata | None = None


class MemoryCompressionService:
    """LLM-backed transcript chunk compression into summaries and memory candidates."""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        settings: ModelGatewaySettings | None = None,
    ) -> None:
        self._model_client = model_client
        self._settings = settings or ModelGatewaySettings()

    def compress_chunk(
        self,
        request: MemoryCompressionRequest,
        *,
        model: str | None = None,
    ) -> MemoryCompressionResult:
        model_request = ModelRequest(
            model=model or self._settings.default_model,
            instructions=MEMORY_COMPRESSION_SYSTEM_INSTRUCTIONS,
            input_text=_build_memory_compression_input(request),
            response_schema_name="MemoryCompressionResult",
            response_schema=openai_strict_json_schema(MemoryCompressionModelOutput),
            max_output_tokens=self._settings.max_output_tokens,
            store=self._settings.store_model_responses,
            metadata={
                "session_id": request.session_id[:64],
                "chunk_id": request.chunk_id[:64],
                "scenario_id": request.scenario_id[:64],
                "contract": "memory_compression_v1",
            },
        )
        response = self._model_client.generate_structured(model_request)
        return _parse_memory_compression_result(response, request)


def _build_memory_compression_input(request: MemoryCompressionRequest) -> str:
    payload = {
        "session_id": request.session_id,
        "chunk_id": request.chunk_id,
        "scenario_id": request.scenario_id,
        "locale": request.locale,
        "max_candidate_memories": request.max_candidate_memories,
        "privacy_constraints": request.privacy_constraints,
        "session_context": request.session_context,
        "memory_context": request.memory_context,
        "meeting_state": request.meeting_state,
        "transcript_window": [
            {
                "transcript_id": item.transcript_id,
                "speaker": item.speaker,
                "timestamp_ms": item.timestamp_ms,
                "topic": item.topic,
                "text": item.text,
            }
            for item in request.transcript_window
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def _parse_memory_compression_result(
    response: ModelResponse,
    request: MemoryCompressionRequest,
) -> MemoryCompressionResult:
    try:
        normalized_payload = _normalize_compression_payload(response.parsed)
        if not normalized_payload.get("source_refs"):
            normalized_payload["source_refs"] = [
                f"transcript:{item.transcript_id}" for item in request.transcript_window
            ]
        output = MemoryCompressionModelOutput.model_validate(normalized_payload)
    except ValidationError as exc:
        raise ModelOutputValidationError("model output failed MemoryCompressionResult validation") from exc

    extraction_request = MemoryExtractionRequest(
        session_id=request.session_id,
        scenario_id=request.scenario_id,
        locale=request.locale,
        transcript_window=request.transcript_window,
        session_context=request.session_context,
        meeting_state=request.meeting_state,
        memory_context=request.memory_context,
        privacy_constraints=request.privacy_constraints,
        max_candidates=max(1, request.max_candidate_memories) if request.max_candidate_memories else 1,
        decision_id=request.decision_id or f"llm_memory_compression:{request.session_id}:{request.chunk_id}",
    )
    candidates = [
        _memory_candidate_from_model_candidate(extraction_request, candidate)
        for candidate in output.candidate_memories[: request.max_candidate_memories]
    ]
    usage = ModelUsageMetadata(
        provider=response.provider,
        model=response.model,
        latency_ms=response.latency_ms,
        raw_response_id=response.raw_response_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached=response.cached,
    )
    return MemoryCompressionResult(
        session_id=request.session_id,
        chunk_id=request.chunk_id,
        chunk_summary=output.chunk_summary,
        key_points=list(output.key_points),
        open_questions=list(output.open_questions),
        candidate_memories=candidates,
        source_refs=list(output.source_refs),
        compression_quality=output.compression_quality,
        coverage_score=output.coverage_score,
        loss_risk_score=output.loss_risk_score,
        compression_notes=output.compression_notes,
        safety_flags=list(output.safety_flags),
        model_usage=usage,
    )


def _normalize_compression_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "candidate_memories" not in normalized and "candidates" in normalized:
        normalized["candidate_memories"] = normalized["candidates"]
    normalized.pop("candidates", None)
    normalized["chunk_summary"] = _summary_text(normalized.get("chunk_summary") or normalized.get("summary"))
    normalized["key_points"] = _string_list(normalized.get("key_points"))
    normalized["open_questions"] = _string_list(normalized.get("open_questions"))
    raw_candidates = normalized.get("candidate_memories", [])
    if raw_candidates is None:
        raw_candidates = []
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    normalized_candidates = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        normalized_candidate = _normalize_extraction_candidate(candidate)
        if not normalized_candidate.get("text"):
            continue
        normalized_candidates.append(normalized_candidate)
    normalized["candidate_memories"] = normalized_candidates
    normalized["source_refs"] = _normalize_compression_source_refs(normalized)
    for key in ["source_ref", "sources", "source"]:
        normalized.pop(key, None)
    normalized["compression_quality"] = _float_score(
        normalized.get("compression_quality"),
        default=_default_compression_quality(normalized),
    )
    normalized["coverage_score"] = _float_score(
        normalized.get("coverage_score"),
        default=normalized["compression_quality"],
    )
    normalized["loss_risk_score"] = _float_score(
        normalized.get("loss_risk_score"),
        default=round(1.0 - normalized["coverage_score"], 4),
    )
    normalized["compression_notes"] = str(normalized.get("compression_notes") or normalized.get("notes") or "")
    normalized["safety_flags"] = _string_list(normalized.get("safety_flags"))
    return {
        key: normalized[key]
        for key in {
            "chunk_summary",
            "key_points",
            "open_questions",
            "candidate_memories",
            "source_refs",
            "compression_quality",
            "coverage_score",
            "loss_risk_score",
            "compression_notes",
            "safety_flags",
        }
        if key in normalized
    }


def _normalize_compression_source_refs(payload: dict[str, Any]) -> list[str]:
    refs = _normalize_source_refs(payload)
    candidate_refs: list[str] = []
    for candidate in payload.get("candidate_memories", []):
        if isinstance(candidate, dict):
            candidate_refs.extend(str(ref) for ref in candidate.get("source_refs", []) if str(ref))
        elif isinstance(candidate, MemoryExtractionModelCandidate):
            candidate_refs.extend(candidate.source_refs)
    item_refs: list[str] = []
    for field in ["key_points", "open_questions"]:
        raw_items = payload.get(field, [])
        if not isinstance(raw_items, list):
            raw_items = [raw_items]
        for item in raw_items:
            if isinstance(item, dict):
                item_refs.extend(_normalize_source_refs(item))
    return list(dict.fromkeys([*refs, *candidate_refs, *item_refs]))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = item.get("point") or item.get("question") or item.get("text") or item.get("content")
            if text:
                items.append(str(text))
            continue
        if str(item):
            items.append(str(item))
    return items


def _summary_text(value: Any) -> str:
    if isinstance(value, dict):
        text = value.get("text") or value.get("summary") or value.get("content")
        return str(text or "").strip()
    return str(value or "").strip()


def _default_compression_quality(payload: dict[str, Any]) -> float:
    summary = str(payload.get("chunk_summary", ""))
    candidate_count = len(payload.get("candidate_memories", []))
    refs = len(payload.get("source_refs", []))
    score = 0.45
    if len(summary) >= 20:
        score += 0.2
    if candidate_count:
        score += 0.2
    if refs:
        score += 0.1
    return round(min(0.95, score), 4)


def _float_score(value: Any, *, default: float) -> float:
    if isinstance(value, dict):
        for key in ["score", "value", "rating"]:
            if key in value:
                return _float_score(value[key], default=default)
        return default
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(1.0, max(0.0, score)), 4)
