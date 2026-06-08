from __future__ import annotations

import json
from hashlib import sha1
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from proactive_assistant.model_gateway import ModelClient, ModelOutputValidationError, ModelRequest, ModelResponse
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import ModelUsageMetadata, PrivacyLevel, TranscriptWindowItem, openai_strict_json_schema
from proactive_assistant.runtime import MemoryCandidate, MemoryCandidateType, MemoryWritePolicy


MEMORY_EXTRACTION_SYSTEM_INSTRUCTIONS = """You extract durable memory candidates from meeting transcripts.

Return a JSON object only. The top-level JSON object must contain a candidates array.
Return only facts that are grounded in the supplied transcript or meeting state.
Do not invent names, owners, deadlines, decisions, or background context.
Prefer a small number of high-value candidates over exhaustive notes.
Use candidate_type, not type. Use source_refs for every candidate, usually transcript:{transcript_id}.
Include confidence for every candidate.

Candidate type policy:
- action_item: owner, deadline, next step, or task commitment.
- decision: explicit decision or conclusion.
- person_or_fact: people, roles, company/project facts, or factual answers worth recalling.
- project_context: recurring project background likely useful across meetings.
- summary: short meeting summary or unresolved gap check.
- user_preference / negative_preference / privacy_preference: only when the transcript explicitly states user preferences.

Write policy:
- eligible: high-confidence low-risk candidate.
- needs_confirmation: owner/deadline/privacy-sensitive or moderate-confidence candidate.
- blocked: low confidence, ungrounded, or unsafe candidate.

Keep text concise and retrieval-ready. Do not include private customer details unless they are essential and clearly grounded.
"""


class MemoryExtractionModelCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    candidate_type: MemoryCandidateType
    text: str = Field(min_length=1, max_length=800)
    confidence: float = Field(ge=0.0, le=1.0)
    write_policy: MemoryWritePolicy
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    source_refs: list[str] = Field(min_length=1, max_length=12)
    reason: str = Field(default="", max_length=400)
    entity: str = Field(default="", max_length=240)
    owner: str = Field(default="", max_length=120)
    deadline: str = Field(default="", max_length=120)
    status: str = Field(default="", max_length=120)
    topic: str = Field(default="", max_length=240)
    tags: list[str] = Field(default_factory=list, max_length=12)
    promotion_candidate: bool = False


class MemoryExtractionModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[MemoryExtractionModelCandidate] = Field(default_factory=list, max_length=20)
    extraction_notes: str = Field(default="", max_length=800)
    safety_flags: list[str] = Field(default_factory=list, max_length=12)


class MemoryExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    scenario_id: str = "meeting_business_v1"
    locale: str = "zh-CN"
    transcript_window: list[TranscriptWindowItem] = Field(min_length=1)
    session_context: dict[str, Any] = Field(default_factory=dict)
    meeting_state: dict[str, Any] = Field(default_factory=dict)
    memory_context: list[str] = Field(default_factory=list)
    privacy_constraints: list[str] = Field(default_factory=list)
    max_candidates: int = Field(default=8, ge=1, le=20)
    decision_id: str | None = None


class MemoryExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[MemoryCandidate] = Field(default_factory=list)
    extraction_notes: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    model_usage: ModelUsageMetadata | None = None


class MemoryExtractionService:
    """Structured LLM-backed memory extraction into runtime MemoryCandidate objects."""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        settings: ModelGatewaySettings | None = None,
    ) -> None:
        self._model_client = model_client
        self._settings = settings or ModelGatewaySettings()

    def extract_candidates(
        self,
        request: MemoryExtractionRequest,
        *,
        model: str | None = None,
    ) -> MemoryExtractionResult:
        model_request = ModelRequest(
            model=model or self._settings.default_model,
            instructions=MEMORY_EXTRACTION_SYSTEM_INSTRUCTIONS,
            input_text=_build_memory_extraction_input(request),
            response_schema_name="MemoryExtractionResult",
            response_schema=openai_strict_json_schema(MemoryExtractionModelOutput),
            max_output_tokens=self._settings.max_output_tokens,
            store=self._settings.store_model_responses,
            metadata={
                "session_id": request.session_id[:64],
                "scenario_id": request.scenario_id[:64],
                "contract": "memory_extraction_v1",
            },
        )
        response = self._model_client.generate_structured(model_request)
        return _parse_memory_extraction_result(response, request)


def _build_memory_extraction_input(request: MemoryExtractionRequest) -> str:
    payload = {
        "session_id": request.session_id,
        "scenario_id": request.scenario_id,
        "locale": request.locale,
        "max_candidates": request.max_candidates,
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


def _parse_memory_extraction_result(
    response: ModelResponse,
    request: MemoryExtractionRequest,
) -> MemoryExtractionResult:
    try:
        output = MemoryExtractionModelOutput.model_validate(_normalize_extraction_payload(response.parsed))
    except ValidationError as exc:
        raise ModelOutputValidationError("model output failed MemoryExtractionResult validation") from exc

    candidates = []
    for item in output.candidates[: request.max_candidates]:
        candidate = _memory_candidate_from_model_candidate(request, item)
        if MemoryWritePolicy(candidate.write_policy) == MemoryWritePolicy.BLOCKED:
            continue
        candidates.append(candidate)
    usage = ModelUsageMetadata(
        provider=response.provider,
        model=response.model,
        latency_ms=response.latency_ms,
        raw_response_id=response.raw_response_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached=response.cached,
    )
    return MemoryExtractionResult(
        candidates=candidates,
        extraction_notes=output.extraction_notes,
        safety_flags=list(output.safety_flags),
        model_usage=usage,
    )


def _normalize_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    raw_candidates = normalized.get("candidates", [])
    if raw_candidates is None:
        raw_candidates = []
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    normalized["candidates"] = [
        _normalize_extraction_candidate(candidate)
        for candidate in raw_candidates
        if isinstance(candidate, dict)
    ]
    normalized.setdefault("extraction_notes", "")
    normalized.setdefault("safety_flags", [])
    if isinstance(normalized.get("safety_flags"), str):
        normalized["safety_flags"] = [normalized["safety_flags"]]
    return normalized


def _normalize_extraction_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(candidate)
    if "candidate_type" not in normalized and "type" in normalized:
        normalized["candidate_type"] = normalized["type"]
    normalized.pop("type", None)
    if "text" not in normalized:
        for alias in ["memory_text", "content", "fact", "description", "summary"]:
            if normalized.get(alias):
                normalized["text"] = normalized[alias]
                break
    if "text" not in normalized:
        fallback_text = _fallback_candidate_text(normalized)
        if fallback_text:
            normalized["text"] = fallback_text
    for alias in ["memory_text", "content", "fact", "description", "summary"]:
        normalized.pop(alias, None)
    _absorb_compatible_extra_fields(normalized)
    normalized.setdefault("confidence", _default_extraction_confidence(normalized))
    normalized.setdefault("write_policy", "needs_confirmation")
    normalized.setdefault("privacy_level", "low")
    normalized.setdefault("privacy_risk", 0.0)
    normalized["source_refs"] = _normalize_source_refs(normalized)
    for key in ["source_ref", "sources", "source"]:
        normalized.pop(key, None)
    for key in ["reason", "entity", "owner", "deadline", "status", "topic"]:
        normalized[key] = _normalized_string(normalized.get(key))
    normalized.setdefault("tags", [])
    if isinstance(normalized["tags"], str):
        normalized["tags"] = [normalized["tags"]]
    normalized.setdefault("promotion_candidate", False)
    return {key: normalized[key] for key in _MODEL_CANDIDATE_FIELDS if key in normalized}


_MODEL_CANDIDATE_FIELDS = {
    "candidate_type",
    "text",
    "confidence",
    "write_policy",
    "privacy_level",
    "privacy_risk",
    "source_refs",
    "reason",
    "entity",
    "owner",
    "deadline",
    "status",
    "topic",
    "tags",
    "promotion_candidate",
}


def _absorb_compatible_extra_fields(candidate: dict[str, Any]) -> None:
    project = _normalized_string(candidate.pop("project", None))
    person = _normalized_string(candidate.pop("person", None))
    role = _normalized_string(candidate.pop("role", None))
    if project and not candidate.get("topic"):
        candidate["topic"] = project
    if person and not candidate.get("entity"):
        candidate["entity"] = person
    if role:
        existing_entity = _normalized_string(candidate.get("entity"))
        if person and existing_entity == person:
            candidate["entity"] = f"{person} ({role})"
        elif not existing_entity:
            candidate["entity"] = role
    tags = candidate.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []
    for value in [project, role]:
        if value:
            tags.append(value)
    if tags:
        candidate["tags"] = list(dict.fromkeys(str(tag) for tag in tags if str(tag)))


def _fallback_candidate_text(candidate: dict[str, Any]) -> str:
    candidate_type = str(candidate.get("candidate_type") or candidate.get("type") or "").strip()
    topic = _normalized_string(candidate.get("topic"))
    entity = _normalized_string(candidate.get("entity") or candidate.get("canonical_entity"))
    owner = _normalized_string(candidate.get("owner") or candidate.get("assignee"))
    deadline = _normalized_string(candidate.get("deadline") or candidate.get("normalized_deadline"))
    status = _normalized_string(candidate.get("status"))
    subject = entity or topic
    if not subject and not any([owner, deadline, status]):
        return ""
    if candidate_type == MemoryCandidateType.ACTION_ITEM.value:
        parts = [subject or "行动项"]
        if owner:
            parts.append(f"负责人：{owner}")
        if deadline:
            parts.append(f"截止：{deadline}")
        if status:
            parts.append(f"状态：{status}")
        return "；".join(parts) + "。"
    if candidate_type == MemoryCandidateType.PERSON_OR_FACT.value and owner and subject:
        return f"{owner} 与 {subject} 相关。"
    if candidate_type == MemoryCandidateType.DECISION.value:
        return f"{subject or topic or '决策'}：{status or '已决定'}。"
    parts = [value for value in [subject, f"负责人：{owner}" if owner else "", f"截止：{deadline}" if deadline else "", status] if value]
    return "；".join(parts) + "。"


def _normalized_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _default_extraction_confidence(candidate: dict[str, Any]) -> float:
    policy = str(candidate.get("write_policy", "")).strip().lower()
    if policy == "eligible":
        return 0.72
    if policy == "blocked":
        return 0.3
    return 0.62


def _normalize_source_refs(candidate: dict[str, Any]) -> list[str]:
    raw = (
        candidate.get("source_refs")
        or candidate.get("source_ref")
        or candidate.get("sources")
        or candidate.get("source")
        or []
    )
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    refs: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            value = item.get("ref") or item.get("id") or item.get("source_ref") or item.get("transcript_id")
        else:
            value = item
        if value is None:
            continue
        ref = str(value).strip()
        if not ref:
            continue
        if ref.startswith("transcript:"):
            refs.append(ref)
        elif ref.startswith("transcript_") or ref.startswith("seg_"):
            refs.append(f"transcript:{ref}")
        else:
            refs.append(ref)
    return list(dict.fromkeys(refs))


def _memory_candidate_from_model_candidate(
    request: MemoryExtractionRequest,
    item: MemoryExtractionModelCandidate,
) -> MemoryCandidate:
    candidate_type = MemoryCandidateType(item.candidate_type)
    privacy_level = PrivacyLevel(item.privacy_level)
    metadata = _candidate_metadata(request, item, candidate_type, privacy_level)
    return MemoryCandidate(
        memory_candidate_id=_candidate_id(request.session_id, item),
        decision_id=request.decision_id or f"llm_memory_extraction:{request.session_id}",
        session_id=request.session_id,
        source_event_ids=[],
        candidate_type=candidate_type,
        text=item.text,
        confidence=item.confidence,
        write_policy=_resolved_write_policy(item, candidate_type, privacy_level),
        privacy_level=privacy_level,
        reason=item.reason or "llm memory extraction",
        metadata=metadata,
    )


def _candidate_metadata(
    request: MemoryExtractionRequest,
    item: MemoryExtractionModelCandidate,
    candidate_type: MemoryCandidateType,
    privacy_level: PrivacyLevel,
) -> dict[str, Any]:
    tags = _candidate_tags(item, candidate_type)
    metadata: dict[str, Any] = {
        "org_id": str(request.session_context.get("org_id", request.session_context.get("metadata", {}).get("org_id", "default_org"))),
        "subject_user_id": str(
            request.session_context.get(
                "subject_user_id",
                request.session_context.get("metadata", {}).get("subject_user_id", "default_user"),
            )
        ),
        "session_id": request.session_id,
        "memory_extraction_source": "llm_memory_extraction_v1",
        "source_refs": list(item.source_refs),
        "provenance": list(item.source_refs),
        "privacy_level": privacy_level.value,
        "privacy_risk": item.privacy_risk,
        "tags": tags,
    }
    optional_fields = {
        "entity": item.entity,
        "canonical_entity": item.entity,
        "owner": item.owner,
        "assignee": item.owner,
        "deadline": item.deadline,
        "status": item.status,
        "topic": item.topic,
    }
    metadata.update({key: value for key, value in optional_fields.items() if value})
    transcript_ids = [_transcript_id_from_ref(ref) for ref in item.source_refs]
    transcript_ids = [transcript_id for transcript_id in transcript_ids if transcript_id]
    if transcript_ids:
        metadata["source_utterance_ids"] = transcript_ids
        metadata["source_utterance_id"] = transcript_ids[0]
    if item.promotion_candidate:
        metadata["promotion_target"] = "long_term"
    return metadata


def _candidate_tags(item: MemoryExtractionModelCandidate, candidate_type: MemoryCandidateType) -> list[str]:
    tags = {"llm_extracted", candidate_type.value}
    tags.update(str(tag) for tag in item.tags if str(tag))
    if item.owner:
        tags.add("owner")
    if item.deadline:
        tags.add("deadline")
    if item.status:
        tags.add("status")
    if item.promotion_candidate:
        tags.add("promotion_candidate")
    return sorted(tags)


def _resolved_write_policy(
    item: MemoryExtractionModelCandidate,
    candidate_type: MemoryCandidateType,
    privacy_level: PrivacyLevel,
) -> MemoryWritePolicy:
    policy = MemoryWritePolicy(item.write_policy)
    if policy == MemoryWritePolicy.BLOCKED:
        return policy
    if candidate_type == MemoryCandidateType.PRIVACY_PREFERENCE:
        return policy
    if item.confidence < 0.45:
        return MemoryWritePolicy.BLOCKED
    if item.confidence < 0.68:
        return MemoryWritePolicy.NEEDS_CONFIRMATION
    if privacy_level == PrivacyLevel.HIGH or item.privacy_risk >= 0.7:
        return MemoryWritePolicy.NEEDS_CONFIRMATION
    if item.owner or item.deadline:
        return MemoryWritePolicy.NEEDS_CONFIRMATION
    return policy


def _candidate_id(session_id: str, item: MemoryExtractionModelCandidate) -> str:
    payload = json.dumps(
        {
            "session_id": session_id,
            "candidate_type": str(item.candidate_type),
            "text": item.text,
            "source_refs": sorted(item.source_refs),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"memcand_llm_{digest}"


def _transcript_id_from_ref(ref: str) -> str:
    if ref.startswith("transcript:"):
        return ref.split(":", 1)[1]
    return ""
