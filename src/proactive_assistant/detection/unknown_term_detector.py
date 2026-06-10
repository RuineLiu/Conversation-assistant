"""LLM-as-detector for unfamiliar terms and acronyms in meeting transcripts.

This module isolates a single model call: given a recent transcript window
plus the user's known vocabulary, decide which terms a typical knowledge
worker would not recognize, and produce a one-line explanation for each.

Detection and explanation are produced in the same call. The downstream
orchestrator can therefore fast-path the candidate straight into a
``PromptGenerationResult`` without a second LLM round trip.

The module mirrors the shape of ``MemoryExtractionService``: strict
Pydantic models for model output, an offline-safe contract that works
with ``FakeModelClient`` in tests, and a small business-facing service.
"""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha1
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from proactive_assistant.model_gateway import (
    ModelClient,
    ModelOutputValidationError,
    ModelRequest,
    ModelResponse,
)
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import (
    ModelUsageMetadata,
    PrivacyLevel,
    TranscriptWindowItem,
    openai_strict_json_schema,
)


UNKNOWN_TERM_SYSTEM_INSTRUCTIONS = """You detect unfamiliar terms in meeting transcripts and produce short explanations.

Return a JSON object only. The top-level object must contain a candidates array.

For each transcript window decide which terms a typical knowledge worker would NOT recognize.
Each candidate must be a single term that actually appears in the transcript.

Skip terms that:
- already appear in known_vocabulary,
- are common general-knowledge words (NBA, KFC, USA, OK, etc.),
- the same speaker explicitly defines in the same sentence
  ("X is ...", "X 是 ...", "X stands for ..."),
- relate to private business data the user marked sensitive.

For each kept term, produce explanation:
- Chinese explanation MUST be ≤30 Chinese characters.
- English explanation MUST be ≤20 English words.
- Be plain and useful in a meeting context, not encyclopedic.
- Do not invent facts; if you cannot give a safe explanation, drop the term.

term_type policy:
- acronym: ASCII-letter abbreviation (e.g. GMV, OKR, gRPC).
- jargon: a common word with a specialized meaning here (e.g. "bucket" as data partition).
- named_entity: a project name, product name, company, internal system.
- concept: a domain concept that needs unpacking (e.g. "回流").

confidence reflects how confident you are the term is unfamiliar AND your explanation is correct.
privacy_level / privacy_risk follow the same scale as the rest of the system.
source_segment_id MUST be one of the transcript_id values supplied in the input.

Output empty candidates array if nothing in the window is worth explaining."""


class UnknownTermType(StrEnum):
    ACRONYM = "acronym"
    JARGON = "jargon"
    NAMED_ENTITY = "named_entity"
    CONCEPT = "concept"


class UnknownTermModelCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    term: str = Field(min_length=1, max_length=64)
    term_type: UnknownTermType
    explanation: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    source_segment_id: str = Field(min_length=1, max_length=128)
    rationale: str = Field(default="", max_length=400)


class UnknownTermModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[UnknownTermModelCandidate] = Field(default_factory=list, max_length=8)
    detection_notes: str = Field(default="", max_length=400)
    safety_flags: list[str] = Field(default_factory=list, max_length=8)


class UnknownTermDetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    locale: str = "zh-CN"
    transcript_window: list[TranscriptWindowItem] = Field(min_length=1)
    known_vocabulary: list[str] = Field(default_factory=list, max_length=128)
    explained_in_session: list[str] = Field(default_factory=list, max_length=64)
    privacy_constraints: list[str] = Field(default_factory=list)
    max_candidates: int = Field(default=3, ge=1, le=8)


class UnknownTermCandidate(BaseModel):
    """Detection result enriched with a stable opportunity id."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    candidate_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    term: str = Field(min_length=1)
    term_type: UnknownTermType
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    privacy_level: PrivacyLevel
    privacy_risk: float = Field(ge=0.0, le=1.0)
    source_segment_id: str = Field(min_length=1)
    rationale: str = ""


class UnknownTermDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[UnknownTermCandidate] = Field(default_factory=list)
    detection_notes: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    model_usage: ModelUsageMetadata | None = None


class UnknownTermDetector:
    """LLM-backed detector for unfamiliar terms and acronyms.

    The detector issues exactly one model call per detection request and
    returns a list of ``UnknownTermCandidate`` with stable ids. The
    downstream ``PromptOpportunityDetector`` adapts these into
    ``PromptOpportunity`` records so the orchestrator can fast-path them
    without a second LLM round trip.
    """

    def __init__(
        self,
        *,
        model_client: ModelClient,
        settings: ModelGatewaySettings | None = None,
    ) -> None:
        self._model_client = model_client
        self._settings = settings or ModelGatewaySettings()

    def detect(
        self,
        request: UnknownTermDetectionRequest,
        *,
        model: str | None = None,
    ) -> UnknownTermDetectionResult:
        # Prefer the configured fast model for low-latency detection,
        # falling back to the default model if no fast model is set.
        resolved_model = model or self._settings.fast_model or self._settings.default_model
        model_request = ModelRequest(
            model=resolved_model,
            instructions=UNKNOWN_TERM_SYSTEM_INSTRUCTIONS,
            input_text=_build_detection_input(request),
            response_schema_name="UnknownTermDetectionResult",
            response_schema=openai_strict_json_schema(UnknownTermModelOutput),
            max_output_tokens=min(self._settings.max_output_tokens, 600),
            store=self._settings.store_model_responses,
            metadata={
                "session_id": request.session_id[:64],
                "contract": "unknown_term_detection_v1",
            },
        )
        response = self._model_client.generate_structured(model_request)
        return _parse_detection_result(response, request)


def _build_detection_input(request: UnknownTermDetectionRequest) -> str:
    payload = {
        "session_id": request.session_id,
        "locale": request.locale,
        "max_candidates": request.max_candidates,
        "known_vocabulary": list(request.known_vocabulary),
        "explained_in_session": list(request.explained_in_session),
        "privacy_constraints": list(request.privacy_constraints),
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
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _parse_detection_result(
    response: ModelResponse,
    request: UnknownTermDetectionRequest,
) -> UnknownTermDetectionResult:
    try:
        output = UnknownTermModelOutput.model_validate(response.parsed)
    except ValidationError as exc:
        raise ModelOutputValidationError("model output failed UnknownTermDetectionResult validation") from exc

    valid_segment_ids = {item.transcript_id for item in request.transcript_window}
    known_vocab_norm = {term.strip().lower() for term in request.known_vocabulary if term.strip()}
    explained_norm = {term.strip().lower() for term in request.explained_in_session if term.strip()}

    deduped_terms: set[str] = set()
    candidates: list[UnknownTermCandidate] = []
    for raw in output.candidates:
        term_norm = raw.term.strip().lower()
        if not term_norm or term_norm in deduped_terms:
            continue
        # Defensive filtering: even if the LLM ignored the policy, do not
        # re-explain something the user has already seen.
        if term_norm in known_vocab_norm or term_norm in explained_norm:
            continue
        if raw.source_segment_id not in valid_segment_ids:
            # Drop hallucinated segment ids rather than crashing.
            continue
        deduped_terms.add(term_norm)
        candidates.append(
            UnknownTermCandidate(
                candidate_id=_candidate_id(request.session_id, raw),
                session_id=request.session_id,
                term=raw.term.strip(),
                term_type=raw.term_type,
                explanation=raw.explanation.strip(),
                confidence=raw.confidence,
                privacy_level=raw.privacy_level,
                privacy_risk=raw.privacy_risk,
                source_segment_id=raw.source_segment_id,
                rationale=raw.rationale.strip(),
            )
        )
        if len(candidates) >= request.max_candidates:
            break

    usage = ModelUsageMetadata(
        provider=response.provider,
        model=response.model,
        latency_ms=response.latency_ms,
        raw_response_id=response.raw_response_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached=response.cached,
    )
    return UnknownTermDetectionResult(
        candidates=candidates,
        detection_notes=output.detection_notes,
        safety_flags=list(output.safety_flags),
        model_usage=usage,
    )


def _candidate_id(session_id: str, raw: UnknownTermModelCandidate) -> str:
    digest = sha1(
        ":".join([session_id, raw.source_segment_id, raw.term.strip().lower()]).encode("utf-8")
    ).hexdigest()[:12]
    return f"unkterm_{digest}"
