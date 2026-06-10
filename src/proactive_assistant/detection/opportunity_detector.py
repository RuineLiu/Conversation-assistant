"""LLM-as-detector for prompt opportunities in meeting transcripts.

The rule-based detector in ``rules.py`` triggers on hard-coded keyword
tables (GAP_TERMS, QUESTION_TERMS, ...). Those miss any surface form not
in the table -- "ddl" never matches "deadline", "交期" never matches
"截止". This detector replaces the brittle keyword match with a single
LLM call that maps arbitrary phrasing onto structural opportunity slots:
an unanswered question, an action item missing an owner or deadline, a
decision without a conclusion, an open risk, a place a suggestion would
help, or a person/fact worth recalling.

It runs in parallel with the rule-based path (rules stay as the
zero-latency fast path and offline-test fallback) and the unknown-term
detector (which owns concept_explanation). The ``PromptOpportunityDetector``
dedupes overlap by (segment, category, timing).

Mirrors ``UnknownTermDetector``: strict Pydantic model output, offline-safe
with ``FakeModelClient``, single model call, stable opportunity ids.
"""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha1

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
    PromptCategory,
    TranscriptWindowItem,
    openai_strict_json_schema,
)


OPPORTUNITY_SYSTEM_INSTRUCTIONS = """You detect proactive prompt opportunities in meeting transcripts.

Return a JSON object only. The top-level object must contain an opportunities array.

A prompt opportunity is a moment where a proactive meeting assistant could usefully help.
Decide by MEANING, not surface keywords. Map any phrasing onto these categories:

- question_answer: someone asked an explicit or implicit question that an answer/clarification would help.
- person_or_fact: a person, role, company/project fact, date, number, or prior-context reference worth recalling.
- suggestion: a situation where a concrete next-step or option would help (planning, deciding, stuck).
- summary_gap_check: a structural gap worth flagging. Use gap_type for these:
    * unanswered_question
    * action_missing_owner    (a task/commitment with no clear responsible person)
    * action_missing_deadline (a task with no due date/time; "ddl", "交期", "最晚", "期限" all count)
    * action_missing_next_step
    * decision_missing_conclusion (a decision discussed but not concluded)
    * open_risk               (a risk/blocker raised but not resolved)

Do NOT emit concept_explanation; a separate detector handles unfamiliar terms.

Only emit an opportunity when help value is real. A well-formed action item that already has
owner AND deadline is NOT a gap -- do not emit summary_gap_check for it. Prefer a small number
of high-value opportunities over flagging everything.

For each opportunity also extract any structured fields present:
- owner: the responsible person, "" if none/unclear.
- deadline: the due date/time as stated ("下周五", "12月3日", ""), "" if none.
- entity: the short topic/task the opportunity is about.

priority: P0 (urgent/blocking) | P1 (normal) | P2 (minor/background).
confidence: 0-1 how sure you are this is a real opportunity of this category.
privacy_level / privacy_risk: follow the same scale as the rest of the system; raise them for
sensitive business data (pricing, salary, legal, customer data).
source_segment_id MUST be one of the transcript_id values supplied in the input.

Output an empty opportunities array if nothing in the window is worth a prompt."""


class OpportunityGapType(StrEnum):
    NONE = "none"
    UNANSWERED_QUESTION = "unanswered_question"
    ACTION_MISSING_OWNER = "action_missing_owner"
    ACTION_MISSING_DEADLINE = "action_missing_deadline"
    ACTION_MISSING_NEXT_STEP = "action_missing_next_step"
    DECISION_MISSING_CONCLUSION = "decision_missing_conclusion"
    OPEN_RISK = "open_risk"


class OpportunityPriority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class OpportunityCategory(StrEnum):
    QUESTION_ANSWER = "question_answer"
    PERSON_OR_FACT = "person_or_fact"
    SUGGESTION = "suggestion"
    SUMMARY_GAP_CHECK = "summary_gap_check"


class OpportunityModelCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    prompt_category: OpportunityCategory
    gap_type: OpportunityGapType = OpportunityGapType.NONE
    captured_text: str = Field(min_length=1, max_length=600)
    source_segment_id: str = Field(min_length=1, max_length=128)
    owner: str = Field(default="", max_length=120)
    deadline: str = Field(default="", max_length=120)
    entity: str = Field(default="", max_length=240)
    priority: OpportunityPriority = OpportunityPriority.P1
    confidence: float = Field(ge=0.0, le=1.0)
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=400)


class OpportunityModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opportunities: list[OpportunityModelCandidate] = Field(default_factory=list, max_length=8)
    detection_notes: str = Field(default="", max_length=400)
    safety_flags: list[str] = Field(default_factory=list, max_length=8)


class OpportunityDetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    locale: str = "zh-CN"
    transcript_window: list[TranscriptWindowItem] = Field(min_length=1)
    privacy_constraints: list[str] = Field(default_factory=list)
    max_candidates: int = Field(default=3, ge=1, le=8)


class OpportunityCandidate(BaseModel):
    """Detection result enriched with a stable opportunity id."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    candidate_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    prompt_category: OpportunityCategory
    gap_type: OpportunityGapType
    captured_text: str = Field(min_length=1)
    source_segment_id: str = Field(min_length=1)
    owner: str = ""
    deadline: str = ""
    entity: str = ""
    priority: OpportunityPriority
    confidence: float = Field(ge=0.0, le=1.0)
    privacy_level: PrivacyLevel
    privacy_risk: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class OpportunityDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[OpportunityCandidate] = Field(default_factory=list)
    detection_notes: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    model_usage: ModelUsageMetadata | None = None


class OpportunityDetector:
    """LLM-backed detector for prompt opportunities.

    Issues exactly one model call per request over the recent transcript
    window and returns ``OpportunityCandidate`` records with stable ids.
    ``PromptOpportunityDetector`` adapts these into ``PromptOpportunity``
    objects that flow through normal orchestration (the prompt text itself
    is still produced by ``PromptGenerationService``; this detector only
    decides that an opportunity exists, its category, and its structured
    fields).
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
        request: OpportunityDetectionRequest,
        *,
        model: str | None = None,
    ) -> OpportunityDetectionResult:
        resolved_model = model or self._settings.fast_model or self._settings.default_model
        model_request = ModelRequest(
            model=resolved_model,
            instructions=OPPORTUNITY_SYSTEM_INSTRUCTIONS,
            input_text=_build_detection_input(request),
            response_schema_name="OpportunityDetectionResult",
            response_schema=openai_strict_json_schema(OpportunityModelOutput),
            max_output_tokens=min(self._settings.max_output_tokens, 700),
            store=self._settings.store_model_responses,
            metadata={
                "session_id": request.session_id[:64],
                "contract": "opportunity_detection_v1",
            },
        )
        response = self._model_client.generate_structured(model_request)
        return _parse_detection_result(response, request)


# Maps the detector's compact category enum to the PRD PromptCategory.
PROMPT_CATEGORY_BY_OPPORTUNITY: dict[str, PromptCategory] = {
    OpportunityCategory.QUESTION_ANSWER.value: PromptCategory.QUESTION_ANSWER,
    OpportunityCategory.PERSON_OR_FACT.value: PromptCategory.PERSON_OR_FACT,
    OpportunityCategory.SUGGESTION.value: PromptCategory.SUGGESTION,
    OpportunityCategory.SUMMARY_GAP_CHECK.value: PromptCategory.SUMMARY_GAP_CHECK,
}


def _build_detection_input(request: OpportunityDetectionRequest) -> str:
    payload = {
        "session_id": request.session_id,
        "locale": request.locale,
        "max_candidates": request.max_candidates,
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
    request: OpportunityDetectionRequest,
) -> OpportunityDetectionResult:
    try:
        output = OpportunityModelOutput.model_validate(response.parsed)
    except ValidationError as exc:
        raise ModelOutputValidationError("model output failed OpportunityDetectionResult validation") from exc

    valid_segment_ids = {item.transcript_id for item in request.transcript_window}
    seen_keys: set[tuple[str, str, str, str]] = set()
    candidates: list[OpportunityCandidate] = []
    for raw in output.opportunities:
        if raw.source_segment_id not in valid_segment_ids:
            # Drop hallucinated segment ids rather than crashing.
            continue
        category = OpportunityCategory(raw.prompt_category)
        gap_type = OpportunityGapType(raw.gap_type)
        # Dedup by (segment, category, gap_type, entity) so the identical
        # opportunity is not emitted twice, while two distinct topics on the
        # same utterance (e.g. two separate questions) both survive.
        key = (raw.source_segment_id, category.value, gap_type.value, raw.entity.strip().lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(
            OpportunityCandidate(
                candidate_id=_candidate_id(request.session_id, raw),
                session_id=request.session_id,
                prompt_category=category,
                gap_type=gap_type,
                captured_text=raw.captured_text.strip(),
                source_segment_id=raw.source_segment_id,
                owner=raw.owner.strip(),
                deadline=raw.deadline.strip(),
                entity=raw.entity.strip(),
                priority=OpportunityPriority(raw.priority),
                confidence=raw.confidence,
                privacy_level=raw.privacy_level,
                privacy_risk=raw.privacy_risk,
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
    return OpportunityDetectionResult(
        candidates=candidates,
        detection_notes=output.detection_notes,
        safety_flags=list(output.safety_flags),
        model_usage=usage,
    )


def _candidate_id(session_id: str, raw: OpportunityModelCandidate) -> str:
    digest = sha1(
        ":".join(
            [
                session_id,
                raw.source_segment_id,
                str(raw.prompt_category),
                str(raw.gap_type),
                raw.entity.strip().lower(),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"llmopp_{digest}"
