from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.orchestration import PromptCandidate, PromptCandidateStatus
from proactive_assistant.prompting import ContentGranularity, DisplayMode, DurationPolicy, PRDSurface, PrivacyLevel


class PromptDecisionDisplayStatus(StrEnum):
    SHOWN = "shown"
    SUPPRESSED = "suppressed"
    FAILED = "failed"


class FeedbackSignalSource(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    TASK_OUTCOME = "task_outcome"
    SYSTEM = "system"


class FeedbackPolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FeedbackSignalType(StrEnum):
    ACCEPT = "accept"
    DISMISS = "dismiss"
    IGNORE = "ignore"
    SNOOZE = "snooze"
    ASK_FOLLOWUP = "ask_followup"
    OPEN_DETAIL = "open_detail"
    SECOND_LOOK = "second_look"
    DWELL = "dwell"
    SAVE = "save"
    MARK_HELPFUL = "mark_helpful"
    MARK_NOT_HELPFUL = "mark_not_helpful"
    VERBAL_POSITIVE = "verbal_positive"
    VERBAL_REJECT = "verbal_reject"
    PRIVACY_REJECT = "privacy_reject"
    DISABLE_AUTO_PROMPT = "disable_auto_prompt"
    SWITCH_TO_MANUAL = "switch_to_manual"
    COMPLAIN_INTERRUPTIVE = "complain_interruptive"
    FLOW_BREAK = "flow_break"
    REDUNDANT_TRIGGER = "redundant_trigger"
    TASK_PROGRESS = "task_progress"
    TASK_REGRESSION = "task_regression"
    LATENCY_OBSERVED = "latency_observed"


class MemoryCandidateType(StrEnum):
    USER_PREFERENCE = "user_preference"
    NEGATIVE_PREFERENCE = "negative_preference"
    PRIVACY_PREFERENCE = "privacy_preference"
    MEETING_FACT = "meeting_fact"
    ACTION_ITEM = "action_item"
    DECISION = "decision"
    PERSON_OR_FACT = "person_or_fact"
    PROJECT_CONTEXT = "project_context"
    SUMMARY = "summary"


class MemoryWritePolicy(StrEnum):
    ELIGIBLE = "eligible"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"


class PromptDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    decision_id: str
    session_id: str
    candidate_id: str
    opportunity_id: str
    candidate: PromptCandidate
    display_status: PromptDecisionDisplayStatus
    prompt_category: str | None = None
    content_granularity: ContentGranularity = ContentGranularity.NO_ACTION
    prd_surface: PRDSurface
    display_mode: DisplayMode
    duration_policy: DurationPolicy
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    policy_version: str = "prompt_runtime_v0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    shown_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def shown_records_require_prompt_result(self) -> "PromptDecisionRecord":
        if _enum_value(self.display_status) != PromptDecisionDisplayStatus.SHOWN.value:
            return self
        if self.candidate.prompt_result is None or not self.candidate.prompt_result.should_prompt:
            raise ValueError("shown decision records require a positive prompt_result")
        return self


class RuntimeFeedbackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    event_id: str
    decision_id: str
    session_id: str
    signal_type: FeedbackSignalType
    source: FeedbackSignalSource
    polarity: FeedbackPolarity
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    text: str = ""
    dwell_ms: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    helpfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    timing_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    content_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    task_progress_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    flow_break_score: float = Field(default=0.0, ge=0.0, le=1.0)
    redundancy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accept: float = Field(default=0.0, ge=0.0, le=1.0)
    helpfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    task_progress: float = Field(default=0.0, ge=-1.0, le=1.0)
    timing_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    content_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    annoyance: float = Field(default=0.0, ge=0.0, le=1.0)
    flow_break: float = Field(default=0.0, ge=0.0, le=1.0)
    redundancy: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_penalty: float = Field(default=0.0, ge=0.0, le=1.0)


class RewardObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    decision_id: str
    session_id: str
    event_ids: list[str] = Field(default_factory=list)
    components: RewardComponents
    explicit_score: float
    implicit_score: float
    task_outcome_score: float
    interruption_cost: float
    privacy_penalty: float
    latency_penalty: float
    final_reward: float
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    memory_candidate_id: str
    decision_id: str
    session_id: str
    source_event_ids: list[str] = Field(default_factory=list)
    candidate_type: MemoryCandidateType
    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    write_policy: MemoryWritePolicy
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def display_status_for_candidate(candidate: PromptCandidate) -> PromptDecisionDisplayStatus:
    status = _enum_value(candidate.status)
    if status == PromptCandidateStatus.GENERATED.value:
        return PromptDecisionDisplayStatus.SHOWN
    if status == PromptCandidateStatus.SUPPRESSED.value:
        return PromptDecisionDisplayStatus.SUPPRESSED
    return PromptDecisionDisplayStatus.FAILED


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value
