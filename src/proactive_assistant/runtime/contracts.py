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


class FeedbackInputChannel(StrEnum):
    UNKNOWN = "unknown"
    GESTURE = "gesture"
    TOUCH = "touch"
    BUTTON = "button"
    GAZE = "gaze"
    DWELL_TIME = "dwell_time"
    VOICE = "voice"
    SYSTEM = "system"
    CONVERSATION_OUTCOME = "conversation_outcome"


class FeedbackPolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FeedbackTarget(StrEnum):
    OVERALL = "overall"
    TIMING = "timing"
    CONTENT = "content"
    GRANULARITY = "granularity"
    DISPLAY = "display"
    FREQUENCY = "frequency"
    PRIVACY = "privacy"
    TASK_OUTCOME = "task_outcome"
    LATENCY = "latency"


class ProactiveDisplayStrategy(StrEnum):
    AUTO_POPUP = "auto_popup"
    SUBTLE_AVAILABLE = "subtle_available"
    MANUAL_RESPONSE = "manual_response"


class FeedbackSignalType(StrEnum):
    ACCEPT = "accept"
    NOD_ACCEPT = "nod_accept"
    DEFAULT_ACCEPT = "default_accept"
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
    HEAD_SHAKE_REJECT = "head_shake_reject"
    MANUAL_REQUEST = "manual_request"
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
    input_channel: FeedbackInputChannel = FeedbackInputChannel.UNKNOWN
    target: FeedbackTarget = FeedbackTarget.OVERALL
    polarity: FeedbackPolarity
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    display_strategy: ProactiveDisplayStrategy | None = None
    text: str = ""
    dwell_ms: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    helpfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    timing_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    content_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    granularity_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    display_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    task_progress_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    flow_break_score: float = Field(default=0.0, ge=0.0, le=1.0)
    redundancy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missed_opportunity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accept: float = Field(default=0.0, ge=0.0, le=1.0)
    helpfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    task_progress: float = Field(default=0.0, ge=-1.0, le=1.0)
    timing_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    content_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    granularity_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    display_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    annoyance: float = Field(default=0.0, ge=0.0, le=1.0)
    flow_break: float = Field(default=0.0, ge=0.0, le=1.0)
    redundancy: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    missed_opportunity: float = Field(default=0.0, ge=0.0, le=1.0)


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


class PolicyStateRef(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    session_id: str
    scenario_id: str = "meeting_business_v1"
    transcript_segment_ids: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    retrieved_memory_refs: list[str] = Field(default_factory=list)
    opportunity_id: str
    captured_text: str = ""
    activity_phase: str = ""
    prompt_category: str | None = None
    timing_action: str = ""
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyActionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    decision_id: str
    candidate_id: str
    display_status: PromptDecisionDisplayStatus
    should_prompt: bool
    timing_action: str = ""
    prompt_category: str | None = None
    content_granularity: ContentGranularity = ContentGranularity.NO_ACTION
    prd_surface: PRDSurface
    display_mode: DisplayMode
    display_strategy: ProactiveDisplayStrategy
    duration_policy: DurationPolicy
    policy_version: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime
    shown_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyNextStateRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    next_decision_id: str | None = None
    next_transcript_segment_ids: list[str] = Field(default_factory=list)


class PolicyStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    index: int = Field(ge=0)
    decision_id: str
    state: PolicyStateRef
    action: PolicyActionSnapshot
    feedback_events: list[RuntimeFeedbackEvent] = Field(default_factory=list)
    reward_observation: RewardObservation | None = None
    next_state_ref: PolicyNextStateRef = Field(default_factory=PolicyNextStateRef)
    final_reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    session_id: str
    policy_version: str | None = None
    steps: list[PolicyStep] = Field(default_factory=list)
    total_reward: float = 0.0
    feedback_event_count: int = Field(default=0, ge=0)
    rewarded_step_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_count: int = Field(default=0, ge=0)
    rewarded_step_count: int = Field(default=0, ge=0)
    feedback_event_count: int = Field(default=0, ge=0)
    total_reward: float = 0.0
    average_reward_per_step: float = 0.0
    average_reward_per_rewarded_step: float = 0.0
    accept_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    negative_feedback_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    missed_opportunity_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    interruption_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_rejection_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    granularity_mismatch_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    display_mismatch_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    rewarded_step_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class PolicyActionBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    value: str
    metrics: PolicyMetricSummary


class PolicyEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    episode_id: str
    session_id: str
    policy_version: str | None = None
    summary: PolicyMetricSummary
    by_prompt_category: list[PolicyActionBreakdown] = Field(default_factory=list)
    by_content_granularity: list[PolicyActionBreakdown] = Field(default_factory=list)
    by_timing_action: list[PolicyActionBreakdown] = Field(default_factory=list)
    by_display_strategy: list[PolicyActionBreakdown] = Field(default_factory=list)
    by_display_status: list[PolicyActionBreakdown] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyBaselineName(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class PolicyBaselineRelation(StrEnum):
    SAME = "same"
    MORE_CONSERVATIVE = "more_conservative"
    MORE_AGGRESSIVE = "more_aggressive"


class PolicyBaselineAction(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    should_prompt: bool
    content_granularity: ContentGranularity = ContentGranularity.NO_ACTION
    display_strategy: ProactiveDisplayStrategy = ProactiveDisplayStrategy.MANUAL_RESPONSE
    privacy_blocked: bool = False
    reason: str = ""


class PolicyBaselineDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    decision_id: str
    baseline_name: PolicyBaselineName
    original_should_prompt: bool
    baseline_action: PolicyBaselineAction
    matches_original: bool
    relation_to_original: PolicyBaselineRelation
    action_delta: float = Field(ge=0.0)
    missed_opportunity_covered: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyBaselineMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_count: int = Field(default=0, ge=0)
    agreement_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    more_conservative_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    more_aggressive_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    would_prompt_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    would_suppress_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_block_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    missed_opportunity_coverage_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_action_delta: float = 0.0


class PolicyBaselineEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    baseline_name: PolicyBaselineName
    episode_id: str
    session_id: str
    metrics: PolicyBaselineMetrics
    decisions: list[PolicyBaselineDecision] = Field(default_factory=list)


class PolicyBaselineComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    episode_id: str
    session_id: str
    baselines: list[PolicyBaselineEvaluation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyTrainingLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_reward: float | None = None
    has_feedback: bool = False
    accepted: bool = False
    negative_feedback: bool = False
    missed_opportunity: bool = False
    interruption: bool = False
    privacy_rejected: bool = False
    granularity_mismatch: bool = False
    display_mismatch: bool = False


class PolicyTrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    session_id: str
    episode_id: str
    step_id: str
    decision_id: str
    state: PolicyStateRef
    action: PolicyActionSnapshot
    feedback_events: list[RuntimeFeedbackEvent] = Field(default_factory=list)
    reward_observation: RewardObservation | None = None
    baseline_decisions: list[PolicyBaselineDecision] = Field(default_factory=list)
    label: PolicyTrainingLabel
    next_state_ref: PolicyNextStateRef = Field(default_factory=PolicyNextStateRef)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyTrainingExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    export_id: str
    session_id: str
    episode_id: str
    example_count: int = Field(default=0, ge=0)
    examples: list[PolicyTrainingExample] = Field(default_factory=list)
    evaluation_summary: PolicyMetricSummary
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
