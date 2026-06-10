from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.asr import SpeechTranscriptionResult
from proactive_assistant.meeting_state import MeetingGap, MeetingState
from proactive_assistant.memory import MemoryContext, MemoryRecord, MemoryUpsertResult
from proactive_assistant.prompting import (
    ContentGranularity,
    DisplayMode,
    DurationPolicy,
    ModelUsageMetadata,
    PRDSurface,
    PrivacyLevel,
)
from proactive_assistant.runtime import (
    MemoryCandidate,
    PolicyBaselineComparisonReport,
    PolicyEpisode,
    PolicyEvaluationReport,
    PolicyTrainingExport,
    PromptDecisionDisplayStatus,
    PromptDecisionRecord,
    RewardObservation,
    RuntimeFeedbackEvent,
)
from proactive_assistant.sessions import AssistantSession, SessionContextSnapshot, TranscriptSegmentRecord


class ProductPromptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    decision_id: str
    session_id: str
    candidate_id: str
    opportunity_id: str
    should_display: bool
    display_status: PromptDecisionDisplayStatus
    prompt_category: str | None = None
    content_granularity: ContentGranularity = ContentGranularity.NO_ACTION
    prd_surface: PRDSurface
    display_mode: DisplayMode
    duration_policy: DurationPolicy
    glasses_title: str = ""
    glasses_text: str = ""
    app_detail_text: str = ""
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    safety_flags: list[str] = Field(default_factory=list)


class ProductTranscriptStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    transcript_segment: TranscriptSegmentRecord
    snapshot: SessionContextSnapshot
    meeting_state: MeetingState | None = None
    meeting_gaps: list[MeetingGap] = Field(default_factory=list)
    retrieved_memory_context: MemoryContext | None = None
    prompts: list[ProductPromptPayload] = Field(default_factory=list)
    decisions: list[PromptDecisionRecord] = Field(default_factory=list)
    opportunity_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)


class ProductAudioTranscriptStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcription: SpeechTranscriptionResult
    transcript_step: ProductTranscriptStepResult | None = None


class ProductSessionStateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    meeting_state: MeetingState
    transcript: list[TranscriptSegmentRecord] = Field(default_factory=list)
    prompts: list[ProductPromptPayload] = Field(default_factory=list)
    memory_context: MemoryContext | None = None


class ProductSessionSummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    meeting_state: MeetingState
    meeting_gaps: list[MeetingGap] = Field(default_factory=list)
    retrieved_memory_context: MemoryContext | None = None
    prompts: list[ProductPromptPayload] = Field(default_factory=list)
    decisions: list[PromptDecisionRecord] = Field(default_factory=list)


class ProductSessionLifecycleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    meeting_state: MeetingState
    summary: ProductSessionSummaryResult | None = None


class ProductFeedbackResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: PromptDecisionRecord
    feedback_event: RuntimeFeedbackEvent
    reward_observation: RewardObservation | None = None
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memories: list[MemoryRecord] = Field(default_factory=list)


class ProductPolicyEpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    episode: PolicyEpisode


class ProductPolicyEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    report: PolicyEvaluationReport


class ProductPolicyBaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    report: PolicyBaselineComparisonReport


class ProductPolicyTrainingExportResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    export: PolicyTrainingExport


class ProductMemorySnapshotResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    meeting_state: MeetingState
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memories: list[MemoryRecord] = Field(default_factory=list)
    memory_upserts: list[MemoryUpsertResult] = Field(default_factory=list)
    committed: bool = True


class ProductInlineMemoryCaptureResult(BaseModel):
    """Result of an explicit user-driven "remember this" capture.

    Returned by the inline capture endpoint and the ``capture_inline_memory``
    service method. ``committed`` is False only when the writer rejected the
    candidate (e.g. blocked by privacy policy).
    """

    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    memory: MemoryRecord
    upsert: MemoryUpsertResult
    committed: bool = True


class ProductPrivacyMetrics(BaseModel):
    """Per-session privacy enforcement counters.

    Surfaces what the privacy + rate limiter + memory layers actually
    blocked or routed away from the glasses, so the demo can show the
    system is making conservative decisions.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    total_decisions: int = Field(default=0, ge=0)
    glasses_decisions: int = Field(default=0, ge=0)
    high_privacy_opportunities_detected: int = Field(default=0, ge=0)
    deferred_to_app_by_rate_limiter: int = Field(default=0, ge=0)
    suppressed_by_safety_flag: int = Field(default=0, ge=0)
    deferred_due_to_privacy: int = Field(default=0, ge=0)
    memory_writes_blocked_by_privacy: int = Field(default=0, ge=0)
    enforcer_text_truncated: int = Field(default=0, ge=0)
    by_safety_flag: dict[str, int] = Field(default_factory=dict)
    by_rate_limit_reason: dict[str, int] = Field(default_factory=dict)


class ProductMemoryExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memories: list[MemoryRecord] = Field(default_factory=list)
    memory_upserts: list[MemoryUpsertResult] = Field(default_factory=list)
    committed: bool = True
    extraction_notes: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    model_usage: ModelUsageMetadata | None = None
