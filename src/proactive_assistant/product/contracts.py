from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.meeting_state import MeetingGap, MeetingState
from proactive_assistant.memory import MemoryContext, MemoryRecord
from proactive_assistant.prompting import ContentGranularity, DisplayMode, DurationPolicy, PRDSurface, PrivacyLevel
from proactive_assistant.runtime import (
    MemoryCandidate,
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


class ProductFeedbackResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: PromptDecisionRecord
    feedback_event: RuntimeFeedbackEvent
    reward_observation: RewardObservation | None = None
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memories: list[MemoryRecord] = Field(default_factory=list)


class ProductMemorySnapshotResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    meeting_state: MeetingState
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memories: list[MemoryRecord] = Field(default_factory=list)
    committed: bool = True
