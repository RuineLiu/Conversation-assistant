from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.prompting import ContentGranularity, PrivacyLevel, PromptCategory
from proactive_assistant.schemas.scenario import ActivityPhase


class PromptPriority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class CandidateTimingAction(StrEnum):
    BEFORE_ACTIVITY = "before_activity"
    DURING_ACTIVITY = "during_activity"
    AFTER_ACTIVITY = "after_activity"
    MANUAL = "manual"
    NO_ACTION = "no_action"


class DetectionRuleMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    rule_name: str
    matched_terms: list[str] = Field(default_factory=list)
    confidence_delta: float = 0.0
    reason: str = ""


class PromptOpportunity(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    opportunity_id: str
    session_id: str
    trigger_segment_ids: list[str] = Field(min_length=1)
    captured_text: str = Field(min_length=1)
    prompt_category: PromptCategory
    activity_phase: ActivityPhase
    candidate_timing_action: CandidateTimingAction
    suggested_content_granularity: ContentGranularity
    priority: PromptPriority
    confidence: float = Field(ge=0.0, le=1.0)
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    privacy_risk: float = Field(ge=0.0, le=1.0)
    reason: str
    safety_flags: list[str] = Field(default_factory=list)
    rule_matches: list[DetectionRuleMatch] = Field(default_factory=list)
    # Speaker whose utterance triggered this opportunity. Empty string when
    # the trigger does not map to a single speaker (e.g. structural gap
    # spanning multiple utterances). The product/runtime layer uses this
    # for audit ("responding to 张三's mention") and to attach speaker
    # identity to downstream memory writes.
    target_speaker_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def prompt_opportunity_must_have_actionable_granularity(self) -> "PromptOpportunity":
        if self.suggested_content_granularity == ContentGranularity.NO_ACTION:
            raise ValueError("prompt opportunities must use content granularity greater than 0")
        return self


class PromptOpportunityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    opportunities: list[PromptOpportunity] = Field(default_factory=list)
    inspected_segment_ids: list[str] = Field(default_factory=list)
