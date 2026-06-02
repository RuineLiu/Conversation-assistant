from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FieldSource(StrEnum):
    SOURCE_DERIVED = "source_derived"
    LLM_INFERRED = "llm_inferred"
    SIMULATION_LEARNED = "simulation_learned"
    MANUAL = "manual"
    RULE_BASED = "rule_based"


class PreferenceScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)
    source: FieldSource
    confidence: float = Field(ge=0.0, le=1.0)


class DemographicStub(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_band: str | None = None
    occupation_type: str | None = None
    daily_routine_density: str | None = None


class BigFive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)


class ProactivePreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pre_activity_tolerance: PreferenceScore
    in_activity_tolerance: PreferenceScore
    post_activity_tolerance: PreferenceScore
    detail_preference: PreferenceScore
    permission_first: PreferenceScore
    privacy_sensitivity: PreferenceScore
    notification_budget_per_hour: int = Field(ge=0, le=60)


class ContextSensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meeting_interrupt_cost: PreferenceScore
    solo_task_interrupt_cost: PreferenceScore
    social_interrupt_cost: PreferenceScore
    deadline_help_value: PreferenceScore


class FeedbackStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explicit_feedback_rate: PreferenceScore
    negative_feedback_threshold: PreferenceScore
    politeness_bias: PreferenceScore


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "SynthLabsAI/PERSONA"
    source_persona_id: str
    internal_persona_id: str
    source_license: str = "cc-by-nc-sa-4.0"
    demographic_stub: DemographicStub
    big_five: BigFive
    proactive_preferences: ProactivePreferences
    context_sensitivity: ContextSensitivity
    feedback_style: FeedbackStyle
    source_metadata: dict[str, Any] = Field(default_factory=dict)
