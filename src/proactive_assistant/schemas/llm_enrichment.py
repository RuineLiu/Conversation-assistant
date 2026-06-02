from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


LLM_ENRICHMENT_SCHEMA_VERSION = "llm_enrichment_contract_v0"


class LLMEnrichmentField(StrEnum):
    PRE_ACTIVITY_TOLERANCE = "pre_activity_tolerance"
    IN_ACTIVITY_TOLERANCE = "in_activity_tolerance"
    POST_ACTIVITY_TOLERANCE = "post_activity_tolerance"
    DETAIL_PREFERENCE = "detail_preference"
    PERMISSION_FIRST = "permission_first"
    PRIVACY_SENSITIVITY = "privacy_sensitivity"
    MEETING_INTERRUPT_COST = "meeting_interrupt_cost"
    SOLO_TASK_INTERRUPT_COST = "solo_task_interrupt_cost"
    SOCIAL_INTERRUPT_COST = "social_interrupt_cost"
    DEADLINE_HELP_VALUE = "deadline_help_value"
    EXPLICIT_FEEDBACK_RATE = "explicit_feedback_rate"
    NEGATIVE_FEEDBACK_THRESHOLD = "negative_feedback_threshold"
    POLITENESS_BIAS = "politeness_bias"


class LLMEnrichmentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: LLMEnrichmentField
    value: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1, max_length=5)
    rationale: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def evidence_items_must_be_non_empty(self) -> "LLMEnrichmentProposal":
        if any(not item.strip() for item in self.evidence):
            raise ValueError("evidence items must be non-empty strings")
        return self


class LLMEnrichmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    schema_version: str = LLM_ENRICHMENT_SCHEMA_VERSION
    proposals: list[LLMEnrichmentProposal] = Field(default_factory=list, max_length=13)
    model_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def proposal_fields_must_be_unique(self) -> "LLMEnrichmentResponse":
        field_names = [proposal.field_name for proposal in self.proposals]
        if len(field_names) != len(set(field_names)):
            raise ValueError("duplicate proposal field_name values are not allowed")
        return self
