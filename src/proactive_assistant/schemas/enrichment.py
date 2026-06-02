from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.schemas.persona import FieldSource, Persona


class EnrichmentEvidenceType(StrEnum):
    PERSONA_TEXT = "persona_text"
    PREFERENCE_SAMPLE = "preference_sample"
    KEYWORD_MATCH = "keyword_match"
    MANUAL_NOTE = "manual_note"
    LLM_PROPOSAL = "llm_proposal"


class EnrichmentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: str
    evidence_type: EnrichmentEvidenceType
    text: str = Field(min_length=1)
    matched_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PersonaFieldEnrichment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: str
    value: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: FieldSource
    evidence: list[EnrichmentEvidence] = Field(min_length=1)


class PersonaEnrichmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    enricher_version: str
    updates: list[PersonaFieldEnrichment] = Field(default_factory=list)
    skipped_fields: list[str] = Field(default_factory=list)
    enriched_persona: Persona
