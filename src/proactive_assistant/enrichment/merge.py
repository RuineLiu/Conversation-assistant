from __future__ import annotations

from dataclasses import dataclass

from proactive_assistant.schemas.enrichment import (
    EnrichmentEvidence,
    EnrichmentEvidenceType,
    PersonaEnrichmentResult,
    PersonaFieldEnrichment,
)
from proactive_assistant.schemas.llm_enrichment import (
    LLMEnrichmentField,
    LLMEnrichmentProposal,
    LLMEnrichmentResponse,
)
from proactive_assistant.schemas.persona import FieldSource, Persona, PreferenceScore


LLM_ENRICHMENT_MERGE_VERSION = "llm_enrichment_merge_v0"
MIN_LLM_PROPOSAL_CONFIDENCE = 0.55
HIGH_CONFIDENCE_SOURCE_THRESHOLD = 0.7

FIELD_TARGETS = {
    LLMEnrichmentField.PRE_ACTIVITY_TOLERANCE: (
        "proactive_preferences",
        "pre_activity_tolerance",
    ),
    LLMEnrichmentField.IN_ACTIVITY_TOLERANCE: (
        "proactive_preferences",
        "in_activity_tolerance",
    ),
    LLMEnrichmentField.POST_ACTIVITY_TOLERANCE: (
        "proactive_preferences",
        "post_activity_tolerance",
    ),
    LLMEnrichmentField.DETAIL_PREFERENCE: ("proactive_preferences", "detail_preference"),
    LLMEnrichmentField.PERMISSION_FIRST: ("proactive_preferences", "permission_first"),
    LLMEnrichmentField.PRIVACY_SENSITIVITY: (
        "proactive_preferences",
        "privacy_sensitivity",
    ),
    LLMEnrichmentField.MEETING_INTERRUPT_COST: (
        "context_sensitivity",
        "meeting_interrupt_cost",
    ),
    LLMEnrichmentField.SOLO_TASK_INTERRUPT_COST: (
        "context_sensitivity",
        "solo_task_interrupt_cost",
    ),
    LLMEnrichmentField.SOCIAL_INTERRUPT_COST: (
        "context_sensitivity",
        "social_interrupt_cost",
    ),
    LLMEnrichmentField.DEADLINE_HELP_VALUE: (
        "context_sensitivity",
        "deadline_help_value",
    ),
    LLMEnrichmentField.EXPLICIT_FEEDBACK_RATE: (
        "feedback_style",
        "explicit_feedback_rate",
    ),
    LLMEnrichmentField.NEGATIVE_FEEDBACK_THRESHOLD: (
        "feedback_style",
        "negative_feedback_threshold",
    ),
    LLMEnrichmentField.POLITENESS_BIAS: ("feedback_style", "politeness_bias"),
}


@dataclass(frozen=True)
class MergeSkip:
    field_name: str
    reason: str


def merge_llm_enrichment(
    persona: Persona,
    response: LLMEnrichmentResponse,
) -> PersonaEnrichmentResult:
    """Merge validated LLM proposals into a Persona without trusting them blindly."""

    if response.persona_id not in {
        persona.internal_persona_id,
        persona.source_persona_id,
    }:
        raise ValueError("LLM enrichment response persona_id does not match persona")

    updates: list[PersonaFieldEnrichment] = []
    skipped: list[MergeSkip] = []
    for proposal in response.proposals:
        skip_reason = _skip_reason(persona, proposal)
        if skip_reason is not None:
            skipped.append(MergeSkip(field_name=proposal.field_name.value, reason=skip_reason))
            continue
        updates.append(_proposal_to_update(proposal))

    enriched_persona = _apply_updates(persona, updates, skipped, response)
    return PersonaEnrichmentResult(
        persona_id=persona.internal_persona_id,
        enricher_version=LLM_ENRICHMENT_MERGE_VERSION,
        updates=updates,
        skipped_fields=sorted({item.field_name for item in skipped}),
        enriched_persona=enriched_persona,
    )


def _skip_reason(persona: Persona, proposal: LLMEnrichmentProposal) -> str | None:
    if proposal.confidence < MIN_LLM_PROPOSAL_CONFIDENCE:
        return "below_min_confidence"

    existing = _existing_score(persona, proposal.field_name)
    if existing.source == FieldSource.MANUAL:
        return "manual_field"
    if (
        existing.source == FieldSource.SOURCE_DERIVED
        and existing.confidence >= HIGH_CONFIDENCE_SOURCE_THRESHOLD
    ):
        return "high_confidence_source_derived"
    if (
        existing.source == FieldSource.RULE_BASED
        and existing.confidence >= proposal.confidence
    ):
        return "rule_based_confidence_not_improved"
    if (
        existing.source == FieldSource.LLM_INFERRED
        and existing.confidence >= proposal.confidence
        and existing.confidence > 0.0
    ):
        return "llm_confidence_not_improved"
    return None


def _proposal_to_update(proposal: LLMEnrichmentProposal) -> PersonaFieldEnrichment:
    return PersonaFieldEnrichment(
        field_name=proposal.field_name.value,
        value=proposal.value,
        confidence=proposal.confidence,
        source=FieldSource.LLM_INFERRED,
        evidence=[
            EnrichmentEvidence(
                field_name=proposal.field_name.value,
                evidence_type=EnrichmentEvidenceType.LLM_PROPOSAL,
                text=evidence,
                matched_terms=[],
                confidence=proposal.confidence,
            )
            for evidence in proposal.evidence
        ],
    )


def _existing_score(persona: Persona, field_name: LLMEnrichmentField) -> PreferenceScore:
    group_name, attribute_name = FIELD_TARGETS[field_name]
    group = getattr(persona, group_name)
    return getattr(group, attribute_name)


def _apply_updates(
    persona: Persona,
    updates: list[PersonaFieldEnrichment],
    skipped: list[MergeSkip],
    response: LLMEnrichmentResponse,
) -> Persona:
    payload = persona.model_dump(mode="python")
    metadata = dict(payload.get("source_metadata") or {})
    pending_fields = set(metadata.get("pending_enrichment_fields", []))
    updated_fields = {update.field_name for update in updates}

    for update in updates:
        field = LLMEnrichmentField(update.field_name)
        group_name, attribute_name = FIELD_TARGETS[field]
        payload[group_name][attribute_name] = PreferenceScore(
            value=update.value,
            source=update.source,
            confidence=update.confidence,
        ).model_dump(mode="python")

    metadata["pending_enrichment_fields"] = sorted(pending_fields - updated_fields)
    llm_runs = list(metadata.get("llm_enrichment_runs", []))
    llm_runs.append(
        {
            "merge_version": LLM_ENRICHMENT_MERGE_VERSION,
            "schema_version": response.schema_version,
            "updated_fields": sorted(updated_fields),
            "skipped_proposals": [
                {"field_name": item.field_name, "reason": item.reason}
                for item in skipped
            ],
            "model_metadata": dict(response.model_metadata),
        }
    )
    metadata["llm_enrichment_runs"] = llm_runs

    enrichment_runs = list(metadata.get("enrichment_runs", []))
    enrichment_runs.append(
        {
            "enricher_version": LLM_ENRICHMENT_MERGE_VERSION,
            "updated_fields": sorted(updated_fields),
            "skipped_fields": sorted({item.field_name for item in skipped}),
        }
    )
    metadata["enrichment_runs"] = enrichment_runs

    payload["source_metadata"] = metadata
    return Persona.model_validate(payload)
