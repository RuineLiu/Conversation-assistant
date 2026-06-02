from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from proactive_assistant.schemas.enrichment import (
    EnrichmentEvidence,
    EnrichmentEvidenceType,
    PersonaEnrichmentResult,
    PersonaFieldEnrichment,
)
from proactive_assistant.schemas.persona import FieldSource, Persona, PreferenceScore


RULE_BASED_ENRICHER_VERSION = "rule_based_persona_enricher_v0"
HIGH_CONFIDENCE_SOURCE_THRESHOLD = 0.7

FIELD_TARGETS = {
    "pre_activity_tolerance": ("proactive_preferences", "pre_activity_tolerance"),
    "in_activity_tolerance": ("proactive_preferences", "in_activity_tolerance"),
    "post_activity_tolerance": ("proactive_preferences", "post_activity_tolerance"),
    "detail_preference": ("proactive_preferences", "detail_preference"),
    "permission_first": ("proactive_preferences", "permission_first"),
    "privacy_sensitivity": ("proactive_preferences", "privacy_sensitivity"),
}


@dataclass(frozen=True)
class RuleDefinition:
    field_name: str
    value: float
    confidence: float
    terms: tuple[str, ...]


RULES = (
    RuleDefinition(
        field_name="pre_activity_tolerance",
        value=0.82,
        confidence=0.62,
        terms=(
            "likes preparation",
            "preparation",
            "prepare",
            "planning before",
            "plan before",
            "checklist",
            "organized",
            "clear deadlines",
        ),
    ),
    RuleDefinition(
        field_name="in_activity_tolerance",
        value=0.18,
        confidence=0.68,
        terms=(
            "dislikes interruption",
            "dislike interruption",
            "uninterrupted",
            "deep focus",
            "while studying",
            "focused work",
            "concentrating",
        ),
    ),
    RuleDefinition(
        field_name="post_activity_tolerance",
        value=0.81,
        confidence=0.64,
        terms=(
            "recap",
            "summary",
            "summarize",
            "after finishing",
            "after task",
            "after the task",
            "review after",
        ),
    ),
    RuleDefinition(
        field_name="detail_preference",
        value=0.26,
        confidence=0.58,
        terms=(
            "concise",
            "short checklist",
            "short reminder",
            "brief",
            "minimal",
        ),
    ),
    RuleDefinition(
        field_name="detail_preference",
        value=0.78,
        confidence=0.58,
        terms=(
            "detailed",
            "step by step",
            "step-by-step",
            "thorough",
            "comprehensive",
        ),
    ),
    RuleDefinition(
        field_name="permission_first",
        value=0.84,
        confidence=0.65,
        terms=(
            "privacy-sensitive",
            "privacy sensitive",
            "ask first",
            "permission",
            "consent",
        ),
    ),
    RuleDefinition(
        field_name="privacy_sensitivity",
        value=0.9,
        confidence=0.7,
        terms=(
            "privacy-sensitive",
            "privacy sensitive",
            "private",
            "privacy",
            "confidential",
        ),
    ),
)


class RuleBasedPersonaEnricher:
    """Deterministic enrichment baseline for PERSONA-derived personas."""

    def enrich(self, persona: Persona) -> PersonaEnrichmentResult:
        documents = self._evidence_documents(persona)
        candidates = self._collect_candidates(documents)
        updates: list[PersonaFieldEnrichment] = []
        skipped_fields: list[str] = []

        for field_name, field_candidates in sorted(candidates.items()):
            chosen = max(field_candidates, key=lambda candidate: candidate.confidence)
            if self._should_skip(persona, chosen.field_name):
                skipped_fields.append(chosen.field_name)
                continue
            updates.append(chosen)

        enriched_persona = self._apply_updates(persona, updates, skipped_fields)
        return PersonaEnrichmentResult(
            persona_id=persona.internal_persona_id,
            enricher_version=RULE_BASED_ENRICHER_VERSION,
            updates=updates,
            skipped_fields=sorted(set(skipped_fields)),
            enriched_persona=enriched_persona,
        )

    def _evidence_documents(self, persona: Persona) -> list[tuple[EnrichmentEvidenceType, str]]:
        metadata = persona.source_metadata or {}
        documents: list[tuple[EnrichmentEvidenceType, str]] = []

        persona_text = metadata.get("persona_text")
        if isinstance(persona_text, str) and persona_text.strip():
            documents.append((EnrichmentEvidenceType.PERSONA_TEXT, persona_text.strip()))

        preference_samples = metadata.get("preference_samples", [])
        if isinstance(preference_samples, list):
            for sample in preference_samples:
                if isinstance(sample, str) and sample.strip():
                    documents.append((EnrichmentEvidenceType.PREFERENCE_SAMPLE, sample.strip()))

        return documents

    def _collect_candidates(
        self,
        documents: list[tuple[EnrichmentEvidenceType, str]],
    ) -> dict[str, list[PersonaFieldEnrichment]]:
        matched_by_field: dict[str, list[PersonaFieldEnrichment]] = defaultdict(list)
        for rule in RULES:
            evidence = self._match_rule(rule, documents)
            if not evidence:
                continue
            matched_by_field[rule.field_name].append(
                PersonaFieldEnrichment(
                    field_name=rule.field_name,
                    value=rule.value,
                    confidence=rule.confidence,
                    source=FieldSource.RULE_BASED,
                    evidence=evidence,
                )
            )
        return dict(matched_by_field)

    def _match_rule(
        self,
        rule: RuleDefinition,
        documents: list[tuple[EnrichmentEvidenceType, str]],
    ) -> list[EnrichmentEvidence]:
        evidence: list[EnrichmentEvidence] = []
        for evidence_type, text in documents:
            lowered = text.lower()
            matched_terms = sorted({term for term in rule.terms if term in lowered})
            if not matched_terms:
                continue
            evidence.append(
                EnrichmentEvidence(
                    field_name=rule.field_name,
                    evidence_type=evidence_type,
                    text=text,
                    matched_terms=matched_terms,
                    confidence=rule.confidence,
                )
            )
        return evidence

    def _should_skip(self, persona: Persona, field_name: str) -> bool:
        existing = self._existing_score(persona, field_name)
        if existing is None:
            return False
        if existing.source == FieldSource.MANUAL:
            return True
        return (
            existing.source == FieldSource.SOURCE_DERIVED
            and existing.confidence >= HIGH_CONFIDENCE_SOURCE_THRESHOLD
        )

    def _existing_score(self, persona: Persona, field_name: str) -> PreferenceScore | None:
        target = FIELD_TARGETS.get(field_name)
        if target is None:
            return None
        group_name, attribute_name = target
        group = getattr(persona, group_name)
        return getattr(group, attribute_name)

    def _apply_updates(
        self,
        persona: Persona,
        updates: list[PersonaFieldEnrichment],
        skipped_fields: list[str],
    ) -> Persona:
        payload = persona.model_dump(mode="python")
        metadata = dict(payload.get("source_metadata") or {})
        pending_fields = set(metadata.get("pending_enrichment_fields", []))
        updated_fields = {update.field_name for update in updates}

        for update in updates:
            group_name, attribute_name = FIELD_TARGETS[update.field_name]
            payload[group_name][attribute_name] = PreferenceScore(
                value=update.value,
                source=update.source,
                confidence=update.confidence,
            ).model_dump(mode="python")

        metadata["pending_enrichment_fields"] = sorted(pending_fields - updated_fields)
        enrichment_runs = list(metadata.get("enrichment_runs", []))
        enrichment_runs.append(
            {
                "enricher_version": RULE_BASED_ENRICHER_VERSION,
                "updated_fields": sorted(updated_fields),
                "skipped_fields": sorted(set(skipped_fields)),
            }
        )
        metadata["enrichment_runs"] = enrichment_runs
        payload["source_metadata"] = metadata
        return Persona.model_validate(payload)
