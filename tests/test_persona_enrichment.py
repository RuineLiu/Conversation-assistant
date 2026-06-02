from pathlib import Path

from proactive_assistant.adapters import PersonaDatasetAdapter
from proactive_assistant.enrichment import RuleBasedPersonaEnricher
from proactive_assistant.schemas.persona import FieldSource


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "persona"
    / "sample_persona_records.jsonl"
)


def test_rule_based_enricher_updates_low_confidence_pending_fields() -> None:
    beta = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[1]

    result = RuleBasedPersonaEnricher().enrich(beta)
    enriched = result.enriched_persona

    assert enriched.proactive_preferences.in_activity_tolerance.source == FieldSource.RULE_BASED
    assert enriched.proactive_preferences.in_activity_tolerance.value == 0.18
    assert "in_activity_tolerance" not in enriched.source_metadata["pending_enrichment_fields"]
    assert result.updates


def test_rule_based_enricher_does_not_overwrite_high_confidence_source_fields() -> None:
    alpha = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[0]

    result = RuleBasedPersonaEnricher().enrich(alpha)
    enriched = result.enriched_persona

    assert enriched.proactive_preferences.pre_activity_tolerance.value == 0.86
    assert enriched.proactive_preferences.pre_activity_tolerance.source == FieldSource.SOURCE_DERIVED
    assert "pre_activity_tolerance" in result.skipped_fields


def test_rule_based_enricher_records_evidence_for_each_update() -> None:
    beta = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[1]

    result = RuleBasedPersonaEnricher().enrich(beta)
    update = next(
        update
        for update in result.updates
        if update.field_name == "in_activity_tolerance"
    )

    assert update.evidence[0].matched_terms
    assert "studying" in update.evidence[0].text


def test_rule_based_enricher_records_enrichment_run_metadata() -> None:
    beta = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[1]

    enriched = RuleBasedPersonaEnricher().enrich(beta).enriched_persona
    enrichment_runs = enriched.source_metadata["enrichment_runs"]

    assert enrichment_runs[-1]["enricher_version"] == "rule_based_persona_enricher_v0"
    assert "in_activity_tolerance" in enrichment_runs[-1]["updated_fields"]
