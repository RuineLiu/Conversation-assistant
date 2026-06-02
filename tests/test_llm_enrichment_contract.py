from pathlib import Path

import pytest
from pydantic import ValidationError

from proactive_assistant.adapters import PersonaDatasetAdapter
from proactive_assistant.enrichment import RuleBasedPersonaEnricher
from proactive_assistant.enrichment.llm_contract import parse_llm_enrichment_response
from proactive_assistant.enrichment.merge import merge_llm_enrichment
from proactive_assistant.schemas.llm_enrichment import (
    LLMEnrichmentField,
    LLMEnrichmentResponse,
)
from proactive_assistant.schemas.persona import FieldSource


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "persona"
    / "sample_persona_records.jsonl"
)


def valid_payload(persona_id: str) -> dict:
    return {
        "persona_id": persona_id,
        "schema_version": "llm_enrichment_contract_v0",
        "proposals": [
            {
                "field_name": "in_activity_tolerance",
                "value": 0.24,
                "confidence": 0.72,
                "evidence": ["dislikes interruption while studying"],
                "rationale": "The persona prefers uninterrupted focus during study.",
            }
        ],
        "model_metadata": {"model": "fixture-model"},
    }


def test_parse_llm_enrichment_response_accepts_valid_payload() -> None:
    response = parse_llm_enrichment_response(valid_payload("P_persona_beta"))

    assert response.proposals[0].field_name == LLMEnrichmentField.IN_ACTIVITY_TOLERANCE
    assert response.schema_version == "llm_enrichment_contract_v0"


def test_parse_llm_enrichment_response_rejects_unknown_fields() -> None:
    payload = valid_payload("P_persona_beta")
    payload["proposals"][0]["field_name"] = "made_up_field"

    with pytest.raises(ValidationError):
        parse_llm_enrichment_response(payload)


def test_parse_llm_enrichment_response_rejects_empty_evidence() -> None:
    payload = valid_payload("P_persona_beta")
    payload["proposals"][0]["evidence"] = [""]

    with pytest.raises(ValidationError):
        parse_llm_enrichment_response(payload)


def test_parse_llm_enrichment_response_rejects_duplicate_fields() -> None:
    payload = valid_payload("P_persona_beta")
    payload["proposals"].append(dict(payload["proposals"][0]))

    with pytest.raises(ValidationError):
        parse_llm_enrichment_response(payload)


def test_merge_llm_enrichment_updates_low_confidence_placeholder() -> None:
    beta = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[1]
    response = LLMEnrichmentResponse.model_validate(valid_payload(beta.internal_persona_id))

    result = merge_llm_enrichment(beta, response)
    enriched = result.enriched_persona

    assert enriched.proactive_preferences.in_activity_tolerance.source == FieldSource.LLM_INFERRED
    assert enriched.proactive_preferences.in_activity_tolerance.value == 0.24
    assert "in_activity_tolerance" not in enriched.source_metadata["pending_enrichment_fields"]
    assert enriched.source_metadata["llm_enrichment_runs"][-1]["updated_fields"] == [
        "in_activity_tolerance"
    ]


def test_merge_llm_enrichment_does_not_overwrite_high_confidence_source_field() -> None:
    alpha = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[0]
    payload = valid_payload(alpha.internal_persona_id)
    payload["proposals"][0]["field_name"] = "pre_activity_tolerance"
    payload["proposals"][0]["value"] = 0.1
    payload["proposals"][0]["confidence"] = 0.95
    response = LLMEnrichmentResponse.model_validate(payload)

    result = merge_llm_enrichment(alpha, response)
    enriched = result.enriched_persona

    assert enriched.proactive_preferences.pre_activity_tolerance.value == 0.86
    assert enriched.proactive_preferences.pre_activity_tolerance.source == FieldSource.SOURCE_DERIVED
    assert "pre_activity_tolerance" in result.skipped_fields
    skipped = enriched.source_metadata["llm_enrichment_runs"][-1]["skipped_proposals"]
    assert skipped[0]["reason"] == "high_confidence_source_derived"


def test_merge_llm_enrichment_rejects_low_confidence_proposal() -> None:
    beta = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[1]
    payload = valid_payload(beta.internal_persona_id)
    payload["proposals"][0]["confidence"] = 0.3
    response = LLMEnrichmentResponse.model_validate(payload)

    result = merge_llm_enrichment(beta, response)
    enriched = result.enriched_persona

    assert enriched.proactive_preferences.in_activity_tolerance.confidence == 0.0
    assert "in_activity_tolerance" in result.skipped_fields


def test_merge_llm_enrichment_replaces_rule_based_when_confidence_improves() -> None:
    beta = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[1]
    rule_enriched = RuleBasedPersonaEnricher().enrich(beta).enriched_persona
    response = LLMEnrichmentResponse.model_validate(valid_payload(rule_enriched.internal_persona_id))

    result = merge_llm_enrichment(rule_enriched, response)
    enriched = result.enriched_persona

    assert enriched.proactive_preferences.in_activity_tolerance.source == FieldSource.LLM_INFERRED
    assert enriched.proactive_preferences.in_activity_tolerance.confidence == 0.72


def test_merge_llm_enrichment_rejects_persona_id_mismatch() -> None:
    beta = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[1]
    response = LLMEnrichmentResponse.model_validate(valid_payload("different_persona"))

    with pytest.raises(ValueError):
        merge_llm_enrichment(beta, response)
