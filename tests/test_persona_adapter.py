from pathlib import Path

import pytest

from proactive_assistant.adapters import PersonaDatasetAdapter
from proactive_assistant.io import read_jsonl
from proactive_assistant.schemas import Persona
from proactive_assistant.schemas.persona import FieldSource


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "persona"
    / "sample_persona_records.jsonl"
)


def test_adapter_groups_records_and_builds_personas() -> None:
    personas = PersonaDatasetAdapter().from_path(FIXTURE_PATH)

    assert [persona.source_persona_id for persona in personas] == [
        "persona_alpha",
        "persona_beta",
    ]
    assert personas[0].source_metadata["record_count"] == 2
    assert personas[1].source_metadata["record_count"] == 1


def test_adapter_extracts_source_derived_fields_when_available() -> None:
    alpha = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[0]

    assert alpha.demographic_stub.occupation_type == "project_manager"
    assert alpha.big_five.conscientiousness == 0.91
    assert alpha.proactive_preferences.pre_activity_tolerance.value == 0.86
    assert alpha.proactive_preferences.pre_activity_tolerance.source == FieldSource.SOURCE_DERIVED
    assert alpha.proactive_preferences.notification_budget_per_hour == 3


def test_adapter_preserves_preference_samples_for_enrichment() -> None:
    alpha = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[0]

    samples = alpha.source_metadata["preference_samples"]
    assert any("planning before the activity" in sample for sample in samples)


def test_adapter_marks_unknown_proactive_fields_as_pending() -> None:
    beta = PersonaDatasetAdapter().from_path(FIXTURE_PATH)[1]

    assert beta.proactive_preferences.privacy_sensitivity.value == 0.93
    assert beta.proactive_preferences.privacy_sensitivity.source == FieldSource.SOURCE_DERIVED
    assert beta.proactive_preferences.in_activity_tolerance.confidence == 0.0
    assert "in_activity_tolerance" in beta.source_metadata["pending_enrichment_fields"]


def test_adapter_writes_normalized_jsonl(tmp_path: Path) -> None:
    adapter = PersonaDatasetAdapter()
    personas = adapter.from_path(FIXTURE_PATH)
    output_path = tmp_path / "normalized_personas.jsonl"

    adapter.write_normalized_jsonl(personas, output_path)
    records = list(read_jsonl(output_path))
    restored = [Persona.model_validate(record) for record in records]

    assert len(restored) == 2
    assert restored[0].source_license == "cc-by-nc-sa-4.0"


def test_adapter_rejects_unsupported_file_type(tmp_path: Path) -> None:
    path = tmp_path / "personas.txt"
    path.write_text("not supported")

    with pytest.raises(ValueError):
        PersonaDatasetAdapter().from_path(path)
