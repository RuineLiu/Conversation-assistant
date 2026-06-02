from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from proactive_assistant.io.jsonl import read_jsonl, write_jsonl
from proactive_assistant.schemas.persona import (
    BigFive,
    ContextSensitivity,
    DemographicStub,
    FeedbackStyle,
    FieldSource,
    Persona,
    PreferenceScore,
    ProactivePreferences,
)


Record = Mapping[str, Any]

ADAPTER_VERSION = "persona_dataset_v0"
DEFAULT_SOURCE = "SynthLabsAI/PERSONA"
DEFAULT_LICENSE = "cc-by-nc-sa-4.0"

PERSONA_ID_KEYS = (
    "persona_id",
    "source_persona_id",
    "person_id",
    "profile_id",
    "user_id",
)
PERSONA_TEXT_KEYS = (
    "persona",
    "persona_text",
    "persona_description",
    "profile",
    "profile_text",
    "description",
    "user_profile",
)

PREFERENCE_SAMPLE_KEYS = (
    "chosen",
    "rejected",
    "preference",
    "preference_text",
    "response",
    "answer",
    "prompt",
)


class PersonaDatasetAdapter:
    """Convert PERSONA-like records into the project's internal Persona schema.

    This adapter intentionally does not call Hugging Face, LLMs, or simulators.
    It consumes already-downloaded local files or in-memory records.
    """

    def __init__(
        self,
        source: str = DEFAULT_SOURCE,
        source_license: str = DEFAULT_LICENSE,
    ) -> None:
        self.source = source
        self.source_license = source_license

    def from_path(self, path: str | Path) -> list[Persona]:
        records = self._load_records(Path(path))
        return self.from_records(records)

    def from_records(self, records: Iterable[Record]) -> list[Persona]:
        grouped = self._group_by_persona(records)
        personas = [
            self._build_persona(source_persona_id=persona_id, records=persona_records)
            for persona_id, persona_records in grouped.items()
        ]
        return sorted(personas, key=lambda persona: persona.internal_persona_id)

    def write_normalized_jsonl(self, personas: Iterable[Persona], path: str | Path) -> None:
        write_jsonl(Path(path), (persona.model_dump(mode="json") for persona in personas))

    def _load_records(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix == ".jsonl":
            return list(read_jsonl(path))
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                return payload["records"]
            raise ValueError("JSON persona input must be a list or contain a records list")
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet support requires pyarrow. Install with: "
                    'uv pip install -e ".[parquet]"'
                ) from exc

            table = pq.read_table(path)
            return table.to_pylist()
        raise ValueError(f"Unsupported persona input format: {path.suffix}")

    def _group_by_persona(self, records: Iterable[Record]) -> dict[str, list[Record]]:
        grouped: dict[str, list[Record]] = defaultdict(list)
        for record in records:
            persona_id = self._extract_persona_id(record)
            grouped[persona_id].append(record)
        return dict(grouped)

    def _extract_persona_id(self, record: Record) -> str:
        for key in PERSONA_ID_KEYS:
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

        persona_text = self._extract_persona_text([record])
        if persona_text:
            digest = hashlib.sha256(persona_text.encode("utf-8")).hexdigest()[:12]
            return f"persona_hash_{digest}"

        digest = hashlib.sha256(json.dumps(dict(record), sort_keys=True).encode()).hexdigest()[:12]
        return f"record_hash_{digest}"

    def _build_persona(self, source_persona_id: str, records: list[Record]) -> Persona:
        source_columns = sorted({key for record in records for key in record.keys()})
        persona_text = self._extract_persona_text(records)
        preference_samples = self._extract_preference_samples(records)
        pending_fields: list[str] = []

        proactive_preferences = ProactivePreferences(
            pre_activity_tolerance=self._extract_score(
                records,
                "pre_activity_tolerance",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            in_activity_tolerance=self._extract_score(
                records,
                "in_activity_tolerance",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            post_activity_tolerance=self._extract_score(
                records,
                "post_activity_tolerance",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            detail_preference=self._extract_score(
                records,
                "detail_preference",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            permission_first=self._extract_score(
                records,
                "permission_first",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            privacy_sensitivity=self._extract_score(
                records,
                "privacy_sensitivity",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            notification_budget_per_hour=self._extract_notification_budget(records),
        )

        context_sensitivity = ContextSensitivity(
            meeting_interrupt_cost=self._extract_score(
                records,
                "meeting_interrupt_cost",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            solo_task_interrupt_cost=self._extract_score(
                records,
                "solo_task_interrupt_cost",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            social_interrupt_cost=self._extract_score(
                records,
                "social_interrupt_cost",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            deadline_help_value=self._extract_score(
                records,
                "deadline_help_value",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
        )

        feedback_style = FeedbackStyle(
            explicit_feedback_rate=self._extract_score(
                records,
                "explicit_feedback_rate",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            negative_feedback_threshold=self._extract_score(
                records,
                "negative_feedback_threshold",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
            politeness_bias=self._extract_score(
                records,
                "politeness_bias",
                default_value=0.5,
                pending_fields=pending_fields,
            ),
        )

        return Persona(
            source=self.source,
            source_persona_id=source_persona_id,
            internal_persona_id=self._internal_persona_id(source_persona_id),
            source_license=self.source_license,
            demographic_stub=self._extract_demographic_stub(records),
            big_five=self._extract_big_five(records),
            proactive_preferences=proactive_preferences,
            context_sensitivity=context_sensitivity,
            feedback_style=feedback_style,
            source_metadata={
                "adapter_version": ADAPTER_VERSION,
                "record_count": len(records),
                "source_columns": source_columns,
                "persona_text": persona_text,
                "preference_samples": preference_samples,
                "pending_enrichment_fields": sorted(set(pending_fields)),
            },
        )

    def _internal_persona_id(self, source_persona_id: str) -> str:
        normalized = source_persona_id.strip().replace(" ", "_")
        if normalized.startswith("P"):
            return normalized
        return f"P_{normalized}"

    def _extract_demographic_stub(self, records: list[Record]) -> DemographicStub:
        return DemographicStub(
            age_band=self._first_string(records, ("age_band", "age_range", "age")),
            occupation_type=self._first_string(
                records,
                ("occupation_type", "occupation", "job", "profession"),
            ),
            daily_routine_density=self._first_string(
                records,
                ("daily_routine_density", "routine_density", "schedule_density"),
            ),
        )

    def _extract_big_five(self, records: list[Record]) -> BigFive:
        nested = self._first_mapping(records, "big_five")
        values: dict[str, float] = {}
        for key in (
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ):
            raw = nested.get(key) if nested else None
            if raw is None:
                raw = self._first_value(records, (key,))
            values[key] = self._bounded_float(raw, default=0.5)
        return BigFive(**values)

    def _extract_score(
        self,
        records: list[Record],
        key: str,
        default_value: float,
        pending_fields: list[str],
    ) -> PreferenceScore:
        raw = self._first_nested_or_flat_value(records, key)
        if raw is None:
            pending_fields.append(key)
            return PreferenceScore(
                value=default_value,
                source=FieldSource.LLM_INFERRED,
                confidence=0.0,
            )

        if isinstance(raw, Mapping):
            value = raw.get("value")
            source = raw.get("source", FieldSource.SOURCE_DERIVED)
            confidence = raw.get("confidence", 0.8)
            return PreferenceScore(
                value=self._bounded_float(value, default=default_value),
                source=FieldSource(str(source)),
                confidence=self._bounded_float(confidence, default=0.8),
            )

        return PreferenceScore(
            value=self._bounded_float(raw, default=default_value),
            source=FieldSource.SOURCE_DERIVED,
            confidence=0.8,
        )

    def _extract_notification_budget(self, records: list[Record]) -> int:
        raw = self._first_nested_or_flat_value(records, "notification_budget_per_hour")
        if raw is None:
            return 2
        try:
            return max(0, min(60, int(raw)))
        except (TypeError, ValueError):
            return 2

    def _first_nested_or_flat_value(self, records: list[Record], key: str) -> Any:
        for namespace in ("proactive_preferences", "context_sensitivity", "feedback_style"):
            nested = self._first_mapping(records, namespace)
            if nested and key in nested:
                return nested[key]
        return self._first_value(records, (key,))

    def _extract_persona_text(self, records: list[Record]) -> str | None:
        for key in PERSONA_TEXT_KEYS:
            value = self._first_string(records, (key,))
            if value:
                return value
        return None

    def _first_string(self, records: list[Record], keys: tuple[str, ...]) -> str | None:
        value = self._first_value(records, keys)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _first_value(self, records: list[Record], keys: tuple[str, ...]) -> Any:
        for record in records:
            for key in keys:
                value = record.get(key)
                if value is not None and value != "":
                    return value
        return None

    def _first_mapping(self, records: list[Record], key: str) -> Mapping[str, Any] | None:
        for record in records:
            value = record.get(key)
            if isinstance(value, Mapping):
                return value
        return None

    def _extract_preference_samples(self, records: list[Record], limit: int = 8) -> list[str]:
        samples: list[str] = []
        for record in records:
            parts = []
            for key in PREFERENCE_SAMPLE_KEYS:
                value = record.get(key)
                if value is not None and str(value).strip():
                    parts.append(f"{key}: {str(value).strip()}")
            if parts:
                samples.append(" | ".join(parts))
            if len(samples) >= limit:
                break
        return samples

    def _bounded_float(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(0.0, min(1.0, number))
