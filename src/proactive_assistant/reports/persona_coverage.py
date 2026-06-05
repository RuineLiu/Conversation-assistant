from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from proactive_assistant.schemas.persona import Persona, PreferenceScore


PERSONA_SCORE_FIELDS = {
    "proactive_preferences": (
        "pre_activity_tolerance",
        "in_activity_tolerance",
        "post_activity_tolerance",
        "detail_preference",
        "permission_first",
        "privacy_sensitivity",
    ),
    "context_sensitivity": (
        "meeting_interrupt_cost",
        "solo_task_interrupt_cost",
        "social_interrupt_cost",
        "deadline_help_value",
    ),
    "feedback_style": (
        "explicit_feedback_rate",
        "negative_feedback_threshold",
        "politeness_bias",
    ),
}


def build_persona_coverage_report(personas: list[Persona]) -> dict[str, Any]:
    """Summarize source coverage and pending enrichment fields for personas."""

    total = len(personas)
    field_stats: dict[str, dict[str, Any]] = {}
    source_totals: Counter[str] = Counter()
    pending_totals: Counter[str] = Counter()

    for group_name, field_names in PERSONA_SCORE_FIELDS.items():
        for field_name in field_names:
            key = field_name
            sources: Counter[str] = Counter()
            confidence_sum = 0.0
            pending_count = 0

            for persona in personas:
                score = _get_score(persona, group_name, field_name)
                sources[score.source.value] += 1
                source_totals[score.source.value] += 1
                confidence_sum += score.confidence
                pending_fields = set(
                    (persona.source_metadata or {}).get("pending_enrichment_fields", [])
                )
                if field_name in pending_fields:
                    pending_count += 1
                    pending_totals[field_name] += 1

            field_stats[key] = {
                "sources": dict(sorted(sources.items())),
                "pending_count": pending_count,
                "pending_rate": _rate(pending_count, total),
                "average_confidence": confidence_sum / total if total else 0.0,
            }

    return {
        "total_personas": total,
        "field_stats": field_stats,
        "source_totals": dict(sorted(source_totals.items())),
        "pending_totals": dict(sorted(pending_totals.items())),
    }


def _get_score(persona: Persona, group_name: str, field_name: str) -> PreferenceScore:
    group = getattr(persona, group_name)
    return getattr(group, field_name)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total
