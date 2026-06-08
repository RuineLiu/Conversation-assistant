from __future__ import annotations

import re
from dataclasses import dataclass

from proactive_assistant.memory.contracts import (
    MemoryMergeAction,
    MemoryMergeDecision,
    MemoryRecord,
    MemoryType,
    MemoryWriteStatus,
)


CONFLICT_METADATA_FIELDS = ["owner", "assignee", "deadline", "normalized_deadline", "status", "entity", "canonical_entity"]
ENTITY_METADATA_FIELDS = ["normalized_entity", "canonical_entity", "entity", "topic", "project"]
SUPERSEDE_TERMS = ["改为", "更新为", "不再", "替换", "由", "换成", "修正", "更正", "now", "instead", "replace"]


@dataclass(frozen=True)
class MemoryMergeCandidate:
    memory: MemoryRecord
    similarity: float
    same_target: bool
    reasons: list[str]


def detect_memory_merge(proposed: MemoryRecord, existing_memories: list[MemoryRecord]) -> MemoryMergeDecision:
    """Classify how a proposed memory relates to existing memories.

    The detector is deterministic and intentionally conservative. It only
    chooses a non-create path when the memory type and target/entity evidence
    are strong enough to avoid accidental cross-topic merges.
    """

    matches = [_match_candidate(proposed, existing) for existing in existing_memories if existing.memory_id != proposed.memory_id]
    matches = [match for match in matches if match is not None]
    if not matches:
        return _decision(MemoryMergeAction.CREATE, proposed=proposed, reasons=["no_related_memory"])

    best = sorted(matches, key=lambda item: (-item.same_target, -item.similarity, item.memory.memory_id))[0]
    existing = best.memory
    if MemoryWriteStatus(existing.write_status) in {
        MemoryWriteStatus.FORGOTTEN,
        MemoryWriteStatus.REJECTED,
        MemoryWriteStatus.ARCHIVED,
    }:
        return _decision(
            MemoryMergeAction.BLOCKED,
            proposed=proposed,
            existing=existing,
            similarity=best.similarity,
            reasons=[f"related_memory_is_{existing.write_status}"],
            resolution_status="blocked",
        )

    changed_fields = _changed_structured_fields(existing, proposed)
    non_text_changed_fields = [field for field in changed_fields if field != "text"]
    conflict_fields = _conflict_fields(existing, proposed)
    evidence_refs = _merge_unique(existing.source_ids, proposed.source_ids)
    exact_text = _normalize_text(existing.text) == _normalize_text(proposed.text)
    if not changed_fields and exact_text:
        return _decision(
            MemoryMergeAction.DUPLICATE,
            proposed=proposed,
            existing=existing,
            similarity=best.similarity,
            changed_fields=[],
            evidence_refs=evidence_refs,
            reasons=[*best.reasons, "same_text_and_structured_fields"],
        )
    if not non_text_changed_fields and best.similarity >= 0.35:
        return _decision(
            MemoryMergeAction.REINFORCEMENT,
            proposed=proposed,
            existing=existing,
            similarity=best.similarity,
            changed_fields=[],
            evidence_refs=evidence_refs,
            reasons=[*best.reasons, "same_structured_fields_new_evidence"],
        )
    if conflict_fields:
        action = MemoryMergeAction.SUPERSEDE if _has_supersede_signal(proposed) else MemoryMergeAction.CONFLICT
        return _decision(
            action,
            proposed=proposed,
            existing=existing,
            similarity=best.similarity,
            changed_fields=changed_fields,
            conflict_fields=conflict_fields,
            evidence_refs=evidence_refs,
            reasons=[*best.reasons, "structured_conflict"],
            resolution_status="pending_confirmation",
        )
    return _decision(
        MemoryMergeAction.UPDATE,
        proposed=proposed,
        existing=existing,
        similarity=best.similarity,
        changed_fields=changed_fields,
        evidence_refs=evidence_refs,
        reasons=[*best.reasons, "non_conflicting_update"],
    )


def _match_candidate(proposed: MemoryRecord, existing: MemoryRecord) -> MemoryMergeCandidate | None:
    if str(existing.memory_type) != str(proposed.memory_type):
        return None
    if existing.org_id != proposed.org_id or existing.user_id != proposed.user_id:
        return None
    same_target = _same_target(existing, proposed)
    similarity = _text_similarity(existing.text, proposed.text)
    reasons: list[str] = []
    if same_target:
        reasons.append("same_target")
    if similarity >= 0.72:
        reasons.append("text_similarity")
    if not same_target and similarity < 0.86:
        return None
    if MemoryType(proposed.memory_type) == MemoryType.ACTION_ITEM and not same_target and similarity < 0.92:
        return None
    return MemoryMergeCandidate(memory=existing, similarity=similarity, same_target=same_target, reasons=reasons)


def _same_target(existing: MemoryRecord, proposed: MemoryRecord) -> bool:
    existing_values = _target_values(existing)
    proposed_values = _target_values(proposed)
    if not existing_values or not proposed_values:
        return False
    return bool(existing_values & proposed_values)


def _target_values(memory: MemoryRecord) -> set[str]:
    values = set()
    for field in ENTITY_METADATA_FIELDS:
        value = memory.metadata.get(field)
        if value:
            values.add(_normalize_text(str(value)))
    return {value for value in values if value}


def _changed_structured_fields(existing: MemoryRecord, proposed: MemoryRecord) -> list[str]:
    changed: list[str] = []
    if _normalize_text(existing.text) != _normalize_text(proposed.text):
        changed.append("text")
    for field in CONFLICT_METADATA_FIELDS:
        before = _normalized_metadata_value(existing, field)
        after = _normalized_metadata_value(proposed, field)
        if before != after:
            changed.append(f"metadata.{field}")
    return changed


def _conflict_fields(existing: MemoryRecord, proposed: MemoryRecord) -> list[str]:
    conflicts: list[str] = []
    for field in ["owner", "assignee", "deadline", "normalized_deadline", "status"]:
        before = _normalized_metadata_value(existing, field)
        after = _normalized_metadata_value(proposed, field)
        if before and after and before != after:
            conflicts.append(f"metadata.{field}")
    return conflicts


def _normalized_metadata_value(memory: MemoryRecord, field: str) -> str:
    return _normalize_text(str(memory.metadata.get(field, "")))


def _has_supersede_signal(memory: MemoryRecord) -> bool:
    text = " ".join([memory.text, str(memory.metadata.get("reason", "")), str(memory.metadata.get("update_reason", ""))]).lower()
    return any(term in text for term in SUPERSEDE_TERMS)


def _text_similarity(left: str, right: str) -> float:
    left_terms = set(_terms(left))
    right_terms = set(_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return round(len(left_terms & right_terms) / len(left_terms | right_terms), 4)


def _terms(text: str) -> list[str]:
    normalized = text.lower()
    terms = re.findall(r"[a-z0-9_]+", normalized)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.append(chunk)
        for size in (2, 3):
            if len(chunk) >= size:
                terms.extend(chunk[index : index + size] for index in range(0, len(chunk) - size + 1))
    return [term for term in terms if term]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower()).strip()


def _merge_unique(first: list[str], second: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in [*first, *second]:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _decision(
    action: MemoryMergeAction,
    *,
    proposed: MemoryRecord,
    existing: MemoryRecord | None = None,
    similarity: float = 0.0,
    changed_fields: list[str] | None = None,
    conflict_fields: list[str] | None = None,
    evidence_refs: list[str] | None = None,
    reasons: list[str] | None = None,
    resolution_status: str = "resolved",
) -> MemoryMergeDecision:
    return MemoryMergeDecision(
        action=action,
        existing_memory_id=existing.memory_id if existing is not None else None,
        proposed_memory_id=proposed.memory_id,
        similarity=similarity,
        changed_fields=changed_fields or [],
        conflict_fields=conflict_fields or [],
        evidence_refs=evidence_refs or list(proposed.source_ids),
        reasons=reasons or [],
        resolution_status=resolution_status,
    )
