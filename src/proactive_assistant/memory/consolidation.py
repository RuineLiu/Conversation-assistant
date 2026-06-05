from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from proactive_assistant.runtime import MemoryCandidate, MemoryCandidateType, MemoryWritePolicy


ACTION_STATUS_TERMS: list[tuple[str, str]] = [
    ("已完成", "done"),
    ("完成", "done"),
    ("done", "done"),
    ("closed", "done"),
    ("闭环", "done"),
    ("blocked", "blocked"),
    ("阻塞", "blocked"),
    ("进行中", "in_progress"),
    ("in progress", "in_progress"),
    ("pending", "pending"),
    ("待确认", "pending"),
]


def consolidated_memory_metadata(candidate: MemoryCandidate) -> dict[str, Any]:
    """Build retrieval-ready metadata for a MemoryRecord from a candidate.

    Candidate metadata is treated as the highest-trust source. Deterministic
    extraction only fills missing fields so manual or upstream structured values
    are never overwritten.
    """

    candidate_type = MemoryCandidateType(candidate.candidate_type)
    original = dict(candidate.metadata)
    metadata: dict[str, Any] = {
        "candidate_id": candidate.memory_candidate_id,
        "candidate_type": str(candidate.candidate_type),
        "decision_id": candidate.decision_id,
        "source_event_ids": list(candidate.source_event_ids),
        "write_policy": str(candidate.write_policy),
        "reason": candidate.reason,
        "candidate_metadata": original,
        "raw_text": candidate.text,
        "memory_schema_version": "memory_consolidation_v1",
    }
    _copy_if_present(
        original,
        metadata,
        [
            "prompt_category",
            "content_granularity",
            "prd_surface",
            "display_mode",
            "duration_policy",
            "source_capture_ref",
            "privacy_level",
            "privacy_risk",
            "feedback_affinity",
            "project",
            "topic",
            "status",
            "owner",
            "assignee",
            "deadline",
            "normalized_deadline",
            "entity",
            "canonical_entity",
            "normalized_entity",
            "provenance",
            "memory_snapshot_source",
            "meeting_state_object_type",
            "meeting_state_object_id",
            "meeting_state_target_type",
            "meeting_state_target_id",
            "source_utterance_id",
            "source_utterance_ids",
            "source_ts_ms",
            "evidence",
            "gap_type",
            "gap_priority",
        ],
    )
    provenance = _provenance(candidate, original)
    if provenance:
        metadata.setdefault("provenance", provenance)
    if candidate_type == MemoryCandidateType.ACTION_ITEM:
        _fill_action_metadata(candidate.text, metadata, _reference_time(original))
    elif candidate_type in {MemoryCandidateType.MEETING_FACT, MemoryCandidateType.USER_PREFERENCE}:
        _fill_general_metadata(candidate.text, metadata, _reference_time(original))
    elif candidate_type == MemoryCandidateType.PRIVACY_PREFERENCE:
        metadata.setdefault("feedback_affinity", 0.9)
    elif candidate_type == MemoryCandidateType.NEGATIVE_PREFERENCE:
        metadata.setdefault("feedback_affinity", 0.85)
    metadata.setdefault("feedback_affinity", _default_feedback_affinity(candidate_type, MemoryWritePolicy(candidate.write_policy)))
    return metadata


def consolidated_memory_tags(candidate: MemoryCandidate, base_tags: set[str]) -> list[str]:
    metadata = consolidated_memory_metadata(candidate)
    tags = set(base_tags)
    for key in ["prompt_category", "prd_surface", "status"]:
        value = metadata.get(key)
        if value:
            tags.add(str(value))
    if metadata.get("owner") or metadata.get("assignee"):
        tags.add("owner")
    if metadata.get("normalized_deadline") or metadata.get("deadline"):
        tags.add("deadline")
    if metadata.get("normalized_entity"):
        tags.add("entity")
    return sorted(tags)


def _fill_action_metadata(text: str, metadata: dict[str, Any], reference_time: datetime) -> None:
    owner = metadata.get("owner") or metadata.get("assignee") or _extract_owner(text)
    if owner:
        metadata.setdefault("owner", owner)
        metadata.setdefault("assignee", owner)
    deadline = metadata.get("normalized_deadline") or _extract_deadline(text, reference_time)
    if deadline:
        metadata.setdefault("normalized_deadline", deadline)
        metadata.setdefault("deadline", deadline)
    status = metadata.get("status") or _extract_status(text)
    if status:
        metadata.setdefault("status", status)
    entity = metadata.get("normalized_entity") or metadata.get("canonical_entity") or metadata.get("entity") or _extract_action_entity(text, owner)
    if entity:
        metadata.setdefault("normalized_entity", _normalize_entity(entity))
        metadata.setdefault("canonical_entity", entity.strip())


def _fill_general_metadata(text: str, metadata: dict[str, Any], reference_time: datetime) -> None:
    deadline = metadata.get("normalized_deadline") or _extract_deadline(text, reference_time)
    if deadline:
        metadata.setdefault("normalized_deadline", deadline)
    owner = metadata.get("owner") or metadata.get("assignee") or _extract_owner(text)
    if owner:
        metadata.setdefault("owner", owner)
        metadata.setdefault("assignee", owner)
    status = metadata.get("status") or _extract_status(text)
    if status:
        metadata.setdefault("status", status)
    entity = metadata.get("normalized_entity") or metadata.get("canonical_entity") or metadata.get("entity")
    if entity:
        metadata.setdefault("normalized_entity", _normalize_entity(str(entity)))


def _extract_owner(text: str) -> str | None:
    patterns = [
        r"\b([A-Z][A-Za-z0-9_\-]{1,40})\s+owns\b",
        r"\bowner\s*(?:is|=|:)?\s*([A-Z][A-Za-z0-9_\-]{1,40})\b",
        r"\bassigned\s+to\s+([A-Z][A-Za-z0-9_\-]{1,40})\b",
        r"负责人(?:是|为|:|：)?\s*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_\-]{1,20})",
        r"由\s*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_\-]{1,20})\s*负责",
        r"([\u4e00-\u9fff]{2,4})\s*负责",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _strip_punctuation(match.group(1))
    return None


def _extract_deadline(text: str, reference_time: datetime) -> str | None:
    iso = re.search(r"20\d{2}-\d{1,2}-\d{1,2}", text)
    if iso:
        year, month, day = iso.group(0).split("-")
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    chinese = re.search(r"(\d{1,2})月(\d{1,2})[日号]", text)
    if chinese:
        month, day = chinese.groups()
        return f"{reference_time.year:04d}-{int(month):02d}-{int(day):02d}"
    slash = re.search(r"(\d{1,2})/(\d{1,2})", text)
    if slash:
        month, day = slash.groups()
        return f"{reference_time.year:04d}-{int(month):02d}-{int(day):02d}"
    weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    relative = re.search(r"(下周|本周|这周)([一二三四五六日天])", text)
    if relative:
        prefix, weekday = relative.groups()
        base = reference_time.date()
        week_start = base - timedelta(days=base.weekday())
        if prefix == "下周":
            week_start = week_start + timedelta(days=7)
        return (week_start + timedelta(days=weekday_map[weekday])).isoformat()
    return None


def _extract_status(text: str) -> str | None:
    lowered = text.lower()
    for term, status in ACTION_STATUS_TERMS:
        if term in lowered or term in text:
            return status
    return None


def _extract_action_entity(text: str, owner: str | None) -> str | None:
    cleaned = text.strip()
    if owner:
        cleaned = re.sub(rf"^\s*{re.escape(owner)}\s+owns\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"由\s*{re.escape(owner)}\s*负责", "", cleaned)
        cleaned = re.sub(rf"{re.escape(owner)}\s*负责", "", cleaned)
    cleaned = re.sub(r"\b(owner\s*(?:is|=|:)?\s*[A-Z][A-Za-z0-9_\-]{1,40})\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"负责人(?:是|为|:|：)?\s*[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_\-]{1,20}", "", cleaned)
    cleaned = re.sub(r"\bby\s+20\d{2}-\d{1,2}-\d{1,2}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"20\d{2}-\d{1,2}-\d{1,2}", "", cleaned)
    cleaned = re.sub(r"\d{1,2}月\d{1,2}[日号]", "", cleaned)
    cleaned = re.sub(r"(下周|本周|这周)[一二三四五六日天]", "", cleaned)
    cleaned = re.sub(r"\b(deadline|follow-up|follow up|pending confirmation)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = _strip_punctuation(cleaned)
    return cleaned or None


def _provenance(candidate: MemoryCandidate, metadata: dict[str, Any]) -> list[str]:
    existing = metadata.get("provenance")
    if isinstance(existing, list):
        return [str(item) for item in existing if str(item)]
    if isinstance(existing, str) and existing:
        return [existing]
    refs: list[str] = []
    for key in ["source_refs", "source_capture_ref"]:
        value = metadata.get(key)
        if isinstance(value, list):
            refs.extend(str(item) for item in value if str(item))
        elif isinstance(value, str) and value:
            refs.append(value)
    refs.append(f"decision:{candidate.decision_id}")
    refs.extend(f"feedback:{event_id}" for event_id in candidate.source_event_ids)
    return list(dict.fromkeys(refs))


def _copy_if_present(source: dict[str, Any], target: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        value = source.get(key)
        if value is not None and value != "":
            target[key] = value


def _reference_time(metadata: dict[str, Any]) -> datetime:
    value = metadata.get("reference_time") or metadata.get("meeting_date") or metadata.get("session_date")
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


def _default_feedback_affinity(candidate_type: MemoryCandidateType, write_policy: MemoryWritePolicy) -> float:
    if candidate_type == MemoryCandidateType.PRIVACY_PREFERENCE:
        return 0.9
    if candidate_type == MemoryCandidateType.NEGATIVE_PREFERENCE:
        return 0.85
    if candidate_type == MemoryCandidateType.USER_PREFERENCE:
        return 0.78 if write_policy == MemoryWritePolicy.ELIGIBLE else 0.62
    if candidate_type == MemoryCandidateType.ACTION_ITEM:
        return 0.55
    return 0.5


def _normalize_entity(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_punctuation(text).lower()).strip()


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[\s,.;:，。；：!?！？]+$", "", text.strip())
