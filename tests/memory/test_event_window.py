"""Phase 2: event start/end time semantics + entity-based dedup.

Rules consolidation must honor:
- Both start + end provided → both normalized.
- Only end (deadline) provided → start = reference date.
- Only start provided → end = "未定".
- Same entity mentioned later → merge into existing memory, not create
  a sibling.
"""

from datetime import UTC, datetime

from proactive_assistant.memory.consolidation import (
    EVENT_END_UNDEFINED,
    consolidated_memory_metadata,
)
from proactive_assistant.prompting import PrivacyLevel
from proactive_assistant.runtime import (
    MemoryCandidate,
    MemoryCandidateType,
    MemoryWritePolicy,
)


def _candidate(metadata: dict) -> MemoryCandidate:
    return MemoryCandidate(
        memory_candidate_id="memcand_test",
        decision_id="dec_test",
        session_id="session_test",
        source_event_ids=[],
        candidate_type=MemoryCandidateType.ACTION_ITEM,
        text="客户报价确认。",
        confidence=0.85,
        write_policy=MemoryWritePolicy.ELIGIBLE,
        privacy_level=PrivacyLevel.LOW,
        reason="test",
        metadata={
            "reference_time": datetime(2026, 6, 10, tzinfo=UTC),
            "canonical_entity": "客户报价确认",
            **metadata,
        },
    )


def test_event_window_both_start_and_end_normalized() -> None:
    candidate = _candidate({"start_time": "2026-06-15", "end_time": "2026-06-20"})

    meta = consolidated_memory_metadata(candidate)

    assert meta["normalized_start_time"] == "2026-06-15"
    assert meta["normalized_end_time"] == "2026-06-20"


def test_event_window_only_end_fills_start_from_reference_date() -> None:
    candidate = _candidate({"end_time": "2026-06-15"})

    meta = consolidated_memory_metadata(candidate)

    assert meta["normalized_start_time"] == "2026-06-10"
    assert meta["normalized_end_time"] == "2026-06-15"


def test_event_window_only_start_marks_end_as_undefined() -> None:
    candidate = _candidate({"start_time": "2026-06-15"})

    meta = consolidated_memory_metadata(candidate)

    assert meta["normalized_start_time"] == "2026-06-15"
    assert meta["normalized_end_time"] == EVENT_END_UNDEFINED


def test_event_window_deadline_treated_as_end_when_no_explicit_end() -> None:
    candidate = _candidate({"deadline": "2026-06-15"})

    meta = consolidated_memory_metadata(candidate)

    assert meta["normalized_end_time"] == "2026-06-15"
    assert meta["normalized_start_time"] == "2026-06-10"


def test_event_window_absent_when_nothing_given() -> None:
    candidate = _candidate({})

    meta = consolidated_memory_metadata(candidate)

    assert "normalized_start_time" not in meta
    assert "normalized_end_time" not in meta


def test_event_window_chinese_relative_dates_normalize_to_iso() -> None:
    candidate = _candidate({"end_time": "下周五"})

    meta = consolidated_memory_metadata(candidate)

    assert meta["normalized_end_time"].startswith("2026-06-")
    assert meta["normalized_start_time"] == "2026-06-10"
