from __future__ import annotations

from hashlib import sha1
from typing import Any

from proactive_assistant.meeting_state import ActionItem, Decision, MeetingGap, MeetingState, Risk
from proactive_assistant.runtime import MemoryCandidate, MemoryCandidateType, MemoryWritePolicy


SNAPSHOT_DECISION_PREFIX = "meeting_state_snapshot"


def memory_candidates_from_meeting_state(
    state: MeetingState,
    *,
    gaps: list[MeetingGap] | None = None,
) -> list[MemoryCandidate]:
    """Convert structured meeting state into deterministic memory candidates."""

    candidates: list[MemoryCandidate] = []
    for item in state.action_items:
        candidates.append(_candidate_from_action_item(state, item))
    for item in state.decisions:
        candidates.append(_candidate_from_decision(state, item))
    for item in state.risks:
        candidates.append(_candidate_from_risk(state, item))
    for item in gaps or []:
        candidates.append(_candidate_from_gap(state, item))
    return candidates


def _candidate_from_action_item(state: MeetingState, item: ActionItem) -> MemoryCandidate:
    metadata = _base_metadata(state, object_type="action_item", object_id=item.action_item_id)
    metadata.update(
        {
            "source_utterance_id": item.source_utterance_id,
            "source_ts_ms": item.source_ts_ms,
            "owner": item.owner,
            "assignee": item.owner,
            "deadline": item.deadline,
            "status": item.status,
            "entity": item.desc,
            "canonical_entity": item.desc,
            "evidence": item.evidence,
            "provenance": [f"transcript:{item.source_utterance_id}"],
            "tags": ["meeting_state", "action_item"],
        }
    )
    has_commitment = bool(item.owner or item.deadline or item.next_step)
    return _candidate(
        state,
        object_type="action_item",
        object_id=item.action_item_id,
        candidate_type=MemoryCandidateType.ACTION_ITEM,
        text=item.evidence or item.desc,
        confidence=0.78 if has_commitment else 0.62,
        write_policy=MemoryWritePolicy.ELIGIBLE if has_commitment else MemoryWritePolicy.NEEDS_CONFIRMATION,
        reason="meeting state action item snapshot",
        metadata=metadata,
    )


def _candidate_from_decision(state: MeetingState, item: Decision) -> MemoryCandidate:
    metadata = _base_metadata(state, object_type="decision", object_id=item.decision_id)
    metadata.update(
        {
            "source_utterance_id": item.source_utterance_id,
            "source_ts_ms": item.ts_ms,
            "topic": item.topic,
            "entity": item.topic,
            "canonical_entity": item.topic,
            "status": "decided" if item.conclusion else "open",
            "evidence": item.evidence,
            "provenance": [f"transcript:{item.source_utterance_id}"],
            "tags": ["meeting_state", "decision"],
        }
    )
    text = f"Decision topic: {item.topic}."
    if item.conclusion:
        text += f" Conclusion: {item.conclusion}"
    else:
        text += " Conclusion is not tracked yet."
    return _candidate(
        state,
        object_type="decision",
        object_id=item.decision_id,
        candidate_type=MemoryCandidateType.MEETING_FACT,
        text=text,
        confidence=0.76 if item.conclusion else 0.6,
        write_policy=MemoryWritePolicy.ELIGIBLE if item.conclusion else MemoryWritePolicy.NEEDS_CONFIRMATION,
        reason="meeting state decision snapshot",
        metadata=metadata,
    )


def _candidate_from_risk(state: MeetingState, item: Risk) -> MemoryCandidate:
    metadata = _base_metadata(state, object_type="risk", object_id=item.risk_id)
    metadata.update(
        {
            "source_utterance_id": item.source_utterance_id,
            "source_ts_ms": item.ts_ms,
            "entity": item.desc,
            "canonical_entity": item.desc,
            "status": "closed" if item.closed else "open",
            "evidence": item.evidence,
            "provenance": [f"transcript:{item.source_utterance_id}"],
            "tags": ["meeting_state", "risk"],
        }
    )
    text = f"Risk: {item.desc}. Status: {'closed' if item.closed else 'open'}."
    return _candidate(
        state,
        object_type="risk",
        object_id=item.risk_id,
        candidate_type=MemoryCandidateType.MEETING_FACT,
        text=text,
        confidence=0.68,
        write_policy=MemoryWritePolicy.ELIGIBLE,
        reason="meeting state risk snapshot",
        metadata=metadata,
    )


def _candidate_from_gap(state: MeetingState, item: MeetingGap) -> MemoryCandidate:
    metadata = _base_metadata(state, object_type="gap", object_id=item.gap_id)
    metadata.update(
        {
            "gap_type": item.gap_type,
            "gap_priority": item.priority,
            "meeting_state_target_type": item.object_type,
            "meeting_state_target_id": item.object_id,
            "source_utterance_ids": list(item.source_utterance_ids),
            "source_ts_ms": item.first_seen_ms,
            "entity": item.text,
            "canonical_entity": item.text,
            "status": "open",
            "evidence": item.text,
            "provenance": [f"transcript:{utterance_id}" for utterance_id in item.source_utterance_ids],
            "tags": ["meeting_state", "gap", str(item.gap_type), str(item.priority)],
        }
    )
    text = f"Meeting gap: {item.gap_type}. {item.text} Reason: {item.reason}"
    return _candidate(
        state,
        object_type="gap",
        object_id=item.gap_id,
        candidate_type=MemoryCandidateType.MEETING_FACT,
        text=text,
        confidence=0.64,
        write_policy=MemoryWritePolicy.NEEDS_CONFIRMATION,
        reason="meeting state gap snapshot",
        metadata=metadata,
    )


def _candidate(
    state: MeetingState,
    *,
    object_type: str,
    object_id: str,
    candidate_type: MemoryCandidateType,
    text: str,
    confidence: float,
    write_policy: MemoryWritePolicy,
    reason: str,
    metadata: dict[str, Any],
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_candidate_id=_candidate_id(state.session_id, object_type, object_id),
        decision_id=f"{SNAPSHOT_DECISION_PREFIX}:{state.session_id}",
        session_id=state.session_id,
        source_event_ids=[],
        candidate_type=candidate_type,
        text=text,
        confidence=confidence,
        write_policy=write_policy,
        reason=reason,
        metadata=metadata,
    )


def _base_metadata(state: MeetingState, *, object_type: str, object_id: str) -> dict[str, Any]:
    return {
        "org_id": state.org_id,
        "subject_user_id": state.subject_user_id,
        "meeting_state_object_type": object_type,
        "meeting_state_object_id": object_id,
        "session_id": state.session_id,
        "reference_time": state.started_at,
        "memory_snapshot_source": "meeting_state_snapshot_v1",
    }


def _candidate_id(session_id: str, object_type: str, object_id: str) -> str:
    digest = sha1(f"{session_id}:{object_type}:{object_id}".encode("utf-8")).hexdigest()[:12]
    return f"memcand_mstate_{digest}"
