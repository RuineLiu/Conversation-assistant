from datetime import UTC, datetime
from typing import TypeVar

from proactive_assistant.meeting_state.contracts import (
    ActionItem,
    Decision,
    MeetingGap,
    MeetingState,
    MeetingStateUpdateResult,
    OpenQuestion,
)
from proactive_assistant.meeting_state.rules import (
    extract_action_item,
    extract_decision,
    extract_question,
    extract_refs,
    extract_risk,
    looks_like_end_signal,
    maybe_answer_questions,
    scan_gaps,
    utterance_from_segment,
)
from proactive_assistant.sessions import TranscriptSegmentRecord


T = TypeVar("T")


class MeetingStateTracker:
    """Rule-based v0 tracker for live meeting structure and gaps."""

    def create_state(
        self,
        session_id: str,
        *,
        org_id: str = "default_org",
        subject_user_id: str = "default_user",
        participants: list[str] | None = None,
        scheduled_end_ms: int | None = None,
    ) -> MeetingState:
        return MeetingState(
            session_id=session_id,
            org_id=org_id,
            subject_user_id=subject_user_id,
            participants=participants or [],
            scheduled_end_ms=scheduled_end_ms,
        )

    def update_from_segment(
        self,
        state: MeetingState,
        segment: TranscriptSegmentRecord,
        *,
        force_end_summary: bool | None = None,
    ) -> MeetingStateUpdateResult:
        if segment.session_id != state.session_id:
            raise ValueError("segment session_id must match meeting state session_id")
        utterance = utterance_from_segment(segment)
        participants = _add_participant(state.participants, utterance.speaker)
        utterances = _append_unique(state.utterances, utterance, key="utterance_id")

        updated_questions, answered_questions = maybe_answer_questions(state.open_questions, utterance)
        question = extract_question(utterance)
        added_questions = [] if question is None or _has_id(updated_questions, question.question_id, "question_id") else [question]
        questions = updated_questions + added_questions

        action_item = extract_action_item(utterance)
        added_action_items = (
            [] if action_item is None or _has_id(state.action_items, action_item.action_item_id, "action_item_id") else [action_item]
        )
        action_items = state.action_items + added_action_items

        decision = extract_decision(utterance)
        updated_decisions, changed_decisions = _update_open_decisions(state.decisions, decision)
        added_decisions = []
        if decision is not None and not changed_decisions and not _has_id(updated_decisions, decision.decision_id, "decision_id"):
            added_decisions = [decision]
        decisions = updated_decisions + added_decisions

        risk = extract_risk(utterance)
        added_risks = [] if risk is None or _has_id(state.risks, risk.risk_id, "risk_id") else [risk]
        risks = state.risks + added_risks

        refs = extract_refs(utterance)
        added_refs = [ref for ref in refs if not _has_id(state.mentioned_refs, ref.mentioned_ref_id, "mentioned_ref_id")]
        mentioned_refs = state.mentioned_refs + added_refs

        next_state = state.model_copy(
            update={
                "participants": participants,
                "utterances": utterances,
                "open_questions": questions,
                "action_items": action_items,
                "decisions": decisions,
                "risks": risks,
                "mentioned_refs": mentioned_refs,
                "updated_at": datetime.now(UTC),
            }
        )
        gaps = self.scan_state_gaps(
            next_state,
            current_ms=utterance.end_ms,
            force_end_summary=looks_like_end_signal(utterance.text) if force_end_summary is None else force_end_summary,
        )
        return MeetingStateUpdateResult(
            state=next_state,
            added_questions=added_questions,
            answered_questions=answered_questions,
            added_action_items=added_action_items,
            added_decisions=added_decisions,
            updated_decisions=changed_decisions,
            added_risks=added_risks,
            added_refs=added_refs,
            gaps=gaps,
        )

    def scan_state_gaps(
        self,
        state: MeetingState,
        *,
        current_ms: int | None = None,
        force_end_summary: bool = False,
    ) -> list[MeetingGap]:
        return scan_gaps(
            session_id=state.session_id,
            questions=state.open_questions,
            action_items=state.action_items,
            decisions=state.decisions,
            risks=state.risks,
            current_ms=current_ms,
            scheduled_end_ms=state.scheduled_end_ms,
            force_end_summary=force_end_summary,
        )


def _add_participant(participants: list[str], speaker: str) -> list[str]:
    if speaker in participants:
        return participants
    return [*participants, speaker]


def _append_unique(items: list[T], item: T, *, key: str) -> list[T]:
    value = getattr(item, key)
    if any(getattr(existing, key) == value for existing in items):
        return items
    return [*items, item]


def _has_id(items: list[object], value: str, key: str) -> bool:
    return any(getattr(item, key) == value for item in items)


def _update_open_decisions(existing: list[Decision], new_decision: Decision | None) -> tuple[list[Decision], list[Decision]]:
    if new_decision is None or new_decision.conclusion is None:
        return existing, []
    updated: list[Decision] = []
    changed: list[Decision] = []
    has_updated = False
    for decision in existing:
        if decision.conclusion is None and not has_updated:
            changed_decision = decision.model_copy(
                update={
                    "conclusion": new_decision.conclusion,
                    "metadata": {
                        **decision.metadata,
                        "conclusion_source_utterance_id": new_decision.source_utterance_id,
                    },
                }
            )
            updated.append(changed_decision)
            changed.append(changed_decision)
            has_updated = True
        else:
            updated.append(decision)
    return updated, changed
