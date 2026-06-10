from hashlib import sha1

from proactive_assistant.detection.contracts import (
    CandidateTimingAction,
    DetectionRuleMatch,
    PromptOpportunity,
    PromptPriority,
)
from proactive_assistant.detection.rules import PrivacyAssessment, assess_privacy
from proactive_assistant.meeting_state import MeetingGap, MeetingGapPriority, MeetingGapType
from proactive_assistant.prompting import ContentGranularity, PrivacyLevel, PromptCategory
from proactive_assistant.schemas.scenario import ActivityPhase


def opportunities_from_meeting_gaps(
    session_id: str,
    gaps: list[MeetingGap],
    *,
    fallback_segment_id: str,
    privacy_constraints: list[str] | None = None,
    speaker_by_segment_id: dict[str, str] | None = None,
) -> list[PromptOpportunity]:
    promptable_gaps = _select_promptable_gaps(gaps, fallback_segment_id=fallback_segment_id)
    speaker_map = speaker_by_segment_id or {}
    return [
        _opportunity_from_gap(
            session_id,
            gap,
            fallback_segment_id=fallback_segment_id,
            privacy=assess_privacy(gap.text, privacy_constraints or []),
            speaker_by_segment_id=speaker_map,
        )
        for gap in promptable_gaps
    ]


def _opportunity_from_gap(
    session_id: str,
    gap: MeetingGap,
    *,
    fallback_segment_id: str,
    privacy: PrivacyAssessment,
    speaker_by_segment_id: dict[str, str],
) -> PromptOpportunity:
    trigger_segment_ids = gap.source_utterance_ids or [fallback_segment_id]
    category, phase, timing, granularity = _policy_for_gap(gap)
    priority = _priority_for_gap(gap)
    # Inherit target_speaker_id when all triggers come from the same speaker;
    # otherwise leave empty to signal a multi-speaker structural gap.
    speakers_seen = {speaker_by_segment_id.get(seg_id, "") for seg_id in trigger_segment_ids}
    speakers_seen.discard("")
    target_speaker_id = next(iter(speakers_seen)) if len(speakers_seen) == 1 else ""
    return PromptOpportunity(
        opportunity_id=_opportunity_id(session_id, gap.gap_id),
        session_id=session_id,
        trigger_segment_ids=trigger_segment_ids,
        captured_text=gap.text,
        prompt_category=category,
        activity_phase=phase,
        candidate_timing_action=timing,
        suggested_content_granularity=granularity,
        priority=priority,
        confidence=_confidence_for_gap(gap),
        privacy_level=privacy.privacy_level,
        privacy_risk=privacy.privacy_risk,
        reason=f"Meeting state gap detected: {gap.reason}",
        safety_flags=list(privacy.safety_flags),
        rule_matches=[
            DetectionRuleMatch(
                rule_name="meeting_state_gap",
                matched_terms=[str(gap.gap_type)],
                confidence_delta=0.0,
                reason=gap.reason,
            )
        ],
        target_speaker_id=target_speaker_id,
        metadata={
            "source": "meeting_state",
            "gap_id": gap.gap_id,
            "gap_type": str(gap.gap_type),
            "object_type": gap.object_type,
            "object_id": gap.object_id,
            "gap_priority": str(gap.priority),
            "gap_metadata": gap.metadata,
            "privacy_terms": list(privacy.matched_terms),
        },
    )


def _select_promptable_gaps(gaps: list[MeetingGap], *, fallback_segment_id: str) -> list[MeetingGap]:
    end_summary_gaps = [gap for gap in gaps if MeetingGapType(gap.gap_type) == MeetingGapType.END_SUMMARY_NEEDED]
    if end_summary_gaps:
        return sorted(end_summary_gaps, key=_gap_sort_key)

    best_by_segment: dict[str, MeetingGap] = {}
    for gap in gaps:
        if not _should_emit_gap(gap):
            continue
        source_segment_id = (gap.source_utterance_ids or [fallback_segment_id])[0]
        current = best_by_segment.get(source_segment_id)
        if current is None or _is_better_gap(gap, current):
            best_by_segment[source_segment_id] = gap
    return sorted(best_by_segment.values(), key=_gap_sort_key)


def _should_emit_gap(gap: MeetingGap) -> bool:
    gap_type = MeetingGapType(gap.gap_type)
    return gap_type not in {MeetingGapType.UNANSWERED_QUESTION}


def _is_better_gap(candidate: MeetingGap, current: MeetingGap) -> bool:
    return _gap_sort_key(candidate) < _gap_sort_key(current)


def _gap_sort_key(gap: MeetingGap) -> tuple[int, int, str]:
    priority_rank = {
        MeetingGapPriority.P0: 0,
        MeetingGapPriority.P1: 1,
        MeetingGapPriority.P2: 2,
    }[MeetingGapPriority(gap.priority)]
    type_rank = {
        MeetingGapType.END_SUMMARY_NEEDED: 0,
        MeetingGapType.ACTION_MISSING_OWNER: 1,
        MeetingGapType.ACTION_MISSING_DEADLINE: 2,
        MeetingGapType.ACTION_MISSING_NEXT_STEP: 3,
        MeetingGapType.DECISION_MISSING_CONCLUSION: 4,
        MeetingGapType.OPEN_RISK: 5,
        MeetingGapType.UNANSWERED_QUESTION: 6,
    }[MeetingGapType(gap.gap_type)]
    return (priority_rank, type_rank, gap.gap_id)


def _policy_for_gap(
    gap: MeetingGap,
) -> tuple[PromptCategory, ActivityPhase, CandidateTimingAction, ContentGranularity]:
    gap_type = MeetingGapType(gap.gap_type)
    if gap_type == MeetingGapType.UNANSWERED_QUESTION:
        return (
            PromptCategory.QUESTION_ANSWER,
            ActivityPhase.IN_ACTIVITY,
            CandidateTimingAction.DURING_ACTIVITY,
            ContentGranularity.ONE_LINE_ANSWER,
        )
    if gap_type == MeetingGapType.END_SUMMARY_NEEDED:
        return (
            PromptCategory.SUMMARY_GAP_CHECK,
            ActivityPhase.POST_ACTIVITY,
            CandidateTimingAction.AFTER_ACTIVITY,
            ContentGranularity.CONCISE_BULLETS,
        )
    return (
        PromptCategory.SUMMARY_GAP_CHECK,
        ActivityPhase.IN_ACTIVITY,
        CandidateTimingAction.DURING_ACTIVITY,
        ContentGranularity.ONE_LINE_ANSWER,
    )


def _priority_for_gap(gap: MeetingGap) -> PromptPriority:
    priority = MeetingGapPriority(gap.priority)
    return {
        MeetingGapPriority.P0: PromptPriority.P0,
        MeetingGapPriority.P1: PromptPriority.P1,
        MeetingGapPriority.P2: PromptPriority.P2,
    }[priority]


def _confidence_for_gap(gap: MeetingGap) -> float:
    gap_type = MeetingGapType(gap.gap_type)
    if gap_type == MeetingGapType.END_SUMMARY_NEEDED:
        return 0.9
    if MeetingGapPriority(gap.priority) == MeetingGapPriority.P0:
        return 0.86
    if MeetingGapPriority(gap.priority) == MeetingGapPriority.P1:
        return 0.78
    return 0.66


def _opportunity_id(session_id: str, gap_id: str) -> str:
    digest = sha1(f"{session_id}:{gap_id}:meeting_state".encode("utf-8")).hexdigest()[:12]
    return f"opp_state_{digest}"
