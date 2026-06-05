from proactive_assistant.detection import CandidateTimingAction, opportunities_from_meeting_gaps
from proactive_assistant.meeting_state import MeetingGap, MeetingGapPriority, MeetingGapType
from proactive_assistant.prompting import ContentGranularity, PrivacyLevel, PromptCategory
from proactive_assistant.schemas.scenario import ActivityPhase


def gap(
    gap_type: MeetingGapType,
    *,
    priority: MeetingGapPriority = MeetingGapPriority.P1,
    text: str = "这个问题谁负责？",
    source_utterance_ids: list[str] | None = None,
) -> MeetingGap:
    return MeetingGap(
        gap_id=f"gap_{gap_type.value}",
        session_id="session_001",
        gap_type=gap_type,
        priority=priority,
        object_type="action_item",
        object_id="object_001",
        text=text,
        reason="test gap",
        source_utterance_ids=["seg_0"] if source_utterance_ids is None else source_utterance_ids,
        first_seen_ms=0,
    )


def test_adapter_converts_action_gap_to_summary_gap_opportunity() -> None:
    opportunities = opportunities_from_meeting_gaps(
        "session_001",
        [gap(MeetingGapType.ACTION_MISSING_OWNER, priority=MeetingGapPriority.P0)],
        fallback_segment_id="seg_fallback",
    )

    opportunity = opportunities[0]
    assert opportunity.prompt_category == PromptCategory.SUMMARY_GAP_CHECK
    assert opportunity.activity_phase == ActivityPhase.IN_ACTIVITY
    assert opportunity.candidate_timing_action == CandidateTimingAction.DURING_ACTIVITY
    assert opportunity.suggested_content_granularity == ContentGranularity.ONE_LINE_ANSWER
    assert opportunity.priority == "P0"
    assert opportunity.metadata["source"] == "meeting_state"
    assert opportunity.metadata["gap_type"] == "action_missing_owner"


def test_adapter_skips_raw_unanswered_question_to_avoid_duplicate_realtime_detection() -> None:
    opportunities = opportunities_from_meeting_gaps(
        "session_001",
        [gap(MeetingGapType.UNANSWERED_QUESTION, text="腾讯是哪一年成立的？")],
        fallback_segment_id="seg_fallback",
    )

    assert opportunities == []


def test_adapter_prefers_single_structured_gap_per_source_segment() -> None:
    opportunities = opportunities_from_meeting_gaps(
        "session_001",
        [
            gap(MeetingGapType.ACTION_MISSING_NEXT_STEP, priority=MeetingGapPriority.P1),
            gap(MeetingGapType.ACTION_MISSING_OWNER, priority=MeetingGapPriority.P0),
        ],
        fallback_segment_id="seg_fallback",
    )

    assert len(opportunities) == 1
    assert opportunities[0].metadata["gap_type"] == "action_missing_owner"


def test_adapter_routes_end_summary_gap_after_activity_with_fallback_segment() -> None:
    opportunities = opportunities_from_meeting_gaps(
        "session_001",
        [
            gap(
                MeetingGapType.END_SUMMARY_NEEDED,
                priority=MeetingGapPriority.P0,
                text="Meeting has unresolved items.",
                source_utterance_ids=[],
            )
        ],
        fallback_segment_id="seg_closing",
    )

    opportunity = opportunities[0]
    assert opportunity.trigger_segment_ids == ["seg_closing"]
    assert opportunity.activity_phase == ActivityPhase.POST_ACTIVITY
    assert opportunity.candidate_timing_action == CandidateTimingAction.AFTER_ACTIVITY
    assert opportunity.suggested_content_granularity == ContentGranularity.CONCISE_BULLETS


def test_adapter_applies_privacy_constraints_to_state_opportunity() -> None:
    opportunities = opportunities_from_meeting_gaps(
        "session_001",
        [
            gap(
                MeetingGapType.OPEN_RISK,
                priority=MeetingGapPriority.P2,
                text="这个客户报价风险还没有闭环。",
            )
        ],
        fallback_segment_id="seg_0",
        privacy_constraints=["avoid customer data"],
    )

    opportunity = opportunities[0]
    assert opportunity.privacy_level == PrivacyLevel.HIGH
    assert opportunity.privacy_risk >= 0.7
    assert "sensitive_business_context" in opportunity.safety_flags
