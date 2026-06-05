from proactive_assistant.meeting_state import MeetingGapType, MeetingStateTracker, MentionedRefType, QuestionType
from proactive_assistant.sessions import TranscriptSegmentRecord


def segment(text: str, *, speaker: str = "Bao", index: int = 0) -> TranscriptSegmentRecord:
    return TranscriptSegmentRecord(
        session_id="session_001",
        segment_id=f"seg_{index}",
        speaker=speaker,
        start_ms=index * 1000,
        end_ms=index * 1000 + 800,
        text=text,
        asr_confidence=0.94,
    )


def gap_types(result):  # type: ignore[no-untyped-def]
    return {gap.gap_type for gap in result.gaps}


def test_tracker_records_question_and_marks_it_answered() -> None:
    tracker = MeetingStateTracker()
    state = tracker.create_state("session_001")

    first = tracker.update_from_segment(state, segment("这个数据为什么和上周不一样？", index=0))

    assert len(first.added_questions) == 1
    assert first.added_questions[0].question_type == QuestionType.REASON
    assert MeetingGapType.UNANSWERED_QUESTION in gap_types(first)
    assert first.state.participants == ["Bao"]

    second = tracker.update_from_segment(first.state, segment("主要是因为统计口径换了。", speaker="Alex", index=1))

    assert len(second.answered_questions) == 1
    assert second.state.open_questions[0].answered is True
    assert second.state.open_questions[0].answer_utterance_id == "seg_1"
    assert MeetingGapType.UNANSWERED_QUESTION not in gap_types(second)
    assert second.state.participants == ["Bao", "Alex"]


def test_tracker_extracts_action_item_and_scans_missing_owner_gap() -> None:
    tracker = MeetingStateTracker()
    state = tracker.create_state("session_001")

    result = tracker.update_from_segment(state, segment("这个问题谁负责，下周五 deadline 前能不能定？", index=0))

    assert len(result.added_action_items) == 1
    item = result.added_action_items[0]
    assert item.owner is None
    assert item.deadline == "下周五"
    assert MeetingGapType.ACTION_MISSING_OWNER in gap_types(result)
    assert MeetingGapType.ACTION_MISSING_DEADLINE not in gap_types(result)


def test_tracker_does_not_emit_action_gap_when_owner_deadline_and_next_step_exist() -> None:
    tracker = MeetingStateTracker()
    state = tracker.create_state("session_001")

    result = tracker.update_from_segment(state, segment("请Alex负责推进这个事项，周五前同步下一步。", index=0))

    item = result.added_action_items[0]
    assert item.owner == "Alex"
    assert item.deadline == "周五"
    assert item.next_step is not None
    assert MeetingGapType.ACTION_MISSING_OWNER not in gap_types(result)
    assert MeetingGapType.ACTION_MISSING_DEADLINE not in gap_types(result)
    assert MeetingGapType.ACTION_MISSING_NEXT_STEP not in gap_types(result)


def test_tracker_emits_end_summary_gap_when_meeting_closes_with_open_items() -> None:
    tracker = MeetingStateTracker()
    state = tracker.create_state("session_001")
    first = tracker.update_from_segment(state, segment("这个问题谁负责，下周五 deadline 前能不能定？", index=0))

    closing = tracker.update_from_segment(first.state, segment("会议结束前我们总结一下，还有哪些 action item 没确认？", index=1))

    assert MeetingGapType.END_SUMMARY_NEEDED in gap_types(closing)
    summary_gap = [gap for gap in closing.gaps if gap.gap_type == MeetingGapType.END_SUMMARY_NEEDED][0]
    assert summary_gap.priority == "P0"
    assert summary_gap.metadata["tracked_gap_count"] >= 1


def test_tracker_extracts_history_deadline_and_commitment_refs() -> None:
    tracker = MeetingStateTracker()
    state = tracker.create_state("session_001")

    result = tracker.update_from_segment(state, segment("上次谁说要跟进这个 deadline？", index=0))

    ref_types = {ref.ref_type for ref in result.added_refs}
    assert MentionedRefType.PREVIOUS_MEETING in ref_types
    assert MentionedRefType.DEADLINE in ref_types
    assert MentionedRefType.COMMITMENT in ref_types


def test_tracker_updates_open_decision_when_conclusion_arrives() -> None:
    tracker = MeetingStateTracker()
    state = tracker.create_state("session_001")
    first = tracker.update_from_segment(state, segment("这个方案需要一个结论。", index=0))

    assert len(first.added_decisions) == 1
    assert first.added_decisions[0].conclusion is None
    assert MeetingGapType.DECISION_MISSING_CONCLUSION in gap_types(first)

    second = tracker.update_from_segment(first.state, segment("结论是就按A方案推进。", index=1))

    assert len(second.updated_decisions) == 1
    assert second.state.decisions[0].conclusion == "结论是就按A方案推进。"
    assert len(second.state.decisions) == 1
    assert MeetingGapType.DECISION_MISSING_CONCLUSION not in gap_types(second)


def test_tracker_emits_scheduled_end_summary_when_near_end() -> None:
    tracker = MeetingStateTracker()
    state = tracker.create_state("session_001", scheduled_end_ms=10_000)
    first = tracker.update_from_segment(state, segment("这个问题谁负责？", index=8))

    assert MeetingGapType.END_SUMMARY_NEEDED in gap_types(first)
