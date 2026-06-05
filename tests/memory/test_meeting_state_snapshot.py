from __future__ import annotations

from datetime import UTC, datetime

from proactive_assistant.meeting_state import ActionItem, ActionItemStatus, Decision, MeetingState, Risk
from proactive_assistant.memory import memory_candidates_from_meeting_state
from proactive_assistant.runtime import MemoryCandidateType, MemoryWritePolicy


def test_meeting_state_snapshot_exports_action_decision_and_risk_candidates() -> None:
    state = MeetingState(
        session_id="session_001",
        org_id="org_001",
        subject_user_id="user_001",
        started_at=datetime(2026, 6, 5, tzinfo=UTC),
        action_items=[
            ActionItem(
                action_item_id="action_001",
                desc="张三负责客户报价确认，下周五截止。",
                source_utterance_id="seg_0",
                source_ts_ms=0,
                owner="张三",
                deadline="下周五",
                next_step="确认客户报价",
                status=ActionItemStatus.ASSIGNED,
                evidence="张三负责客户报价确认，下周五截止。",
            )
        ],
        decisions=[
            Decision(
                decision_id="decision_001",
                topic="报价口径",
                source_utterance_id="seg_1",
                ts_ms=1000,
                conclusion="就按低风险口径同步。",
                evidence="报价口径就按低风险口径同步。",
            )
        ],
        risks=[
            Risk(
                risk_id="risk_001",
                desc="客户报价存在审批延期风险。",
                source_utterance_id="seg_2",
                ts_ms=2000,
                closed=False,
                evidence="客户报价存在审批延期风险。",
            )
        ],
    )

    candidates = memory_candidates_from_meeting_state(state)

    assert [candidate.candidate_type for candidate in candidates] == [
        MemoryCandidateType.ACTION_ITEM,
        MemoryCandidateType.MEETING_FACT,
        MemoryCandidateType.MEETING_FACT,
    ]
    action = candidates[0]
    assert action.write_policy == MemoryWritePolicy.ELIGIBLE
    assert action.metadata["meeting_state_object_type"] == "action_item"
    assert action.metadata["meeting_state_object_id"] == "action_001"
    assert action.metadata["owner"] == "张三"
    assert action.metadata["deadline"] == "下周五"
    assert action.metadata["provenance"] == ["transcript:seg_0"]
    assert action.metadata["memory_snapshot_source"] == "meeting_state_snapshot_v1"
    assert candidates[1].metadata["status"] == "decided"
    assert candidates[2].metadata["status"] == "open"
