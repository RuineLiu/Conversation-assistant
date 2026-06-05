from proactive_assistant.detection import CandidateTimingAction, PromptOpportunityDetector, PromptPriority
from proactive_assistant.prompting import ContentGranularity, PrivacyLevel, PromptCategory
from proactive_assistant.schemas.scenario import ActivityPhase
from proactive_assistant.sessions import InMemorySessionStore, SessionConfig, SessionService, TranscriptSegmentInput


def make_snapshot(*texts: str, privacy_constraints: list[str] | None = None):  # type: ignore[no-untyped-def]
    service = SessionService(InMemorySessionStore())
    service.create_session(
        SessionConfig(privacy_constraints=privacy_constraints or []),
        session_id="session_001",
    )
    for index, text in enumerate(texts):
        service.append_transcript(
            "session_001",
            TranscriptSegmentInput(
                speaker="Bao",
                start_ms=index * 1000,
                end_ms=index * 1000 + 800,
                text=text,
                asr_confidence=0.93,
            ),
            segment_id=f"seg_{index}",
        )
    return service.get_context_snapshot("session_001")


def test_detector_returns_no_opportunity_for_plain_statement() -> None:
    snapshot = make_snapshot("今天项目进展正常，我们继续按计划推进。")

    result = PromptOpportunityDetector().detect(snapshot)

    assert result.opportunities == []
    assert result.inspected_segment_ids == ["seg_0"]


def test_question_rule_detects_general_question() -> None:
    snapshot = make_snapshot("这个数据为什么和上周不一样？")

    result = PromptOpportunityDetector().detect(snapshot)

    opportunity = result.opportunities[0]
    assert opportunity.prompt_category == PromptCategory.QUESTION_ANSWER
    assert opportunity.activity_phase == ActivityPhase.IN_ACTIVITY
    assert opportunity.candidate_timing_action == CandidateTimingAction.DURING_ACTIVITY
    assert opportunity.suggested_content_granularity == ContentGranularity.ONE_LINE_ANSWER
    assert opportunity.priority == PromptPriority.P1


def test_gap_rule_overrides_question_for_owner_deadline() -> None:
    snapshot = make_snapshot("这个问题谁负责，下周五 deadline 前能不能定？")

    result = PromptOpportunityDetector().detect(snapshot)
    opportunity = result.opportunities[0]
    assert opportunity.prompt_category == PromptCategory.SUMMARY_GAP_CHECK
    assert opportunity.priority == PromptPriority.P0
    assert opportunity.rule_matches[0].rule_name == "gap_check_rule"


def test_fact_recall_rule_detects_person_or_fact_need() -> None:
    snapshot = make_snapshot("腾讯是哪一年成立的？")

    opportunity = PromptOpportunityDetector().detect(snapshot).opportunities[0]
    assert opportunity.prompt_category == PromptCategory.PERSON_OR_FACT
    assert opportunity.suggested_content_granularity == ContentGranularity.ONE_LINE_ANSWER


def test_suggestion_rule_detects_next_step_request() -> None:
    snapshot = make_snapshot("这个风险我们下一步应该怎么推进？")

    opportunity = PromptOpportunityDetector().detect(snapshot).opportunities[0]
    assert opportunity.prompt_category == PromptCategory.SUGGESTION
    assert opportunity.suggested_content_granularity == ContentGranularity.CONCISE_BULLETS


def test_high_privacy_downgrades_priority_and_granularity() -> None:
    snapshot = make_snapshot("这个客户报价风险怎么跟老板说？", privacy_constraints=["avoid customer data"])

    opportunity = PromptOpportunityDetector().detect(snapshot).opportunities[0]
    assert opportunity.prompt_category == PromptCategory.SUGGESTION
    assert opportunity.priority == PromptPriority.P2
    assert opportunity.suggested_content_granularity == ContentGranularity.ONE_LINE_ANSWER
    assert opportunity.privacy_level == PrivacyLevel.HIGH
    assert opportunity.privacy_risk >= 0.7
    assert "sensitive_business_context" in opportunity.safety_flags


def test_end_signal_sets_post_activity_gap_check() -> None:
    snapshot = make_snapshot("会议结束前我们总结一下，还有哪些 action item 没确认？")

    opportunity = PromptOpportunityDetector().detect(snapshot).opportunities[0]
    assert opportunity.prompt_category == PromptCategory.SUMMARY_GAP_CHECK
    assert opportunity.activity_phase == ActivityPhase.POST_ACTIVITY
    assert opportunity.candidate_timing_action == CandidateTimingAction.AFTER_ACTIVITY
    assert opportunity.priority == PromptPriority.P0


def test_detector_limits_max_opportunities() -> None:
    snapshot = make_snapshot(
        "这个数据为什么变化？",
        "腾讯是哪一年成立的？",
        "这个风险下一步怎么推进？",
    )

    result = PromptOpportunityDetector(max_opportunities=2).detect(snapshot)

    assert len(result.opportunities) == 2
