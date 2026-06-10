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


def test_concept_rule_detects_unfamiliar_term_explanation_need() -> None:
    snapshot = make_snapshot("刚才说的 entropy 是什么意思？我没有反应过来。")

    opportunity = PromptOpportunityDetector().detect(snapshot).opportunities[0]

    assert opportunity.prompt_category == PromptCategory.CONCEPT_EXPLANATION
    assert opportunity.suggested_content_granularity == ContentGranularity.ONE_LINE_ANSWER
    assert opportunity.rule_matches[0].rule_name == "concept_explanation_rule"


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


def test_detector_emits_concept_explanation_opportunity_from_llm_detector() -> None:
    """End-to-end: PromptOpportunityDetector + UnknownTermDetector together
    should produce a CONCEPT_EXPLANATION opportunity with pre_generated
    explanation in metadata, ready for the orchestrator's fast path.
    """

    from proactive_assistant.detection import UnknownTermDetector
    from proactive_assistant.detection.service import PromptOpportunityDetector
    from proactive_assistant.model_gateway import FakeModelClient
    from proactive_assistant.model_gateway.settings import ModelGatewaySettings
    from proactive_assistant.prompting import PromptCategory
    from proactive_assistant.sessions import (
        InMemorySessionStore,
        SessionConfig,
        SessionService,
        TranscriptSegmentInput,
    )

    response = {
        "candidates": [
            {
                "term": "GMV",
                "term_type": "acronym",
                "explanation": "商品交易总额，电商核心指标。",
                "confidence": 0.86,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_0",
                "rationale": "电商术语",
            }
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    unknown_detector = UnknownTermDetector(
        model_client=FakeModelClient(response),
        settings=ModelGatewaySettings(default_model="gpt-test", max_output_tokens=512),
    )
    detector = PromptOpportunityDetector(
        max_opportunities=5,
        unknown_term_detector=unknown_detector,
        vocabulary_service=None,
    )

    session_service = SessionService(InMemorySessionStore())
    session_service.create_session(
        SessionConfig(title="GMV demo", pre_context="电商业务讨论。"),
        session_id="session_001",
    )
    session_service.append_transcript(
        "session_001",
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=1000,
            text="我们这季度 GMV 增长了 20%。",
            asr_confidence=0.92,
        ),
        segment_id="seg_0",
    )
    snapshot = session_service.get_context_snapshot("session_001")

    result = detector.detect(snapshot)

    concept_opps = [
        o for o in result.opportunities
        if str(o.prompt_category) == PromptCategory.CONCEPT_EXPLANATION.value
    ]
    assert len(concept_opps) == 1
    opp = concept_opps[0]
    assert opp.captured_text == "GMV"
    assert opp.metadata["pre_generated_explanation"] == "商品交易总额，电商核心指标。"
    assert opp.metadata["unknown_term"] == "GMV"
    assert opp.metadata["unknown_term_type"] == "acronym"
    assert opp.metadata["detection_source"] == "llm_unknown_term_detector"


def test_detector_emits_gap_opportunity_from_llm_opportunity_detector() -> None:
    """End-to-end: the keyword rules miss "ddl" but the LLM opportunity
    detector flags action_missing_deadline, and PromptOpportunityDetector
    adapts it into a summary_gap_check PromptOpportunity with structured
    metadata."""

    from proactive_assistant.detection import OpportunityDetector
    from proactive_assistant.detection.service import PromptOpportunityDetector
    from proactive_assistant.model_gateway import FakeModelClient
    from proactive_assistant.model_gateway.settings import ModelGatewaySettings
    from proactive_assistant.prompting import PromptCategory
    from proactive_assistant.sessions import (
        InMemorySessionStore,
        SessionConfig,
        SessionService,
        TranscriptSegmentInput,
    )

    response = {
        "opportunities": [
            {
                "prompt_category": "summary_gap_check",
                "gap_type": "action_missing_deadline",
                "captured_text": "小张把项目文档给我，ddl还没定。",
                "source_segment_id": "seg_0",
                "owner": "小张",
                "deadline": "",
                "entity": "项目文档",
                "priority": "P1",
                "confidence": 0.85,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "rationale": "任务缺截止时间(ddl)。",
            }
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    opportunity_detector = OpportunityDetector(
        model_client=FakeModelClient(response),
        settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
    )
    detector = PromptOpportunityDetector(
        max_opportunities=5,
        opportunity_detector=opportunity_detector,
    )

    session_service = SessionService(InMemorySessionStore())
    session_service.create_session(SessionConfig(title="ddl demo"), session_id="session_001")
    session_service.append_transcript(
        "session_001",
        TranscriptSegmentInput(
            speaker="小张",
            start_ms=0,
            end_ms=1000,
            text="小张把项目文档给我吧，ddl你定一下。",
            asr_confidence=0.92,
        ),
        segment_id="seg_0",
    )
    snapshot = session_service.get_context_snapshot("session_001")

    result = detector.detect(snapshot)

    gap_opps = [
        o for o in result.opportunities
        if str(o.prompt_category) == PromptCategory.SUMMARY_GAP_CHECK.value
    ]
    assert len(gap_opps) == 1
    opp = gap_opps[0]
    assert opp.metadata["detection_source"] == "llm_opportunity_detector"
    assert opp.metadata["gap_type"] == "action_missing_deadline"
    assert opp.metadata["owner"] == "小张"
    assert opp.metadata["entity"] == "项目文档"
    assert opp.target_speaker_id == "小张"
