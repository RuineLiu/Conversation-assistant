from typing import Any

from proactive_assistant.model_gateway import FakeModelClient, ModelGatewayError, ModelRequest
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptCandidateStatus, PromptOrchestrator
from proactive_assistant.prompting import (
    ContentGranularity,
    DisplayMode,
    DurationPolicy,
    PRDSurface,
    PromptGenerationService,
    RuleBasedPromptGenerationService,
)
from proactive_assistant.sessions import InMemorySessionStore, SessionConfig, SessionService, TranscriptSegmentInput


def make_snapshot(*texts: str, privacy_constraints: list[str] | None = None):  # type: ignore[no-untyped-def]
    service = SessionService(InMemorySessionStore())
    service.create_session(
        SessionConfig(
            title="Launch risk sync",
            pre_context="讨论项目风险、负责人和下一步行动。",
            privacy_constraints=privacy_constraints or [],
        ),
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


def make_prompt_service(response: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
    client = FakeModelClient(response or valid_prompt_response())
    service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test", max_output_tokens=256),
    )
    return service, client


def valid_prompt_response(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "should_prompt": True,
        "prompt_category": "summary_gap_check",
        "content_granularity": 2,
        "glasses_title": "负责人待确认",
        "glasses_text": "这个风险还没有明确 owner 和截止时间。",
        "app_detail_text": "会议中出现 owner/deadline gap，需要确认负责人、截止时间和下一步。",
        "source_refs": ["transcript:seg_0"],
        "confidence": 0.84,
        "privacy_level": "low",
        "privacy_risk": 0.08,
        "rationale": "检测到未确认的负责人和 deadline。",
        "safety_flags": [],
    }
    payload.update(overrides)
    return payload


def test_orchestrator_returns_empty_result_without_opportunity() -> None:
    prompt_service, client = make_prompt_service()
    snapshot = make_snapshot("今天项目进展正常，我们继续按计划推进。")

    result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)

    assert result.session_id == "session_001"
    assert result.opportunities == []
    assert result.candidates == []
    assert client.requests == []


def test_orchestrator_generates_candidate_from_gap_opportunity() -> None:
    prompt_service, client = make_prompt_service()
    snapshot = make_snapshot("这个问题谁负责，下周五 deadline 前能不能定？")

    result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)

    candidate = result.candidates[0]
    assert candidate.status == PromptCandidateStatus.GENERATED
    assert candidate.prompt_result is not None
    assert candidate.prompt_result.glasses_text == "这个风险还没有明确 owner 和截止时间。"
    assert candidate.prompt_request.prompt_category_candidate == "summary_gap_check"
    assert candidate.prompt_request.target_content_granularity == ContentGranularity.ONE_LINE_ANSWER
    assert candidate.prompt_request.prd_surface == PRDSurface.GLASSES_POPUP
    assert candidate.prompt_request.display_mode == DisplayMode.AUTO
    assert candidate.prompt_request.duration_policy == DurationPolicy.FIVE_SECONDS
    assert candidate.prompt_request.session_context["opportunity"]["captured_text"] == snapshot.recent_transcript.segments[0].text
    assert "这个问题谁负责" in client.requests[0].input_text


def test_orchestrator_routes_high_privacy_opportunity_to_app_prompt_tab() -> None:
    prompt_service, _client = make_prompt_service(
        valid_prompt_response(
            prompt_category="suggestion",
            glasses_title="建议稍后查看",
            glasses_text="涉及客户报价，建议在 APP 里查看更完整建议。",
            privacy_level="high",
            privacy_risk=0.78,
            safety_flags=["sensitive_business_context"],
        )
    )
    snapshot = make_snapshot("这个客户报价风险怎么跟老板说？", privacy_constraints=["avoid customer data"])

    result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)

    request = result.candidates[0].prompt_request
    assert request.prd_surface == PRDSurface.APP_PROMPT_TAB
    assert request.display_mode == DisplayMode.WRIST_TURN
    assert request.duration_policy == DurationPolicy.AUTO
    assert request.target_content_granularity == ContentGranularity.ONE_LINE_ANSWER
    assert request.session_context["opportunity"]["privacy_level"] == "high"


def test_orchestrator_routes_post_activity_to_summary_tab() -> None:
    prompt_service, _client = make_prompt_service()
    snapshot = make_snapshot("会议结束前我们总结一下，还有哪些 action item 没确认？")

    result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)

    request = result.candidates[0].prompt_request
    assert request.prd_surface == PRDSurface.APP_SUMMARY_TAB
    assert request.duration_policy == DurationPolicy.AUTO
    assert request.session_context["opportunity"]["candidate_timing_action"] == "after_activity"


def test_orchestrator_marks_no_action_model_output_as_suppressed() -> None:
    prompt_service, _client = make_prompt_service(
        {
            "should_prompt": False,
            "content_granularity": 0,
            "confidence": 0.31,
            "privacy_risk": 0.05,
            "rationale": "上下文不足，不主动打断。",
        }
    )
    snapshot = make_snapshot("腾讯是哪一年成立的？")

    result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)

    candidate = result.candidates[0]
    assert candidate.status == PromptCandidateStatus.SUPPRESSED
    assert candidate.prompt_result is not None
    assert candidate.prompt_result.should_prompt is False
    assert candidate.prompt_result.content_granularity == ContentGranularity.NO_ACTION


def test_orchestrator_isolates_invalid_model_output_as_failed_candidate() -> None:
    prompt_service, _client = make_prompt_service(
        {
            "should_prompt": True,
            "content_granularity": 2,
            "confidence": 0.7,
            "privacy_risk": 0.1,
        }
    )
    snapshot = make_snapshot("这个数据为什么和上周不一样？")

    result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)

    candidate = result.candidates[0]
    assert candidate.status == PromptCandidateStatus.GENERATION_FAILED
    assert candidate.prompt_result is None
    assert "ModelOutputValidationError" in candidate.reason


def test_orchestrator_uses_rule_fallback_when_model_generation_fails() -> None:
    def fail_request(_request: ModelRequest) -> dict[str, Any]:
        raise ModelGatewayError("model unavailable")

    client = FakeModelClient(fail_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    snapshot = make_snapshot("这个问题谁负责，下周五 deadline 前能不能定？")

    result = PromptOrchestrator(
        prompt_service=prompt_service,
        fallback_prompt_service=RuleBasedPromptGenerationService(),
    ).run(snapshot)

    candidate = result.candidates[0]
    assert candidate.status == PromptCandidateStatus.GENERATED
    assert candidate.prompt_result is not None
    assert candidate.prompt_result.model_usage is not None
    assert candidate.prompt_result.model_usage.provider == "rules"
    assert candidate.metadata["stage"] == "prompt_generation_fallback"
    assert "model unavailable" in candidate.metadata["fallback_reason"]


def test_orchestrator_limits_generated_candidates() -> None:
    def response_for_request(request: ModelRequest) -> dict[str, Any]:
        return valid_prompt_response(source_refs=["transcript:seg_0"])

    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    snapshot = make_snapshot(
        "这个数据为什么变化？",
        "腾讯是哪一年成立的？",
        "这个风险下一步怎么推进？",
    )

    result = PromptOrchestrator(prompt_service=prompt_service, max_candidates=1).run(snapshot)

    assert len(result.opportunities) == 1
    assert len(result.candidates) == 1
    assert len(client.requests) == 1


def test_orchestrator_enforces_glasses_length_caps_and_records_metadata() -> None:
    long_text = (
        "这个风险的负责人需要立刻确认，下周五之前必须给出明确的下一步动作，"
        "否则会影响发布节奏，建议同步上级和相关方一起决定。"
    )
    prompt_service, _client = make_prompt_service(
        valid_prompt_response(
            glasses_title="负责人待确认",
            glasses_text=long_text,
            app_detail_text="",
        )
    )
    snapshot = make_snapshot("这个问题谁负责，下周五 deadline 前能不能定？")

    result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)
    candidate = result.candidates[0]

    assert candidate.prompt_result is not None
    # glasses_text was truncated to ≤30 CJK chars
    visible = [
        ch
        for ch in candidate.prompt_result.glasses_text
        if not ch.isspace() and ch not in "，。！？；：、,.!?;:"
    ]
    assert len(visible) <= 30
    # original verbose copy preserved into app_detail_text
    assert candidate.prompt_result.app_detail_text == long_text
    # enforcer telemetry stored on candidate metadata
    assert "text_truncated" in candidate.metadata["enforcement_actions"]
    assert candidate.metadata["enforcement_enforced"] is True
    assert candidate.metadata["enforcement_metrics"]["text_limit"] == 30


def test_orchestrator_defers_second_glasses_show_to_app_within_cooldown() -> None:
    from proactive_assistant.orchestration import RateLimitConfig, RateLimiter
    from proactive_assistant.orchestration.rate_limiter import (
        InMemoryRateLimitHistory,
    )

    class _FakeClock:
        def __init__(self) -> None:
            self.t = 0

        def now_ms(self) -> int:
            return self.t

        def advance(self, ms: int) -> None:
            self.t += ms

    clock = _FakeClock()
    limiter = RateLimiter(
        config=RateLimitConfig(surface_cooldown_ms=20_000),
        history=InMemoryRateLimitHistory(),
        clock=clock,
    )

    def response_for_request(request: ModelRequest) -> dict[str, Any]:
        return valid_prompt_response(source_refs=["transcript:seg_0"])

    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    orchestrator = PromptOrchestrator(prompt_service=prompt_service, rate_limiter=limiter)

    snapshot1 = make_snapshot("这个问题谁负责，下周五 deadline 前能不能定？")
    first = orchestrator.run(snapshot1)
    first_candidate = first.candidates[0]

    assert first_candidate.status == PromptCandidateStatus.GENERATED
    assert first_candidate.prompt_request.prd_surface == PRDSurface.GLASSES_POPUP.value
    assert first_candidate.metadata["rate_limit_action"] == "allow"

    clock.advance(3_000)

    snapshot2 = make_snapshot("这个数据为什么变化？")
    second = orchestrator.run(snapshot2)
    second_candidate = second.candidates[0]

    assert second_candidate.status == PromptCandidateStatus.GENERATED
    assert second_candidate.prompt_request.prd_surface == PRDSurface.APP_PROMPT_TAB.value
    assert second_candidate.metadata["rate_limit_action"] == "defer_to_app"
    assert "surface_cooldown" in second_candidate.metadata["rate_limit_reasons"]
    assert second_candidate.metadata["rate_limit_new_surface"] == PRDSurface.APP_PROMPT_TAB.value


def test_orchestrator_drops_candidate_when_no_app_fallback_configured() -> None:
    from proactive_assistant.orchestration import RateLimitConfig, RateLimiter

    class _FakeClock:
        def __init__(self) -> None:
            self.t = 0

        def now_ms(self) -> int:
            return self.t

        def advance(self, ms: int) -> None:
            self.t += ms

    clock = _FakeClock()
    limiter = RateLimiter(
        config=RateLimitConfig(
            surface_cooldown_ms=20_000,
            drop_when_no_app_fallback=True,
        ),
        clock=clock,
    )

    def response_for_request(request: ModelRequest) -> dict[str, Any]:
        return valid_prompt_response(source_refs=["transcript:seg_0"])

    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    orchestrator = PromptOrchestrator(prompt_service=prompt_service, rate_limiter=limiter)

    orchestrator.run(make_snapshot("这个问题谁负责？"))
    clock.advance(3_000)
    result = orchestrator.run(make_snapshot("这个数据为什么变化？"))
    candidate = result.candidates[0]

    assert candidate.status == PromptCandidateStatus.SUPPRESSED
    assert candidate.metadata["rate_limit_action"] == "drop"
    assert candidate.prompt_result is not None
    assert candidate.prompt_result.should_prompt is False
    assert candidate.prompt_result.content_granularity == ContentGranularity.NO_ACTION.value


def test_orchestrator_post_activity_glasses_defers_immediately() -> None:
    from proactive_assistant.orchestration import RateLimiter

    def response_for_request(request: ModelRequest) -> dict[str, Any]:
        return valid_prompt_response(source_refs=["transcript:seg_0"])

    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    orchestrator = PromptOrchestrator(
        prompt_service=prompt_service,
        rate_limiter=RateLimiter(),
    )

    # build_snapshot defaults phase to in_activity. We construct an explicit
    # post-activity opportunity via the orchestrator.generate_candidate API.
    from proactive_assistant.detection import (
        CandidateTimingAction,
        DetectionRuleMatch,
        PromptOpportunity,
        PromptPriority,
    )
    from proactive_assistant.prompting import (
        ContentGranularity as CG,
        PrivacyLevel as PL,
        PromptCategory as PC,
    )
    from proactive_assistant.schemas.scenario import ActivityPhase as AP

    snapshot = make_snapshot("这个问题谁负责？")
    opportunity = PromptOpportunity(
        opportunity_id="opp_post_summary",
        session_id="session_001",
        trigger_segment_ids=["seg_0"],
        captured_text="生成会后总结。",
        prompt_category=PC.SUMMARY_GAP_CHECK,
        activity_phase=AP.POST_ACTIVITY,
        candidate_timing_action=CandidateTimingAction.AFTER_ACTIVITY,
        suggested_content_granularity=CG.CONCISE_BULLETS,
        priority=PromptPriority.P0,
        confidence=0.9,
        privacy_level=PL.LOW,
        privacy_risk=0.1,
        reason="post activity summary",
        rule_matches=[
            DetectionRuleMatch(rule_name="t", matched_terms=["s"], confidence_delta=0.0, reason="t")
        ],
    )
    candidate = orchestrator.generate_candidate(snapshot, opportunity)

    # AFTER_ACTIVITY timing routes to app_summary_tab by default, which is
    # not a glasses surface — so the limiter passes through with
    # NOT_GLASSES_SURFACE rather than POST_ACTIVITY_GLASSES.
    assert candidate.metadata["rate_limit_action"] == "allow"
    assert candidate.prompt_request.prd_surface == PRDSurface.APP_SUMMARY_TAB.value
    assert "not_glasses_surface" in candidate.metadata["rate_limit_reasons"]


def test_orchestrator_fast_paths_pre_generated_explanation_without_second_llm_call() -> None:
    """When the detector pre-generates an explanation for an unknown term,
    the orchestrator should build the PromptCandidate directly and skip the
    PromptGenerationService entirely. This is the hot path for unknown-term
    explanations: ≤1 LLM call per detection cycle.
    """

    from proactive_assistant.detection import (
        CandidateTimingAction,
        DetectionRuleMatch,
        PromptOpportunity,
        PromptPriority,
    )
    from proactive_assistant.prompting import (
        ContentGranularity as CG,
        PromptCategory as PC,
        PrivacyLevel as PL,
    )
    from proactive_assistant.schemas.scenario import ActivityPhase as AP

    prompt_service, client = make_prompt_service()
    orchestrator = PromptOrchestrator(prompt_service=prompt_service)

    snapshot = make_snapshot("我们这季度 GMV 增长了 20%。")
    opportunity = PromptOpportunity(
        opportunity_id="opp_gmv",
        session_id="session_001",
        trigger_segment_ids=["seg_0"],
        captured_text="GMV",
        prompt_category=PC.CONCEPT_EXPLANATION,
        activity_phase=AP.IN_ACTIVITY,
        candidate_timing_action=CandidateTimingAction.DURING_ACTIVITY,
        suggested_content_granularity=CG.ONE_LINE_ANSWER,
        priority=PromptPriority.P1,
        confidence=0.85,
        privacy_level=PL.LOW,
        privacy_risk=0.05,
        reason="LLM detected unfamiliar term: GMV",
        rule_matches=[
            DetectionRuleMatch(
                rule_name="llm_unknown_term_detector",
                matched_terms=["GMV"],
                confidence_delta=0.0,
                reason="electronic commerce term",
            )
        ],
        metadata={
            "pre_generated_explanation": "商品交易总额，电商核心指标。",
            "unknown_term": "GMV",
            "unknown_term_type": "acronym",
            "unknown_term_rationale": "test",
            "unknown_term_candidate_id": "unkterm_abc",
            "detection_source": "llm_unknown_term_detector",
        },
    )

    candidate = orchestrator.generate_candidate(snapshot, opportunity)

    # No second LLM call was issued for prompt generation.
    assert client.requests == []
    assert candidate.status == PromptCandidateStatus.GENERATED
    assert candidate.prompt_result is not None
    assert candidate.prompt_result.glasses_title == "GMV"
    assert candidate.prompt_result.glasses_text == "商品交易总额，电商核心指标。"
    assert candidate.prompt_result.app_detail_text == "商品交易总额，电商核心指标。"
    assert candidate.prompt_result.prompt_category == "concept_explanation"
    assert candidate.prompt_result.source_refs == ["transcript:seg_0"]
    assert candidate.prompt_result.model_usage is not None
    assert candidate.prompt_result.model_usage.provider == "unknown_term_detector"
    assert candidate.metadata["stage"] == "prompt_generation_fast_path"
    assert candidate.metadata["fast_path_source"] == "llm_unknown_term_detector"


def test_orchestrator_uses_generation_model_override_when_set() -> None:
    """A3: when generation_model is set, prompt generation requests use it
    instead of the gateway default."""
    response_for_request = lambda request: valid_prompt_response(source_refs=["transcript:seg_0"])  # noqa: E731
    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-default"),
    )
    orchestrator = PromptOrchestrator(
        prompt_service=prompt_service,
        generation_model="gpt-fast-override",
    )
    snapshot = make_snapshot("这个问题谁负责，下周五 deadline 前能不能定？")

    orchestrator.run(snapshot)

    assert client.requests, "expected a generation request"
    assert client.requests[0].model == "gpt-fast-override"


def test_orchestrator_uses_default_model_when_no_override() -> None:
    response_for_request = lambda request: valid_prompt_response(source_refs=["transcript:seg_0"])  # noqa: E731
    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-default"),
    )
    orchestrator = PromptOrchestrator(prompt_service=prompt_service)
    snapshot = make_snapshot("这个问题谁负责，下周五 deadline 前能不能定？")

    orchestrator.run(snapshot)

    assert client.requests
    assert client.requests[0].model == "gpt-default"


def test_orchestrator_routes_public_knowledge_question_to_public_model() -> None:
    response_for_request = lambda request: valid_prompt_response(  # noqa: E731
        prompt_category="question_answer",
        glasses_title="导演",
        glasses_text="《桃色公寓》导演是比利·怀尔德。",
        source_refs=["transcript:seg_0"],
    )
    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-default"),
    )
    orchestrator = PromptOrchestrator(
        prompt_service=prompt_service,
        generation_model="gpt-fast",
        public_knowledge_model="gpt-factual",
    )
    snapshot = make_snapshot("桃色公寓的导演是谁？")

    orchestrator.run(snapshot)

    assert client.requests
    assert client.requests[0].model == "gpt-factual"


def test_orchestrator_keeps_business_gap_on_generation_model() -> None:
    response_for_request = lambda request: valid_prompt_response(source_refs=["transcript:seg_0"])  # noqa: E731
    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-default"),
    )
    orchestrator = PromptOrchestrator(
        prompt_service=prompt_service,
        generation_model="gpt-fast",
        public_knowledge_model="gpt-factual",
    )
    snapshot = make_snapshot("这个问题谁负责，下周五 deadline 前能不能定？")

    orchestrator.run(snapshot)

    assert client.requests
    assert client.requests[0].model == "gpt-fast"
