from typing import Any

from proactive_assistant.model_gateway import FakeModelClient, ModelRequest
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptCandidateStatus, PromptOrchestrator
from proactive_assistant.prompting import (
    ContentGranularity,
    DisplayMode,
    DurationPolicy,
    PRDSurface,
    PromptGenerationService,
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
