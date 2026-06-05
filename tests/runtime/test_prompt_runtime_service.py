import pytest

from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.prompting import PromptGenerationService
from proactive_assistant.runtime import (
    DecisionRecordNotFoundError,
    FeedbackPolarity,
    FeedbackSignalSource,
    FeedbackSignalType,
    MemoryCandidateType,
    MemoryWritePolicy,
    PromptDecisionDisplayStatus,
    PromptRuntimeService,
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


def valid_prompt_response(**overrides):  # type: ignore[no-untyped-def]
    payload = {
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


def make_candidate(text: str, response: dict | None = None, privacy_constraints: list[str] | None = None):  # type: ignore[type-arg]
    prompt_service = PromptGenerationService(
        model_client=FakeModelClient(response or valid_prompt_response()),
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    snapshot = make_snapshot(text, privacy_constraints=privacy_constraints)
    result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)
    return result.candidates[0]


def test_runtime_logs_prompt_decision_from_generated_candidate() -> None:
    candidate = make_candidate("这个问题谁负责，下周五 deadline 前能不能定？")
    runtime = PromptRuntimeService()

    decision = runtime.log_candidate(candidate, decision_id="dec_001")

    assert decision.decision_id == "dec_001"
    assert decision.display_status == PromptDecisionDisplayStatus.SHOWN
    assert decision.prompt_category == "summary_gap_check"
    assert decision.content_granularity == 2
    assert decision.prd_surface == "glasses_popup"
    assert decision.shown_at is not None
    assert decision.metadata["model_usage"]["model"] == "gpt-test"
    assert runtime.store.list_decisions(session_id="session_001") == [decision]


def test_positive_feedback_builds_positive_reward_and_action_item_memory_candidate() -> None:
    candidate = make_candidate("这个问题谁负责，下周五 deadline 前能不能定？")
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(candidate, decision_id="dec_positive")

    runtime.record_feedback(decision.decision_id, FeedbackSignalType.ACCEPT, event_id="fb_accept")
    runtime.record_feedback(decision.decision_id, FeedbackSignalType.OPEN_DETAIL, event_id="fb_open")
    runtime.record_feedback(decision.decision_id, FeedbackSignalType.TASK_PROGRESS, event_id="fb_progress")

    reward = runtime.compute_reward_observation(decision.decision_id)
    memory_candidates = runtime.propose_memory_candidates(decision.decision_id)

    assert reward.final_reward > 2.0
    assert reward.components.accept == 1.0
    assert reward.components.helpfulness >= 0.7
    assert reward.components.task_progress > 0.0
    assert reward.explicit_score > 0.0
    assert reward.task_outcome_score > 0.0
    assert reward.confidence >= 0.9
    assert any(candidate.candidate_type == MemoryCandidateType.ACTION_ITEM for candidate in memory_candidates)
    action_item = [candidate for candidate in memory_candidates if candidate.candidate_type == MemoryCandidateType.ACTION_ITEM][0]
    assert action_item.write_policy == MemoryWritePolicy.NEEDS_CONFIRMATION
    assert "owner" in action_item.text


def test_negative_feedback_builds_negative_reward_and_privacy_memory_candidate() -> None:
    candidate = make_candidate(
        "这个客户报价风险怎么跟老板说？",
        response=valid_prompt_response(
            prompt_category="suggestion",
            glasses_title="建议稍后查看",
            glasses_text="涉及客户报价，建议在 APP 里查看更完整建议。",
            privacy_level="high",
            privacy_risk=0.78,
            safety_flags=["sensitive_business_context"],
        ),
        privacy_constraints=["avoid customer data"],
    )
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(candidate, decision_id="dec_negative")

    runtime.record_feedback(decision.decision_id, FeedbackSignalType.DISMISS, event_id="fb_dismiss")
    runtime.record_feedback(decision.decision_id, FeedbackSignalType.FLOW_BREAK, event_id="fb_flow")
    runtime.record_feedback(decision.decision_id, FeedbackSignalType.PRIVACY_REJECT, event_id="fb_privacy")
    runtime.record_feedback(
        decision.decision_id,
        FeedbackSignalType.LATENCY_OBSERVED,
        event_id="fb_latency",
        latency_ms=6000,
    )

    reward = runtime.compute_reward_observation(decision.decision_id)
    memory_candidates = runtime.propose_memory_candidates(decision.decision_id)

    assert reward.final_reward < -3.0
    assert reward.components.annoyance == 0.8
    assert reward.components.flow_break == 1.0
    assert reward.components.privacy_risk == 1.0
    assert reward.components.latency_penalty == 1.0
    assert reward.implicit_score < 0.0
    assert any(candidate.candidate_type == MemoryCandidateType.PRIVACY_PREFERENCE for candidate in memory_candidates)
    privacy_candidate = [
        candidate for candidate in memory_candidates if candidate.candidate_type == MemoryCandidateType.PRIVACY_PREFERENCE
    ][0]
    assert privacy_candidate.write_policy == MemoryWritePolicy.ELIGIBLE
    assert "客户报价" not in privacy_candidate.text


def test_ignore_is_neutral_implicit_feedback_not_direct_negative() -> None:
    candidate = make_candidate("腾讯是哪一年成立的？")
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(candidate, decision_id="dec_ignore")

    event = runtime.record_feedback(decision.decision_id, FeedbackSignalType.IGNORE, event_id="fb_ignore")
    reward = runtime.compute_reward_observation(decision.decision_id)

    assert event.source == FeedbackSignalSource.IMPLICIT
    assert event.polarity == FeedbackPolarity.NEUTRAL
    assert reward.components.accept == 0.0
    assert reward.components.annoyance == 0.0
    assert reward.implicit_score == 0.0


def test_no_action_model_output_is_logged_as_suppressed_decision() -> None:
    candidate = make_candidate(
        "腾讯是哪一年成立的？",
        response={
            "should_prompt": False,
            "content_granularity": 0,
            "confidence": 0.31,
            "privacy_risk": 0.05,
            "rationale": "上下文不足，不主动打断。",
        },
    )
    runtime = PromptRuntimeService()

    decision = runtime.log_candidate(candidate, decision_id="dec_suppressed")

    assert decision.display_status == PromptDecisionDisplayStatus.SUPPRESSED
    assert decision.content_granularity == 0
    assert decision.shown_at is None


def test_feedback_requires_existing_decision() -> None:
    runtime = PromptRuntimeService()

    with pytest.raises(DecisionRecordNotFoundError):
        runtime.record_feedback("missing_decision", FeedbackSignalType.ACCEPT)


def test_manual_mode_feedback_becomes_eligible_memory_preference() -> None:
    candidate = make_candidate("这个问题谁负责，下周五 deadline 前能不能定？")
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(candidate, decision_id="dec_manual")

    runtime.record_feedback(decision.decision_id, FeedbackSignalType.SWITCH_TO_MANUAL, event_id="fb_manual")

    memory_candidates = runtime.propose_memory_candidates(decision.decision_id)

    assert len(memory_candidates) == 1
    assert memory_candidates[0].candidate_type == MemoryCandidateType.USER_PREFERENCE
    assert memory_candidates[0].write_policy == MemoryWritePolicy.ELIGIBLE
    assert "manual" in memory_candidates[0].text
