import pytest

from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.prompting import PromptGenerationService
from proactive_assistant.runtime import (
    DecisionRecordNotFoundError,
    FeedbackInputChannel,
    FeedbackPolarity,
    FeedbackSignalSource,
    FeedbackSignalType,
    FeedbackTarget,
    MemoryCandidateType,
    MemoryWritePolicy,
    ProactiveDisplayStrategy,
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
    assert action_item.metadata["source_refs"] == ["transcript:seg_0"]
    assert action_item.metadata["source_capture_ref"] == "transcript:seg_0"
    assert action_item.metadata["captured_text"].startswith("这个问题谁负责")
    assert action_item.metadata["trigger_segment_ids"] == ["seg_0"]
    assert action_item.metadata["prd_surface"] == "glasses_popup"


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


def test_nod_accept_is_positive_gesture_feedback_for_shown_prompt() -> None:
    candidate = make_candidate("这个问题谁负责，下周五 deadline 前能不能定？")
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(candidate, decision_id="dec_nod")

    event = runtime.record_feedback(
        decision.decision_id,
        FeedbackSignalType.NOD_ACCEPT,
        event_id="fb_nod",
        display_strategy=ProactiveDisplayStrategy.AUTO_POPUP,
    )
    reward = runtime.compute_reward_observation(decision.decision_id)

    assert event.source == FeedbackSignalSource.EXPLICIT
    assert event.input_channel == FeedbackInputChannel.GESTURE
    assert event.target == FeedbackTarget.OVERALL
    assert event.polarity == FeedbackPolarity.POSITIVE
    assert event.display_strategy == ProactiveDisplayStrategy.AUTO_POPUP
    assert reward.components.accept == 1.0
    assert reward.final_reward > 0.0


def test_head_shake_reject_is_negative_gesture_feedback_for_shown_prompt() -> None:
    candidate = make_candidate("这个问题谁负责，下周五 deadline 前能不能定？")
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(candidate, decision_id="dec_head_shake")

    event = runtime.record_feedback(
        decision.decision_id,
        FeedbackSignalType.HEAD_SHAKE_REJECT,
        event_id="fb_head_shake",
    )
    reward = runtime.compute_reward_observation(decision.decision_id)

    assert event.source == FeedbackSignalSource.EXPLICIT
    assert event.input_channel == FeedbackInputChannel.GESTURE
    assert event.polarity == FeedbackPolarity.NEGATIVE
    assert reward.components.annoyance == 1.0
    assert reward.components.timing_fit < 0.0
    assert reward.components.display_fit < 0.0
    assert reward.final_reward < 0.0


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


def test_manual_request_on_suppressed_decision_penalizes_missed_opportunity() -> None:
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
    decision = runtime.log_candidate(candidate, decision_id="dec_manual_request")

    event = runtime.record_feedback(
        decision.decision_id,
        FeedbackSignalType.MANUAL_REQUEST,
        event_id="fb_manual_request",
        input_channel=FeedbackInputChannel.TOUCH,
        display_strategy=ProactiveDisplayStrategy.MANUAL_RESPONSE,
    )
    reward = runtime.compute_reward_observation(decision.decision_id)

    assert event.source == FeedbackSignalSource.EXPLICIT
    assert event.input_channel == FeedbackInputChannel.TOUCH
    assert event.target == FeedbackTarget.TIMING
    assert event.display_strategy == ProactiveDisplayStrategy.MANUAL_RESPONSE
    assert reward.components.missed_opportunity == 1.0
    assert reward.components.timing_fit == -1.0
    assert reward.final_reward < -1.0


def test_build_policy_episode_exports_decision_feedback_reward_steps() -> None:
    runtime = PromptRuntimeService()
    first = runtime.log_candidate(
        make_candidate("这个问题谁负责，下周五 deadline 前能不能定？"),
        decision_id="dec_episode_001",
    )
    second = runtime.log_candidate(
        make_candidate("还有哪个风险没有 owner？"),
        decision_id="dec_episode_002",
    )
    runtime.record_feedback(first.decision_id, FeedbackSignalType.NOD_ACCEPT, event_id="fb_episode_nod")
    reward = runtime.compute_reward_observation(first.decision_id)

    episode = runtime.build_policy_episode("session_001")

    assert episode.episode_id == "episode_session_001"
    assert episode.session_id == "session_001"
    assert len(episode.steps) == 2
    assert episode.feedback_event_count == 1
    assert episode.rewarded_step_count == 1
    assert episode.total_reward == reward.final_reward
    assert episode.steps[0].decision_id == first.decision_id
    assert episode.steps[0].state.transcript_segment_ids == ["seg_0"]
    assert episode.steps[0].state.timing_action == "during_activity"
    assert episode.steps[0].action.should_prompt is True
    assert episode.steps[0].action.display_strategy == ProactiveDisplayStrategy.AUTO_POPUP
    assert episode.steps[0].feedback_events[0].event_id == "fb_episode_nod"
    assert episode.steps[0].reward_observation is not None
    assert episode.steps[0].next_state_ref.next_decision_id == second.decision_id
    assert episode.steps[1].reward_observation is None
    assert episode.steps[1].next_state_ref.next_decision_id is None


def test_policy_evaluation_summarizes_positive_feedback_episode() -> None:
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(make_candidate("这个问题谁负责，下周五 deadline 前能不能定？"), decision_id="dec_eval_positive")
    runtime.record_feedback(decision.decision_id, FeedbackSignalType.NOD_ACCEPT, event_id="fb_eval_nod")
    reward = runtime.compute_reward_observation(decision.decision_id)

    report = runtime.evaluate_policy_episode("session_001")

    assert report.report_id == "eval_episode_session_001"
    assert report.summary.step_count == 1
    assert report.summary.rewarded_step_count == 1
    assert report.summary.feedback_event_count == 1
    assert report.summary.total_reward == reward.final_reward
    assert report.summary.average_reward_per_step == reward.final_reward
    assert report.summary.accept_rate == 1.0
    assert report.summary.negative_feedback_rate == 0.0
    assert report.summary.rewarded_step_coverage == 1.0


def test_policy_evaluation_counts_manual_request_as_missed_opportunity() -> None:
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(
        make_candidate(
            "腾讯是哪一年成立的？",
            response={
                "should_prompt": False,
                "content_granularity": 0,
                "confidence": 0.31,
                "privacy_risk": 0.05,
                "rationale": "上下文不足，不主动打断。",
            },
        ),
        decision_id="dec_eval_manual_request",
    )
    runtime.record_feedback(decision.decision_id, FeedbackSignalType.MANUAL_REQUEST, event_id="fb_eval_manual")
    runtime.compute_reward_observation(decision.decision_id)

    report = runtime.evaluate_policy_episode("session_001")

    assert report.summary.missed_opportunity_rate == 1.0
    assert report.summary.negative_feedback_rate == 1.0
    assert report.summary.display_mismatch_rate == 1.0
    assert report.by_display_status[0].value == "suppressed"
    assert report.by_display_status[0].metrics.missed_opportunity_rate == 1.0


def test_policy_evaluation_counts_head_shake_as_negative_feedback() -> None:
    runtime = PromptRuntimeService()
    decision = runtime.log_candidate(make_candidate("这个问题谁负责，下周五 deadline 前能不能定？"), decision_id="dec_eval_head_shake")
    runtime.record_feedback(decision.decision_id, FeedbackSignalType.HEAD_SHAKE_REJECT, event_id="fb_eval_head_shake")
    runtime.compute_reward_observation(decision.decision_id)

    report = runtime.evaluate_policy_episode("session_001")

    assert report.summary.negative_feedback_rate == 1.0
    assert report.summary.display_mismatch_rate == 1.0
    assert report.summary.accept_rate == 0.0
    assert report.summary.total_reward < 0.0


def test_policy_evaluation_builds_action_breakdowns_for_multi_step_episode() -> None:
    runtime = PromptRuntimeService()
    accepted = runtime.log_candidate(make_candidate("这个问题谁负责，下周五 deadline 前能不能定？"), decision_id="dec_eval_accept")
    rejected = runtime.log_candidate(make_candidate("还有哪个风险没有 owner？"), decision_id="dec_eval_reject")
    suppressed = runtime.log_candidate(
        make_candidate(
            "腾讯是哪一年成立的？",
            response={
                "should_prompt": False,
                "content_granularity": 0,
                "confidence": 0.31,
                "privacy_risk": 0.05,
                "rationale": "上下文不足，不主动打断。",
            },
        ),
        decision_id="dec_eval_suppressed",
    )
    runtime.record_feedback(accepted.decision_id, FeedbackSignalType.NOD_ACCEPT, event_id="fb_eval_accept")
    runtime.compute_reward_observation(accepted.decision_id)
    runtime.record_feedback(rejected.decision_id, FeedbackSignalType.HEAD_SHAKE_REJECT, event_id="fb_eval_reject")
    runtime.compute_reward_observation(rejected.decision_id)
    runtime.record_feedback(suppressed.decision_id, FeedbackSignalType.MANUAL_REQUEST, event_id="fb_eval_missed")
    runtime.compute_reward_observation(suppressed.decision_id)

    report = runtime.evaluate_policy_episode("session_001")

    by_display_status = {item.value: item.metrics for item in report.by_display_status}
    by_granularity = {item.value: item.metrics for item in report.by_content_granularity}
    by_display_strategy = {item.value: item.metrics for item in report.by_display_strategy}

    assert report.summary.step_count == 3
    assert report.summary.rewarded_step_count == 3
    assert report.summary.accept_rate == pytest.approx(1 / 3, abs=0.0001)
    assert report.summary.negative_feedback_rate == pytest.approx(2 / 3, abs=0.0001)
    assert report.summary.missed_opportunity_rate == pytest.approx(1 / 3, abs=0.0001)
    assert by_display_status["shown"].step_count == 2
    assert by_display_status["suppressed"].missed_opportunity_rate == 1.0
    assert by_granularity["0"].step_count == 1
    assert by_granularity["2"].step_count == 2
    assert by_display_strategy["auto_popup"].step_count == 2
    assert by_display_strategy["manual_response"].step_count == 1


def test_policy_baselines_compare_three_deterministic_strategies() -> None:
    runtime = PromptRuntimeService()
    shown = runtime.log_candidate(make_candidate("这个问题谁负责，下周五 deadline 前能不能定？"), decision_id="dec_baseline_shown")
    suppressed = runtime.log_candidate(
        make_candidate(
            "这个问题谁负责，下周五 deadline 前能不能定？",
            response={
                "should_prompt": False,
                "content_granularity": 0,
                "confidence": 0.31,
                "privacy_risk": 0.05,
                "rationale": "当前先静默。",
            },
        ),
        decision_id="dec_baseline_suppressed",
    )
    runtime.record_feedback(suppressed.decision_id, FeedbackSignalType.MANUAL_REQUEST, event_id="fb_baseline_manual")
    runtime.compute_reward_observation(suppressed.decision_id)

    report = runtime.evaluate_policy_baselines("session_001")

    baselines = {baseline.baseline_name: baseline for baseline in report.baselines}
    assert set(baselines) == {"conservative", "balanced", "aggressive"}
    assert report.report_id == "baseline_episode_session_001"
    assert baselines["balanced"].metrics.decision_count == 2
    assert baselines["balanced"].metrics.would_prompt_rate == 1.0
    assert baselines["balanced"].metrics.missed_opportunity_coverage_rate == 1.0
    assert baselines["balanced"].decisions[0].decision_id == shown.decision_id
    assert baselines["balanced"].decisions[1].decision_id == suppressed.decision_id
    assert baselines["balanced"].decisions[1].missed_opportunity_covered is True
    assert baselines["conservative"].decisions[0].baseline_action.display_strategy == ProactiveDisplayStrategy.SUBTLE_AVAILABLE
    assert baselines["conservative"].metrics.more_conservative_rate > 0.0
    assert baselines["aggressive"].metrics.would_prompt_rate == 1.0


def test_policy_baselines_block_high_privacy_opportunities_by_threshold() -> None:
    runtime = PromptRuntimeService()
    runtime.log_candidate(
        make_candidate(
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
        ),
        decision_id="dec_baseline_privacy",
    )

    report = runtime.evaluate_policy_baselines("session_001")
    baselines = {baseline.baseline_name: baseline for baseline in report.baselines}

    assert baselines["conservative"].metrics.privacy_block_rate == 1.0
    assert baselines["balanced"].metrics.privacy_block_rate == 1.0
    assert baselines["aggressive"].metrics.privacy_block_rate == 1.0
    assert baselines["conservative"].decisions[0].baseline_action.should_prompt is False


def test_policy_training_export_builds_jsonl_ready_examples_with_labels_and_baselines() -> None:
    runtime = PromptRuntimeService()
    accepted = runtime.log_candidate(make_candidate("这个问题谁负责，下周五 deadline 前能不能定？"), decision_id="dec_export_accept")
    suppressed = runtime.log_candidate(
        make_candidate(
            "腾讯是哪一年成立的？",
            response={
                "should_prompt": False,
                "content_granularity": 0,
                "confidence": 0.31,
                "privacy_risk": 0.05,
                "rationale": "上下文不足，不主动打断。",
            },
        ),
        decision_id="dec_export_missed",
    )
    runtime.record_feedback(accepted.decision_id, FeedbackSignalType.NOD_ACCEPT, event_id="fb_export_nod")
    accepted_reward = runtime.compute_reward_observation(accepted.decision_id)
    runtime.record_feedback(suppressed.decision_id, FeedbackSignalType.MANUAL_REQUEST, event_id="fb_export_manual")
    runtime.compute_reward_observation(suppressed.decision_id)

    export = runtime.export_policy_training_examples("session_001")

    examples = {example.decision_id: example for example in export.examples}
    assert export.export_id == "policy_export_episode_session_001"
    assert export.example_count == 2
    assert export.evaluation_summary.step_count == 2
    assert export.metadata["record_format"] == "jsonl_ready"
    assert examples[accepted.decision_id].label.accepted is True
    assert examples[accepted.decision_id].label.final_reward == accepted_reward.final_reward
    assert examples[accepted.decision_id].state.transcript_segment_ids == ["seg_0"]
    assert len(examples[accepted.decision_id].baseline_decisions) == 3
    assert {decision.baseline_name for decision in examples[accepted.decision_id].baseline_decisions} == {
        "conservative",
        "balanced",
        "aggressive",
    }
    assert examples[suppressed.decision_id].label.missed_opportunity is True
    assert examples[suppressed.decision_id].label.negative_feedback is True
    assert examples[suppressed.decision_id].action.should_prompt is False
    assert examples[suppressed.decision_id].next_state_ref.next_decision_id is None


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
