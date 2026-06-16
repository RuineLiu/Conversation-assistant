from typing import Any
from datetime import UTC, datetime

from proactive_assistant.asr import FakeSpeechRecognizer, SpeechRecognitionService
from proactive_assistant.meeting_state import MeetingGapType
from proactive_assistant.memory import (
    MemoryRecord,
    MemoryExtractionService,
    MemoryQuery,
    MemoryRetrievalIntent,
    MemoryScope,
    MemorySource,
    MemoryType,
    MemoryUpdatePolicyDecision,
    MemoryUpsertStatus,
)
from proactive_assistant.model_gateway import FakeModelClient, ModelRequest
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product import ProductAssistantService
from proactive_assistant.prompting import PromptGenerationService
from proactive_assistant.runtime import (
    FeedbackInputChannel,
    FeedbackSignalType,
    FeedbackTarget,
    MemoryCandidateType,
    MemoryWritePolicy,
    ProactiveDisplayStrategy,
    PromptRuntimeService,
)
from proactive_assistant.sessions import InMemorySessionStore, SessionConfig, SessionService, TranscriptSegmentInput


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


def make_product_service(
    response: dict[str, Any] | None = None,
    *,
    speech_recognizer: FakeSpeechRecognizer | None = None,
    auto_memory_snapshot: bool = True,
):  # type: ignore[no-untyped-def]
    client = FakeModelClient(response or valid_prompt_response())
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    service = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=PromptRuntimeService(),
        speech_recognition_service=SpeechRecognitionService(speech_recognizer) if speech_recognizer is not None else None,
        auto_memory_snapshot=auto_memory_snapshot,
    )
    return service, client


def transcript(text: str) -> TranscriptSegmentInput:
    return TranscriptSegmentInput(
        speaker="Bao",
        start_ms=0,
        end_ms=900,
        text=text,
        asr_confidence=0.94,
    )


def test_product_flow_returns_no_prompt_for_plain_transcript() -> None:
    service, client = make_product_service()
    session = service.create_session(SessionConfig(title="Daily sync"), session_id="session_001")

    result = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("今天项目进展正常，我们继续按计划推进。"),
        segment_id="seg_0",
    )

    assert result.prompts == []
    assert result.decisions == []
    assert result.opportunity_count == 0
    assert result.candidate_count == 0
    assert result.transcript_segment.segment_id == "seg_0"
    assert result.meeting_state is not None
    assert result.meeting_state.utterances[0].text == "今天项目进展正常，我们继续按计划推进。"
    assert result.meeting_gaps == []
    assert client.requests == []


def test_product_flow_appends_transcript_generates_prompt_and_logs_decision() -> None:
    service, client = make_product_service()
    session = service.create_session(
        SessionConfig(title="Launch risk sync", pre_context="讨论风险和负责人。"),
        session_id="session_001",
    )

    result = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )

    prompt = result.prompts[0]
    decision = result.decisions[0]
    assert result.opportunity_count == 1
    assert result.candidate_count == 1
    assert result.meeting_state is not None
    assert result.meeting_state.action_items[0].deadline == "下周五"
    assert MeetingGapType.ACTION_MISSING_OWNER in {gap.gap_type for gap in result.meeting_gaps}
    assert prompt.should_display is True
    assert prompt.display_status == "shown"
    assert prompt.prompt_category == "summary_gap_check"
    assert prompt.prd_surface == "glasses_popup"
    assert prompt.glasses_text == "这个风险还没有明确 owner 和截止时间。"
    assert prompt.source_refs == ["transcript:seg_0"]
    assert decision.decision_id == prompt.decision_id
    assert service.list_prompt_decisions(session_id=session.session_id) == [decision]
    assert service.list_prompt_payloads(session_id=session.session_id)[0].decision_id == decision.decision_id
    assert "这个问题谁负责" in client.requests[0].input_text


def test_product_flow_does_not_relog_prior_segment_prompt_on_followup_final() -> None:
    service, _client = make_product_service(
        valid_prompt_response(
            prompt_category="person_or_fact",
            glasses_title="上周模型测评",
            glasses_text="模型测评问题",
            app_detail_text="需要确认上周模型测评的问题。",
        )
    )
    session = service.create_session(
        SessionConfig(title="Model eval recall", pre_context="讨论上周模型测评。"),
        session_id="session_recall_001",
    )

    first = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("你们谁还记得我们上周说的那个呃模型测评的问题？"),
        segment_id="seg_recall_001",
    )
    second = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("就是呃你说的那个模型测评哦，对哦，好像。"),
        segment_id="seg_recall_002",
    )

    assert len(first.prompts) == 1
    assert second.prompts == []
    assert [item.text for item in service.list_transcript(session.session_id)] == [
        "你们谁还记得我们上周说的那个呃模型测评的问题？",
        "就是呃你说的那个模型测评哦，对哦，好像。",
    ]
    assert len(service.list_prompt_decisions(session_id=session.session_id)) == 1


def test_product_flow_transcribes_audio_and_runs_prompt_flow() -> None:
    speech = FakeSpeechRecognizer(text="这个问题谁负责，下周五 deadline 前能不能定？")
    service, _client = make_product_service(speech_recognizer=speech)
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")

    result = service.append_audio_transcript_and_generate_prompts(
        session.session_id,
        b"fake-wav-bytes",
        speaker="Bao",
        start_ms=0,
        end_ms=900,
        segment_id="seg_audio_0",
        content_type="audio/wav",
    )

    assert result.transcription.text.startswith("这个问题谁负责")
    assert speech.requests[0]["content_type"] == "audio/wav"
    assert result.transcript_step is not None
    assert result.transcript_step.transcript_segment.segment_id == "seg_audio_0"
    assert result.transcript_step.transcript_segment.source == "uploaded_audio_transcript"
    assert result.transcript_step.prompts[0].prompt_category == "summary_gap_check"


def test_product_session_lifecycle_and_state_view() -> None:
    service, _client = make_product_service()
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )

    paused = service.pause_session(session.session_id)
    resumed = service.resume_session(session.session_id)
    state = service.get_session_state(session.session_id)

    assert paused.session.status == "paused"
    assert resumed.session.status == "running"
    assert state.session.session_id == session.session_id
    assert state.meeting_state.session_id == session.session_id
    assert state.transcript[0].segment_id == "seg_0"
    assert state.prompts[0].prompt_category == "summary_gap_check"


def test_product_flow_generates_post_session_summary_prompt() -> None:
    service, client = make_product_service(
        valid_prompt_response(
            content_granularity=3,
            glasses_title="会议总结",
            glasses_text="已整理关键结论、待办和未确认 GAP。",
            app_detail_text="总结：需要确认负责人、deadline 和下一步风险处理。",
            source_refs=["transcript:seg_0"],
        )
    )
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )

    result = service.generate_session_summary(session.session_id, use_memory=False)

    assert result.session.session_id == session.session_id
    assert result.prompts[0].prd_surface == "app_summary_tab"
    assert result.prompts[0].content_granularity == 3
    assert result.prompts[0].glasses_title == "会议总结"
    assert result.decisions[0].policy_version == "product_summary_v0"
    assert result.decisions[0].metadata["summary_source"] == "session_summary_v0"
    assert "会后总结" in client.requests[-1].input_text


def test_product_end_session_can_generate_summary() -> None:
    service, _client = make_product_service(
        valid_prompt_response(
            content_granularity=3,
            glasses_title="会议总结",
            glasses_text="已整理关键结论、待办和未确认 GAP。",
            app_detail_text="总结：需要确认负责人、deadline 和下一步风险处理。",
            source_refs=["transcript:seg_0"],
        )
    )
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )

    ended = service.end_session(session.session_id, generate_summary=True, use_memory=False)

    assert ended.session.status == "ended"
    assert ended.summary is not None
    assert ended.summary.prompts[0].prd_surface == "app_summary_tab"
    assert ended.summary.prompts[0].glasses_title == "会议总结"


def test_product_feedback_flow_computes_reward_and_memory_candidate() -> None:
    service, _client = make_product_service()
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )
    decision_id = step.prompts[0].decision_id

    feedback = service.record_feedback(decision_id, FeedbackSignalType.ACCEPT, event_id="fb_accept")

    assert feedback.feedback_event.signal_type == "accept"
    assert feedback.reward_observation is not None
    assert feedback.reward_observation.final_reward > 0.0
    assert len(feedback.memory_candidates) == 1
    assert feedback.memory_candidates[0].candidate_type == MemoryCandidateType.ACTION_ITEM
    assert feedback.memory_candidates[0].write_policy == MemoryWritePolicy.NEEDS_CONFIRMATION
    assert len(feedback.memories) == 1
    assert feedback.memories[0].write_status == "pending_confirmation"
    assert service.list_memory_candidates(decision_id=decision_id) == feedback.memory_candidates


def test_product_feedback_flow_records_device_feedback_taxonomy() -> None:
    service, _client = make_product_service()
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )

    feedback = service.record_feedback(
        step.prompts[0].decision_id,
        FeedbackSignalType.HEAD_SHAKE_REJECT,
        event_id="fb_head_shake",
        input_channel=FeedbackInputChannel.GESTURE,
        target=FeedbackTarget.TIMING,
        display_strategy=ProactiveDisplayStrategy.AUTO_POPUP,
    )

    assert feedback.feedback_event.signal_type == "head_shake_reject"
    assert feedback.feedback_event.input_channel == "gesture"
    assert feedback.feedback_event.target == "timing"
    assert feedback.feedback_event.display_strategy == "auto_popup"
    assert feedback.reward_observation is not None
    assert feedback.reward_observation.components.timing_fit < 0.0
    assert feedback.reward_observation.final_reward < 0.0


def test_product_feedback_flow_records_manual_request_as_missed_opportunity() -> None:
    service, _client = make_product_service(
        {
            "should_prompt": False,
            "content_granularity": 0,
            "confidence": 0.31,
            "privacy_risk": 0.05,
            "rationale": "上下文不足，不主动打断。",
        }
    )
    session = service.create_session(SessionConfig(title="Fact recall"), session_id="session_001")
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("腾讯是哪一年成立的？"),
        segment_id="seg_0",
    )

    feedback = service.record_feedback(
        step.prompts[0].decision_id,
        FeedbackSignalType.MANUAL_REQUEST,
        event_id="fb_manual_request",
        input_channel=FeedbackInputChannel.BUTTON,
        display_strategy=ProactiveDisplayStrategy.MANUAL_RESPONSE,
        propose_memory=False,
    )

    assert feedback.feedback_event.signal_type == "manual_request"
    assert feedback.feedback_event.target == "timing"
    assert feedback.feedback_event.input_channel == "button"
    assert feedback.reward_observation is not None
    assert feedback.reward_observation.components.missed_opportunity == 1.0
    assert feedback.reward_observation.final_reward < -1.0
    assert feedback.memory_candidates == []


def test_product_flow_exports_policy_episode_for_session() -> None:
    service, _client = make_product_service()
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )
    feedback = service.record_feedback(
        step.prompts[0].decision_id,
        FeedbackSignalType.NOD_ACCEPT,
        event_id="fb_nod",
        propose_memory=False,
    )

    result = service.get_policy_episode(session.session_id)

    assert result.session.session_id == session.session_id
    assert result.episode.session_id == session.session_id
    assert result.episode.total_reward == feedback.reward_observation.final_reward
    assert result.episode.steps[0].decision_id == step.prompts[0].decision_id
    assert result.episode.steps[0].state.memory_refs == []
    assert result.episode.steps[0].action.prompt_category == "summary_gap_check"
    assert result.episode.steps[0].feedback_events[0].signal_type == "nod_accept"


def test_product_flow_exports_policy_evaluation_for_session() -> None:
    service, _client = make_product_service()
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )
    feedback = service.record_feedback(
        step.prompts[0].decision_id,
        FeedbackSignalType.HEAD_SHAKE_REJECT,
        event_id="fb_head_shake",
        propose_memory=False,
    )

    result = service.evaluate_policy_episode(session.session_id)

    assert result.session.session_id == session.session_id
    assert result.report.session_id == session.session_id
    assert result.report.summary.step_count == 1
    assert result.report.summary.negative_feedback_rate == 1.0
    assert result.report.summary.total_reward == feedback.reward_observation.final_reward
    assert result.report.by_prompt_category[0].value == "summary_gap_check"
    assert result.report.by_prompt_category[0].metrics.step_count == 1


def test_product_flow_exports_policy_baselines_for_session() -> None:
    service, _client = make_product_service(
        {
            "should_prompt": False,
            "content_granularity": 0,
            "confidence": 0.31,
            "privacy_risk": 0.05,
            "rationale": "当前先静默。",
        }
    )
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )
    service.record_feedback(
        step.prompts[0].decision_id,
        FeedbackSignalType.MANUAL_REQUEST,
        event_id="fb_manual",
        propose_memory=False,
    )

    result = service.evaluate_policy_baselines(session.session_id)

    baselines = {baseline.baseline_name: baseline for baseline in result.report.baselines}
    assert result.session.session_id == session.session_id
    assert set(baselines) == {"conservative", "balanced", "aggressive"}
    assert baselines["balanced"].metrics.missed_opportunity_coverage_rate == 1.0
    assert baselines["balanced"].decisions[0].missed_opportunity_covered is True


def test_product_flow_exports_policy_training_examples_for_session() -> None:
    service, _client = make_product_service()
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )
    feedback = service.record_feedback(
        step.prompts[0].decision_id,
        FeedbackSignalType.NOD_ACCEPT,
        event_id="fb_export_nod",
        propose_memory=False,
    )

    result = service.export_policy_training_examples(session.session_id)

    example = result.export.examples[0]
    assert result.session.session_id == session.session_id
    assert result.export.session_id == session.session_id
    assert result.export.example_count == 1
    assert result.export.evaluation_summary.total_reward == feedback.reward_observation.final_reward
    assert example.decision_id == step.prompts[0].decision_id
    assert example.label.accepted is True
    assert example.label.final_reward == feedback.reward_observation.final_reward
    assert example.action.prompt_category == "summary_gap_check"
    assert len(example.baseline_decisions) == 3


def test_product_flow_writes_meeting_state_snapshot_memory_once_and_supports_exact_lookup() -> None:
    service, _client = make_product_service()
    session = service.create_session(
        SessionConfig(
            title="Customer pricing sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )
    service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("张三负责客户报价确认，下周五截止。"),
        segment_id="seg_0",
    )

    snapshot = service.write_meeting_state_memory_snapshot(session.session_id)
    duplicate_snapshot = service.write_meeting_state_memory_snapshot(session.session_id)
    stored = service.memory.store.list_memories(MemoryQuery(session_id=session.session_id, include_pending=True, limit=20))
    context = service.search_memory_context_for_session(
        session.session_id,
        query_text="之前这个 deadline 是什么时候？",
        include_pending=True,
    )

    assert snapshot.committed is True
    assert any(candidate.candidate_type == MemoryCandidateType.ACTION_ITEM for candidate in snapshot.memory_candidates)
    assert len(duplicate_snapshot.memories) == len(snapshot.memories)
    assert len({memory.memory_id for memory in stored}) == len(stored)
    assert any(memory.metadata["meeting_state_object_type"] == "action_item" for memory in snapshot.memories)
    assert context.memory_refs
    assert context.results[0].intent == MemoryRetrievalIntent.LOOKUP_DEADLINE.value


def test_product_flow_extracts_llm_memory_candidates_and_upserts_them() -> None:
    prompt_client = FakeModelClient(valid_prompt_response())
    extraction_client = FakeModelClient(
        {
            "candidates": [
                {
                    "candidate_type": "action_item",
                    "text": "张三负责客户报价确认，下周五截止。",
                    "confidence": 0.84,
                    "write_policy": "eligible",
                    "privacy_level": "medium",
                    "privacy_risk": 0.35,
                    "source_refs": ["transcript:seg_0"],
                    "reason": "明确出现负责人和截止时间。",
                    "entity": "客户报价确认",
                    "owner": "张三",
                    "deadline": "下周五",
                    "status": "open",
                    "topic": "客户报价",
                    "tags": ["action_item"],
                    "promotion_candidate": False,
                }
            ],
            "extraction_notes": "extracted action item",
            "safety_flags": [],
        }
    )
    service = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(
            prompt_service=PromptGenerationService(
                model_client=prompt_client,
                settings=ModelGatewaySettings(default_model="gpt-test"),
            )
        ),
        runtime_service=PromptRuntimeService(),
        memory_extraction_service=MemoryExtractionService(
            model_client=extraction_client,
            settings=ModelGatewaySettings(default_model="gpt-memory-test"),
        ),
        auto_memory_snapshot=False,
    )
    session = service.create_session(
        SessionConfig(
            title="Customer pricing sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )
    service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("张三负责客户报价确认，下周五截止。"),
        segment_id="seg_0",
    )

    extraction = service.extract_session_memories(session.session_id)

    assert len(extraction.memory_candidates) == 1
    assert extraction.memory_candidates[0].candidate_type == MemoryCandidateType.ACTION_ITEM.value
    assert extraction.memory_candidates[0].metadata["owner"] == "张三"
    assert extraction.memory_upserts[0].status == MemoryUpsertStatus.CREATED.value
    assert extraction.memories[0].memory_type == "action_item"
    assert extraction.memories[0].metadata["memory_extraction_source"] == "llm_memory_extraction_v1"
    assert extraction.memories[0].metadata["owner"] == "张三"
    assert extraction.model_usage is not None
    assert extraction.model_usage.model == "gpt-memory-test"
    assert extraction_client.requests[0].metadata["contract"] == "memory_extraction_v1"


def test_product_flow_requires_confirmation_for_sensitive_meeting_state_snapshot_update() -> None:
    service, _client = make_product_service()
    session = service.create_session(
        SessionConfig(
            title="Customer pricing sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("张三负责客户报价确认，下周五截止。"),
        segment_id="seg_0",
    )

    first_snapshot = service.write_meeting_state_memory_snapshot(session.session_id, include_gaps=False)
    action_memory = first_snapshot.memories[0]
    action_item = step.meeting_state.action_items[0]
    updated_action_item = action_item.model_copy(
        update={
            "owner": "李四",
            "deadline": "下周一",
            "evidence": "李四负责客户报价确认，下周一截止。",
            "desc": "李四负责客户报价确认，下周一截止。",
        }
    )
    service._meeting_states[session.session_id] = step.meeting_state.model_copy(update={"action_items": [updated_action_item]})

    second_snapshot = service.write_meeting_state_memory_snapshot(session.session_id, include_gaps=False)
    stored = service.memory.store.list_memories(MemoryQuery(session_id=session.session_id, include_pending=True, limit=20))

    assert second_snapshot.memory_upserts[0].status == MemoryUpsertStatus.UNCHANGED.value
    assert second_snapshot.memory_upserts[0].update_policy == MemoryUpdatePolicyDecision.NEEDS_CONFIRMATION.value
    assert second_snapshot.memories[0].memory_id == action_memory.memory_id
    assert second_snapshot.memories[0].metadata["memory_version"] == 1
    assert second_snapshot.memories[0].metadata["owner"] == "张三"
    assert second_snapshot.memories[0].metadata["deadline"] == "下周五"
    assert second_snapshot.memory_upserts[0].proposed_memory is not None
    assert second_snapshot.memory_upserts[0].proposed_memory.metadata["owner"] == "李四"
    assert second_snapshot.memory_upserts[0].proposed_memory.metadata["deadline"] == "下周一"
    assert "metadata.owner" in second_snapshot.memory_upserts[0].changed_fields
    assert len(stored) == 1


def test_product_feedback_can_skip_reward_and_memory_generation() -> None:
    service, _client = make_product_service()
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )

    feedback = service.record_feedback(
        step.prompts[0].decision_id,
        FeedbackSignalType.IGNORE,
        event_id="fb_ignore",
        compute_reward=False,
        propose_memory=False,
    )

    assert feedback.feedback_event.signal_type == "ignore"
    assert feedback.reward_observation is None
    assert feedback.memory_candidates == []
    assert feedback.memories == []


def test_product_flow_retrieves_confirmed_memory_into_next_prompt_snapshot() -> None:
    # This test asserts the memory store is empty before user accepts a
    # prompt; auto_memory_snapshot would populate it eagerly via the
    # MeetingState path. Opt out so we keep testing the feedback path.
    service, _client = make_product_service(auto_memory_snapshot=False)
    session = service.create_session(
        SessionConfig(
            title="Launch risk sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )
    first = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )
    feedback = service.record_feedback(first.prompts[0].decision_id, FeedbackSignalType.ACCEPT, event_id="fb_accept")
    memory = feedback.memories[0]

    assert service.search_memory_context_for_session(
        session.session_id,
        query_text="owner 截止时间",
    ).memory_context == []

    service.confirm_memory(memory.memory_id)
    second = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("owner 和截止时间现在还有谁需要确认？"),
        segment_id="seg_1",
    )

    assert second.retrieved_memory_context is not None
    assert second.retrieved_memory_context.memory_refs == [f"memory:{memory.memory_id}"]
    assert second.snapshot.memory_refs == [f"memory:{memory.memory_id}"]
    assert second.snapshot.memory_context[0].startswith(f"[memory:{memory.memory_id}]")


def test_product_flow_records_memory_context_on_decision_metadata() -> None:
    def response_for_request(request: ModelRequest) -> dict[str, Any]:
        assert "张三负责客户报价确认" in request.input_text
        return valid_prompt_response()

    service, client = make_product_service()
    client._response = response_for_request
    session = service.create_session(
        SessionConfig(
            title="Customer pricing sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )
    created_at = datetime(2026, 6, 5, tzinfo=UTC)
    service.memory.store.add_memory(
        MemoryRecord(
            memory_id="mem_customer_pricing_owner",
            memory_type=MemoryType.ACTION_ITEM,
            scope=MemoryScope.SESSION,
            text="张三负责客户报价确认，下周五截止。",
            org_id="org_001",
            user_id="user_001",
            session_id=session.session_id,
            source=MemorySource.MANUAL,
            source_ids=["manual:pricing_owner"],
            confidence=0.9,
            importance=0.9,
            tags=["客户报价确认", "owner", "deadline"],
            created_at=created_at,
            updated_at=created_at,
            metadata={"entity": "客户报价确认", "owner": "张三", "deadline": "下周五"},
        )
    )

    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("客户报价确认这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
    )

    assert step.snapshot.memory_refs == ["memory:mem_customer_pricing_owner"]
    assert step.decisions[0].metadata["memory_refs"] == ["memory:mem_customer_pricing_owner"]
    assert step.decisions[0].metadata["retrieved_memory_refs"] == ["memory:mem_customer_pricing_owner"]
    assert step.decisions[0].metadata["retrieved_memory_result_count"] == 1
    assert step.decisions[0].metadata["memory_query_prompt_category"] == "summary_gap_check"
    assert "客户报价确认" in step.decisions[0].metadata["memory_query_text"]
    assert client.requests[0].metadata["session_id"] == session.session_id


def test_product_flow_handles_suppressed_prompt_candidate() -> None:
    service, _client = make_product_service(
        {
            "should_prompt": False,
            "content_granularity": 0,
            "confidence": 0.31,
            "privacy_risk": 0.05,
            "rationale": "上下文不足，不主动打断。",
        }
    )
    session = service.create_session(SessionConfig(title="Fact recall"), session_id="session_001")

    result = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("腾讯是哪一年成立的？"),
        segment_id="seg_0",
    )

    prompt = result.prompts[0]
    assert prompt.should_display is False
    assert prompt.display_status == "suppressed"
    assert prompt.content_granularity == 0
    assert prompt.glasses_text == ""
    assert result.decisions[0].shown_at is None


def test_product_flow_routes_high_privacy_prompt_and_feedback_to_privacy_memory() -> None:
    service, _client = make_product_service(
        valid_prompt_response(
            prompt_category="suggestion",
            glasses_title="建议稍后查看",
            glasses_text="涉及客户报价，建议在 APP 里查看更完整建议。",
            privacy_level="high",
            privacy_risk=0.78,
            safety_flags=["sensitive_business_context"],
        )
    )
    session = service.create_session(
        SessionConfig(title="Client pricing", privacy_constraints=["avoid customer data"]),
        session_id="session_001",
    )
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个客户报价风险怎么跟老板说？"),
        segment_id="seg_0",
    )

    prompt = step.prompts[0]
    feedback = service.record_feedback(prompt.decision_id, FeedbackSignalType.PRIVACY_REJECT, event_id="fb_privacy")

    assert prompt.prd_surface == "app_prompt_tab"
    assert prompt.display_mode == "wrist_turn"
    assert prompt.privacy_level == "high"
    assert feedback.reward_observation is not None
    assert feedback.reward_observation.final_reward < 0.0
    assert feedback.memory_candidates[0].candidate_type == MemoryCandidateType.PRIVACY_PREFERENCE
    assert feedback.memory_candidates[0].write_policy == MemoryWritePolicy.ELIGIBLE
    assert feedback.memories[0].write_status == "active"
    assert feedback.memories[0].privacy_level == "high"
    assert "客户报价" not in feedback.memory_candidates[0].text


def test_product_flow_keeps_memory_context_in_snapshot() -> None:
    def response_for_request(request: ModelRequest) -> dict[str, Any]:
        assert "Last meeting: Alex owned release notes." in request.input_text
        return valid_prompt_response()

    client = FakeModelClient(response_for_request)
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    service = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=PromptRuntimeService(),
    )
    session = service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_001")

    result = service.append_transcript_and_generate_prompts(
        session.session_id,
        transcript("这个问题谁负责，下周五 deadline 前能不能定？"),
        segment_id="seg_0",
        memory_context=["Last meeting: Alex owned release notes."],
        memory_refs=["memory:release_notes_owner"],
    )

    assert result.snapshot.memory_context == ["Last meeting: Alex owned release notes."]
    assert result.snapshot.memory_refs == ["memory:release_notes_owner"]
    assert len(client.requests) == 1
