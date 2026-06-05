from typing import Any

from proactive_assistant.meeting_state import MeetingGapType
from proactive_assistant.memory import MemoryQuery, MemoryRetrievalIntent
from proactive_assistant.model_gateway import FakeModelClient, ModelRequest
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product import ProductAssistantService
from proactive_assistant.prompting import PromptGenerationService
from proactive_assistant.runtime import FeedbackSignalType, MemoryCandidateType, MemoryWritePolicy, PromptRuntimeService
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


def make_product_service(response: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
    client = FakeModelClient(response or valid_prompt_response())
    prompt_service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    service = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=PromptRuntimeService(),
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
    service, _client = make_product_service()
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
