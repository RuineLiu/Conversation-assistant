from typing import Any

from fastapi.testclient import TestClient

from proactive_assistant.asr import (
    FakeSpeechRecognizer,
    FakeStreamingSpeechRecognizer,
    SpeechRecognitionService,
    StreamingSpeechRecognitionService,
)
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product.api import create_app
from proactive_assistant.product.service import ProductAssistantService
from proactive_assistant.memory import MemoryExtractionService
from proactive_assistant.prompting import PromptGenerationService
from proactive_assistant.runtime import PromptRuntimeService
from proactive_assistant.sessions import InMemorySessionStore, SessionService


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


def valid_memory_extraction_response(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidates": [
            {
                "candidate_type": "action_item",
                "text": "张三负责客户报价确认，下周五截止。",
                "confidence": 0.84,
                "write_policy": "eligible",
                "privacy_level": "medium",
                "privacy_risk": 0.35,
                "source_refs": ["transcript:seg_action"],
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
    payload.update(overrides)
    return payload


def create_client(
    response: dict[str, Any] | None = None,
    *,
    extraction_response: dict[str, Any] | None = None,
    speech_recognizer: FakeSpeechRecognizer | None = None,
    streaming_speech_recognizer: FakeStreamingSpeechRecognizer | None = None,
    auto_memory_snapshot: bool = True,
) -> TestClient:
    model_client = FakeModelClient(response or valid_prompt_response())
    prompt_service = PromptGenerationService(
        model_client=model_client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    memory_extraction_service = (
        MemoryExtractionService(
            model_client=FakeModelClient(extraction_response),
            settings=ModelGatewaySettings(default_model="gpt-memory-test"),
        )
        if extraction_response is not None
        else None
    )
    service = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=PromptRuntimeService(),
        memory_extraction_service=memory_extraction_service,
        speech_recognition_service=SpeechRecognitionService(speech_recognizer) if speech_recognizer is not None else None,
        streaming_speech_recognition_service=(
            StreamingSpeechRecognitionService(streaming_speech_recognizer)
            if streaming_speech_recognizer is not None
            else None
        ),
        auto_memory_snapshot=auto_memory_snapshot,
    )
    app = create_app(service)
    app.state.fake_model_client = model_client
    return TestClient(app)


def create_session(client: TestClient, session_id: str = "session_api_001") -> dict[str, Any]:
    response = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "config": {
                "title": "Launch risk sync",
                "pre_context": "讨论风险和负责人。",
                "privacy_constraints": [],
                "metadata": {
                    "org_id": "org_001",
                    "subject_user_id": "user_001",
                    "participants": ["Bao", "Alex"],
                    "scheduled_end_ms": 1800000,
                },
            },
        },
    )
    assert response.status_code == 201
    return response.json()


def append_gap_transcript(client: TestClient, session_id: str = "session_api_001") -> dict[str, Any]:
    response = client.post(
        f"/sessions/{session_id}/transcript",
        json={
            "segment_id": "seg_0",
            "segment": {
                "speaker": "Bao",
                "start_ms": 0,
                "end_ms": 900,
                "text": "这个问题谁负责，下周五 deadline 前能不能定？",
                "asr_confidence": 0.94,
            },
        },
    )
    assert response.status_code == 200
    return response.json()


def test_product_api_health() -> None:
    client = create_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "product-api"}


def test_create_and_get_session_with_meeting_state() -> None:
    client = create_client()

    created = create_session(client)

    assert created["session"]["session_id"] == "session_api_001"
    assert created["session"]["status"] == "running"
    assert created["meeting_state"]["org_id"] == "org_001"
    assert created["meeting_state"]["participants"] == ["Bao", "Alex"]

    fetched = client.get("/sessions/session_api_001")
    assert fetched.status_code == 200
    assert fetched.json()["session"]["session_id"] == "session_api_001"

    listed = client.get("/sessions")
    assert listed.status_code == 200
    assert [session["session_id"] for session in listed.json()["sessions"]] == ["session_api_001"]


def test_create_duplicate_session_returns_409() -> None:
    client = create_client()
    create_session(client)

    duplicate = client.post("/sessions", json={"session_id": "session_api_001"})

    assert duplicate.status_code == 409


def test_append_transcript_generates_prompt_and_updates_meeting_state() -> None:
    client = create_client()
    create_session(client)

    payload = append_gap_transcript(client)

    assert payload["transcript_segment"]["segment_id"] == "seg_0"
    assert payload["meeting_state"]["action_items"][0]["deadline"] == "下周五"
    assert {gap["gap_type"] for gap in payload["meeting_gaps"]} >= {"action_missing_owner"}
    assert payload["opportunity_count"] == 1
    assert payload["candidate_count"] == 1
    assert payload["prompts"][0]["should_display"] is True
    assert payload["prompts"][0]["glasses_text"] == "这个风险还没有明确 owner 和截止时间。"


def test_asr_transcribe_endpoint_returns_text_from_audio_payload() -> None:
    speech = FakeSpeechRecognizer(text="这个问题谁负责，下周五 deadline 前能不能定？")
    client = create_client(speech_recognizer=speech)

    response = client.post(
        "/asr/transcribe?language=zh-CN",
        content=b"fake-wav-bytes",
        headers={"content-type": "audio/wav"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["provider"] == "fake"
    assert payload["text"].startswith("这个问题谁负责")
    assert speech.requests[0]["content_type"] == "audio/wav"


def test_audio_transcript_endpoint_transcribes_appends_and_generates_prompt() -> None:
    speech = FakeSpeechRecognizer(text="这个问题谁负责，下周五 deadline 前能不能定？")
    client = create_client(speech_recognizer=speech)
    create_session(client)

    response = client.post(
        "/sessions/session_api_001/audio-transcript?speaker=Bao&start_ms=0&end_ms=900&segment_id=seg_audio_0",
        content=b"fake-wav-bytes",
        headers={"content-type": "audio/wav"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["transcription"]["text"].startswith("这个问题谁负责")
    assert payload["transcript_step"]["transcript_segment"]["segment_id"] == "seg_audio_0"
    assert payload["transcript_step"]["transcript_segment"]["source"] == "uploaded_audio_transcript"
    assert payload["transcript_step"]["prompts"][0]["prompt_category"] == "summary_gap_check"


def test_streaming_asr_websocket_appends_final_transcript_and_generates_prompt() -> None:
    streaming = FakeStreamingSpeechRecognizer(
        partial_text="这个问题谁负责",
        final_text="这个问题谁负责，下周五 deadline 前能不能定？",
    )
    client = create_client(streaming_speech_recognizer=streaming)
    create_session(client)

    with client.websocket_connect("/sessions/session_api_001/asr/stream?speaker=Bao&language=zh-CN") as websocket:
        opened = websocket.receive_json()
        websocket.send_bytes(b"\0" * 3200)
        websocket.send_json({"type": "stop"})
        events = [opened]
        for _ in range(6):
            event = websocket.receive_json()
            events.append(event)
            if event["type"] == "final_transcript":
                break

    preview = next(event for event in events if event["type"] == "prompt_preview")
    assert preview["soft_segment"]["text"].startswith("这个问题谁负责")
    assert preview["soft_segment"]["reason"] == "trigger_terms"
    assert preview["prompt_preview"]["transcript_segment"]["is_final"] is False
    assert preview["prompt_preview"]["prompts"][0]["prompt_category"] == "summary_gap_check"
    assert preview["prompt_preview"]["prompts"][0]["should_display"] is True

    final = next(event for event in events if event["type"] == "final_transcript")
    assert final["transcription"]["text"].startswith("这个问题谁负责")
    assert final["preview_reconciled"] is True
    assert final["transcript_step"]["transcript_segment"]["source"] == "live_asr_future"
    assert final["transcript_step"]["prompts"][0]["prompt_category"] == "summary_gap_check"
    assert final["transcript_step"]["prompts"][0]["should_display"] is True
    assert streaming.sessions[0].received_audio == [b"\0" * 3200]


def test_streaming_asr_websocket_returns_error_when_not_configured() -> None:
    client = create_client()
    create_session(client)

    with client.websocket_connect("/sessions/session_api_001/asr/stream?speaker=Bao") as websocket:
        event = websocket.receive_json()

    assert event["type"] == "error"
    assert "streaming speech recognition service is not configured" in event["detail"]


def test_asr_endpoint_returns_503_when_not_configured() -> None:
    client = create_client()

    response = client.post("/asr/transcribe", content=b"fake-wav-bytes", headers={"content-type": "audio/wav"})

    assert response.status_code == 503


def test_get_meeting_state_endpoint_after_transcript() -> None:
    client = create_client()
    create_session(client)
    append_gap_transcript(client)

    response = client.get("/sessions/session_api_001/meeting-state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session_api_001"
    assert payload["meeting_state"]["utterances"][0]["text"].startswith("这个问题谁负责")


def test_session_state_and_transcript_endpoints_return_product_view() -> None:
    client = create_client()
    create_session(client)
    step = append_gap_transcript(client)

    state = client.get("/sessions/session_api_001/state")
    transcript = client.get("/sessions/session_api_001/transcript")

    assert state.status_code == 200
    assert state.json()["session"]["session_id"] == "session_api_001"
    assert state.json()["transcript"][0]["segment_id"] == "seg_0"
    assert state.json()["prompts"][0]["decision_id"] == step["prompts"][0]["decision_id"]
    assert transcript.status_code == 200
    assert transcript.json()["transcript"][0]["text"].startswith("这个问题谁负责")


def test_session_lifecycle_endpoints_pause_resume_and_end() -> None:
    client = create_client()
    create_session(client)

    paused = client.post("/sessions/session_api_001/pause")
    resumed = client.post("/sessions/session_api_001/resume")
    ended = client.post("/sessions/session_api_001/end")

    assert paused.status_code == 200
    assert paused.json()["session"]["status"] == "paused"
    assert resumed.status_code == 200
    assert resumed.json()["session"]["status"] == "running"
    assert ended.status_code == 200
    assert ended.json()["session"]["status"] == "ended"
    append_after_end = client.post(
        "/sessions/session_api_001/transcript",
        json={
            "segment": {
                "speaker": "Bao",
                "start_ms": 1000,
                "end_ms": 1900,
                "text": "结束后不能继续追加。",
            }
        },
    )
    assert append_after_end.status_code == 400


def test_summary_endpoint_generates_app_summary_prompt() -> None:
    client = create_client(
        valid_prompt_response(
            content_granularity=3,
            glasses_title="会议总结",
            glasses_text="已整理关键结论、待办和未确认 GAP。",
            app_detail_text="总结：需要确认负责人、deadline 和下一步风险处理。",
            source_refs=["transcript:seg_0"],
        )
    )
    create_session(client)
    append_gap_transcript(client)

    response = client.post("/sessions/session_api_001/summary", json={"use_memory": False})

    payload = response.json()
    assert response.status_code == 200
    assert payload["session"]["session_id"] == "session_api_001"
    assert payload["prompts"][0]["prd_surface"] == "app_summary_tab"
    assert payload["prompts"][0]["glasses_title"] == "会议总结"
    assert payload["decisions"][0]["policy_version"] == "product_summary_v0"


def test_end_session_can_return_summary_payload() -> None:
    client = create_client(
        valid_prompt_response(
            content_granularity=3,
            glasses_title="会议总结",
            glasses_text="已整理关键结论、待办和未确认 GAP。",
            app_detail_text="总结：需要确认负责人、deadline 和下一步风险处理。",
            source_refs=["transcript:seg_0"],
        )
    )
    create_session(client)
    append_gap_transcript(client)

    response = client.post("/sessions/session_api_001/end", json={"generate_summary": True, "use_memory": False})

    payload = response.json()
    assert response.status_code == 200
    assert payload["session"]["status"] == "ended"
    assert payload["summary"]["prompts"][0]["prd_surface"] == "app_summary_tab"


def test_list_prompts_for_session() -> None:
    client = create_client()
    create_session(client)
    step = append_gap_transcript(client)

    response = client.get("/sessions/session_api_001/prompts")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session_api_001"
    assert payload["prompts"][0]["decision_id"] == step["prompts"][0]["decision_id"]


def test_record_feedback_computes_reward_and_memory_candidate() -> None:
    client = create_client()
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]

    response = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "accept", "event_id": "fb_accept"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["decision_id"] == decision_id
    assert payload["feedback_event"]["signal_type"] == "accept"
    assert payload["reward_observation"]["final_reward"] > 0
    assert payload["memory_candidates"][0]["candidate_type"] == "action_item"
    assert payload["memories"][0]["write_status"] == "pending_confirmation"

    candidates = client.get("/sessions/session_api_001/memory-candidates")
    assert candidates.status_code == 200
    assert candidates.json()["memory_candidates"][0]["memory_candidate_id"] == payload["memory_candidates"][0]["memory_candidate_id"]


def test_record_feedback_accepts_device_feedback_taxonomy() -> None:
    client = create_client()
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]

    response = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={
            "signal_type": "head_shake_reject",
            "event_id": "fb_head_shake",
            "input_channel": "gesture",
            "target": "timing",
            "display_strategy": "auto_popup",
            "propose_memory": False,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["feedback_event"]["signal_type"] == "head_shake_reject"
    assert payload["feedback_event"]["input_channel"] == "gesture"
    assert payload["feedback_event"]["target"] == "timing"
    assert payload["feedback_event"]["display_strategy"] == "auto_popup"
    assert payload["reward_observation"]["components"]["timing_fit"] < 0
    assert payload["reward_observation"]["final_reward"] < 0
    assert payload["memory_candidates"] == []


def test_record_manual_request_on_suppressed_prompt_returns_missed_opportunity_reward() -> None:
    client = create_client(
        {
            "should_prompt": False,
            "content_granularity": 0,
            "confidence": 0.31,
            "privacy_risk": 0.05,
            "rationale": "上下文不足，不主动打断。",
        }
    )
    create_session(client)
    step = client.post(
        "/sessions/session_api_001/transcript",
        json={
            "segment_id": "seg_0",
            "segment": {
                "speaker": "Bao",
                "start_ms": 0,
                "end_ms": 900,
                "text": "腾讯是哪一年成立的？",
                "asr_confidence": 0.94,
            },
        },
    ).json()
    decision_id = step["prompts"][0]["decision_id"]

    response = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={
            "signal_type": "manual_request",
            "event_id": "fb_manual_request",
            "input_channel": "touch",
            "display_strategy": "manual_response",
            "propose_memory": False,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["feedback_event"]["target"] == "timing"
    assert payload["feedback_event"]["input_channel"] == "touch"
    assert payload["feedback_event"]["display_strategy"] == "manual_response"
    assert payload["reward_observation"]["components"]["missed_opportunity"] == 1.0
    assert payload["reward_observation"]["final_reward"] < -1.0


def test_policy_episode_endpoint_exports_session_steps() -> None:
    client = create_client()
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]
    feedback = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "nod_accept", "event_id": "fb_nod", "propose_memory": False},
    ).json()

    response = client.get("/sessions/session_api_001/policy-episode")

    payload = response.json()
    assert response.status_code == 200
    assert payload["session"]["session_id"] == "session_api_001"
    assert payload["episode"]["session_id"] == "session_api_001"
    assert payload["episode"]["feedback_event_count"] == 1
    assert payload["episode"]["rewarded_step_count"] == 1
    assert payload["episode"]["total_reward"] == feedback["reward_observation"]["final_reward"]
    assert payload["episode"]["steps"][0]["decision_id"] == decision_id
    assert payload["episode"]["steps"][0]["state"]["timing_action"] == "during_activity"
    assert payload["episode"]["steps"][0]["action"]["display_strategy"] == "auto_popup"
    assert payload["episode"]["steps"][0]["feedback_events"][0]["signal_type"] == "nod_accept"


def test_policy_evaluation_endpoint_exports_session_metrics() -> None:
    client = create_client()
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]
    feedback = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "head_shake_reject", "event_id": "fb_head_shake", "propose_memory": False},
    ).json()

    response = client.get("/sessions/session_api_001/policy-evaluation")

    payload = response.json()
    assert response.status_code == 200
    assert payload["session"]["session_id"] == "session_api_001"
    assert payload["report"]["session_id"] == "session_api_001"
    assert payload["report"]["summary"]["step_count"] == 1
    assert payload["report"]["summary"]["negative_feedback_rate"] == 1.0
    assert payload["report"]["summary"]["total_reward"] == feedback["reward_observation"]["final_reward"]
    assert payload["report"]["by_prompt_category"][0]["value"] == "summary_gap_check"
    assert payload["report"]["by_display_strategy"][0]["value"] == "auto_popup"


def test_policy_baselines_endpoint_exports_baseline_audit() -> None:
    client = create_client(
        {
            "should_prompt": False,
            "content_granularity": 0,
            "confidence": 0.31,
            "privacy_risk": 0.05,
            "rationale": "当前先静默。",
        }
    )
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]
    client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "manual_request", "event_id": "fb_manual", "propose_memory": False},
    )

    response = client.get("/sessions/session_api_001/policy-baselines")

    payload = response.json()
    assert response.status_code == 200
    assert payload["session"]["session_id"] == "session_api_001"
    assert payload["report"]["session_id"] == "session_api_001"
    assert {baseline["baseline_name"] for baseline in payload["report"]["baselines"]} == {
        "conservative",
        "balanced",
        "aggressive",
    }
    balanced = [baseline for baseline in payload["report"]["baselines"] if baseline["baseline_name"] == "balanced"][0]
    assert balanced["metrics"]["missed_opportunity_coverage_rate"] == 1.0
    assert balanced["decisions"][0]["missed_opportunity_covered"] is True


def test_policy_export_endpoint_exports_training_examples() -> None:
    client = create_client()
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]
    feedback = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "nod_accept", "event_id": "fb_export_nod", "propose_memory": False},
    ).json()

    response = client.get("/sessions/session_api_001/policy-export")

    payload = response.json()
    example = payload["export"]["examples"][0]
    assert response.status_code == 200
    assert payload["session"]["session_id"] == "session_api_001"
    assert payload["export"]["session_id"] == "session_api_001"
    assert payload["export"]["example_count"] == 1
    assert payload["export"]["evaluation_summary"]["total_reward"] == feedback["reward_observation"]["final_reward"]
    assert example["decision_id"] == decision_id
    assert example["label"]["accepted"] is True
    assert example["label"]["final_reward"] == feedback["reward_observation"]["final_reward"]
    assert len(example["baseline_decisions"]) == 3


def test_confirmed_memory_context_is_returned_and_used_by_transcript_endpoint() -> None:
    client = create_client(auto_memory_snapshot=False)
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]
    feedback = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "accept", "event_id": "fb_accept"},
    ).json()
    memory_id = feedback["memories"][0]["memory_id"]

    empty_context = client.get("/sessions/session_api_001/memory-context?query_text=owner%20deadline")
    assert empty_context.status_code == 200
    assert empty_context.json()["memory_context"]["memory_refs"] == []

    confirmed = client.post(f"/memories/{memory_id}/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["memory"]["write_status"] == "active"

    context = client.get("/sessions/session_api_001/memory-context?query_text=owner%20deadline")
    assert context.status_code == 200
    assert context.json()["memory_context"]["memory_refs"] == [f"memory:{memory_id}"]

    second_step = client.post(
        "/sessions/session_api_001/transcript",
        json={
            "segment_id": "seg_1",
            "segment": {
                "speaker": "Bao",
                "start_ms": 1000,
                "end_ms": 1800,
                "text": "owner 和 deadline 还有谁需要确认？",
                "asr_confidence": 0.94,
            },
        },
    )
    assert second_step.status_code == 200
    assert second_step.json()["retrieved_memory_context"]["memory_refs"] == [f"memory:{memory_id}"]


def test_memory_snapshot_endpoint_can_preview_and_commit_meeting_state_memories() -> None:
    client = create_client(auto_memory_snapshot=False)
    create_session(client)
    append_gap_transcript(client)

    preview = client.post("/sessions/session_api_001/memory-snapshot", json={"commit": False})
    committed = client.post("/sessions/session_api_001/memory-snapshot")
    repeated = client.post("/sessions/session_api_001/memory-snapshot")

    assert preview.status_code == 200
    assert preview.json()["committed"] is False
    assert preview.json()["memory_candidates"]
    assert preview.json()["memories"] == []
    assert committed.status_code == 200
    assert committed.json()["committed"] is True
    assert committed.json()["memory_candidates"]
    assert committed.json()["memories"]
    assert committed.json()["memory_upserts"][0]["status"] == "created"
    assert committed.json()["memories"][0]["metadata"]["memory_snapshot_source"] == "meeting_state_snapshot_v1"
    assert repeated.status_code == 200
    assert repeated.json()["memory_upserts"][0]["status"] == "unchanged"


def test_memory_extraction_endpoint_extracts_and_commits_llm_candidates() -> None:
    client = create_client(extraction_response=valid_memory_extraction_response(), auto_memory_snapshot=False)
    create_session(client)
    transcript_response = client.post(
        "/sessions/session_api_001/transcript",
        json={
            "segment_id": "seg_action",
            "segment": {
                "speaker": "Bao",
                "start_ms": 0,
                "end_ms": 900,
                "text": "张三负责客户报价确认，下周五截止。",
                "asr_confidence": 0.94,
            },
        },
    )
    assert transcript_response.status_code == 200

    response = client.post(
        "/sessions/session_api_001/memory-extraction",
        json={"commit": True, "max_segments": 12, "max_candidates": 4},
    )
    context = client.get("/sessions/session_api_001/memory-context?query_text=客户报价确认&include_pending=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["committed"] is True
    assert payload["memory_candidates"][0]["candidate_type"] == "action_item"
    assert payload["memory_candidates"][0]["metadata"]["owner"] == "张三"
    assert payload["memory_upserts"][0]["status"] == "created"
    assert payload["memories"][0]["metadata"]["memory_extraction_source"] == "llm_memory_extraction_v1"
    assert payload["model_usage"]["model"] == "gpt-memory-test"
    assert context.status_code == 200
    assert context.json()["memory_context"]["memory_refs"] == [f"memory:{payload['memories'][0]['memory_id']}"]


def test_memory_pending_update_endpoints_apply_sensitive_snapshot_update() -> None:
    client = create_client()
    create_session(client)
    transcript_response = client.post(
        "/sessions/session_api_001/transcript",
        json={
            "segment_id": "seg_action",
            "segment": {
                "speaker": "Bao",
                "start_ms": 0,
                "end_ms": 900,
                "text": "张三负责客户报价确认，下周五截止。",
                "asr_confidence": 0.94,
            },
        },
    )
    assert transcript_response.status_code == 200
    committed = client.post("/sessions/session_api_001/memory-snapshot", json={"include_gaps": False})
    memory_id = committed.json()["memories"][0]["memory_id"]

    service = client.app.state.product_service
    state = service.get_meeting_state("session_api_001")
    action_item = state.action_items[0]
    service._meeting_states["session_api_001"] = state.model_copy(
        update={
            "action_items": [
                action_item.model_copy(
                    update={
                        "owner": "李四",
                        "deadline": "下周一",
                        "desc": "李四负责客户报价确认，下周一截止。",
                        "evidence": "李四负责客户报价确认，下周一截止。",
                    }
                )
            ]
        }
    )

    proposed = client.post("/sessions/session_api_001/memory-snapshot", json={"include_gaps": False})
    pending_update = proposed.json()["memory_upserts"][0]["pending_update"]
    listed = client.get(f"/memories/{memory_id}/pending-updates")
    applied = client.post(f"/memory-updates/{pending_update['update_id']}/apply", json={"reason": "confirmed in test"})
    remaining = client.get(f"/memories/{memory_id}/pending-updates")

    assert proposed.status_code == 200
    assert proposed.json()["memory_upserts"][0]["update_policy"] == "needs_confirmation"
    assert pending_update["status"] == "pending"
    assert listed.status_code == 200
    assert listed.json()["pending_updates"][0]["update_id"] == pending_update["update_id"]
    assert applied.status_code == 200
    applied_memory = applied.json()["memory_upsert"]["memory"]
    assert applied.json()["memory_upsert"]["status"] == "updated"
    assert applied.json()["memory_upsert"]["pending_update"]["status"] == "applied"
    assert applied_memory["metadata"]["owner"] == "李四"
    assert applied_memory["metadata"]["deadline"] == "下周一"
    assert remaining.status_code == 200
    assert remaining.json()["pending_updates"] == []


def test_memory_delete_endpoint_forgets_memory_and_invalidates_pending_update() -> None:
    client = create_client()
    create_session(client)
    transcript_response = client.post(
        "/sessions/session_api_001/transcript",
        json={
            "segment_id": "seg_action",
            "segment": {
                "speaker": "Bao",
                "start_ms": 0,
                "end_ms": 900,
                "text": "张三负责客户报价确认，下周五截止。",
                "asr_confidence": 0.94,
            },
        },
    )
    assert transcript_response.status_code == 200
    committed = client.post("/sessions/session_api_001/memory-snapshot", json={"include_gaps": False})
    memory_id = committed.json()["memories"][0]["memory_id"]

    service = client.app.state.product_service
    state = service.get_meeting_state("session_api_001")
    action_item = state.action_items[0]
    service._meeting_states["session_api_001"] = state.model_copy(
        update={
            "action_items": [
                action_item.model_copy(
                    update={
                        "owner": "李四",
                        "deadline": "下周一",
                        "desc": "李四负责客户报价确认，下周一截止。",
                        "evidence": "李四负责客户报价确认，下周一截止。",
                    }
                )
            ]
        }
    )
    proposed = client.post("/sessions/session_api_001/memory-snapshot", json={"include_gaps": False})
    pending_update = proposed.json()["memory_upserts"][0]["pending_update"]

    deleted = client.delete(f"/memories/{memory_id}", params={"reason": "user requested deletion"})
    context = client.get("/sessions/session_api_001/memory-context?query_text=客户报价确认")
    rejected_updates = client.get(f"/memories/{memory_id}/pending-updates?status=rejected")
    stale_apply = client.post(f"/memory-updates/{pending_update['update_id']}/apply")

    assert deleted.status_code == 200
    payload = deleted.json()["memory_forget"]
    assert payload["memory"]["write_status"] == "forgotten"
    assert payload["memory"]["text"] == "[forgotten]"
    assert payload["memory"]["metadata"]["forget_reason"] == "user requested deletion"
    assert payload["invalidated_pending_updates"][0]["update_id"] == pending_update["update_id"]
    assert payload["invalidated_pending_updates"][0]["status"] == "rejected"
    assert context.status_code == 200
    assert context.json()["memory_context"]["memory_refs"] == []
    assert rejected_updates.status_code == 200
    assert rejected_updates.json()["pending_updates"][0]["resolved_reason"] == "memory forgotten: user requested deletion"
    assert stale_apply.status_code == 409


def test_memory_promote_endpoint_makes_approved_memory_visible_to_later_session() -> None:
    client = create_client()
    create_session(client)
    transcript_response = client.post(
        "/sessions/session_api_001/transcript",
        json={
            "segment_id": "seg_action",
            "segment": {
                "speaker": "Bao",
                "start_ms": 0,
                "end_ms": 900,
                "text": "张三负责客户报价确认，下周五截止。",
                "asr_confidence": 0.94,
            },
        },
    )
    assert transcript_response.status_code == 200
    snapshot = client.post("/sessions/session_api_001/memory-snapshot", json={"include_gaps": False})
    memory_id = snapshot.json()["memories"][0]["memory_id"]

    proposed = client.post(
        f"/memories/{memory_id}/promote",
        json={"target_scope": "org", "reason": "action item needs review"},
    )
    approved = client.post(
        f"/memories/{memory_id}/promote",
        json={"target_scope": "org", "reason": "approved for recurring project context", "approved": True},
    )
    create_session(client, session_id="session_api_002")
    context = client.get("/sessions/session_api_002/memory-context?query_text=客户报价确认")

    assert proposed.status_code == 200
    assert proposed.json()["memory_promotion"]["status"] == "needs_confirmation"
    assert proposed.json()["memory_promotion"]["promoted_memory"] is None
    assert approved.status_code == 200
    assert approved.json()["memory_promotion"]["status"] == "promoted"
    promoted_memory = approved.json()["memory_promotion"]["promoted_memory"]
    assert promoted_memory["scope"] == "org"
    assert promoted_memory["source"] == "promoted"
    assert context.status_code == 200
    assert context.json()["memory_context"]["memory_refs"] == [f"memory:{promoted_memory['memory_id']}"]


def test_reject_and_archive_memory_endpoints_hide_memory_context() -> None:
    client = create_client(auto_memory_snapshot=False)
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]
    feedback = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "privacy_reject", "event_id": "fb_privacy"},
    ).json()
    memory_id = feedback["memories"][0]["memory_id"]

    assert client.get("/sessions/session_api_001/memory-context?query_text=privacy").json()["memory_context"]["memory_refs"] == [
        f"memory:{memory_id}"
    ]

    rejected = client.post(f"/memories/{memory_id}/reject", json={"reason": "user correction"})
    assert rejected.status_code == 200
    assert rejected.json()["memory"]["write_status"] == "rejected"
    assert rejected.json()["memory"]["metadata"]["reject_reason"] == "user correction"
    assert client.get("/sessions/session_api_001/memory-context?query_text=privacy").json()["memory_context"]["memory_refs"] == []

    second = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "switch_to_manual", "event_id": "fb_manual"},
    ).json()
    second_memory_id = second["memories"][0]["memory_id"]
    archived = client.post(f"/memories/{second_memory_id}/archive", json={"reason": "superseded"})
    assert archived.status_code == 200
    assert archived.json()["memory"]["write_status"] == "archived"


def test_unknown_session_and_decision_return_404() -> None:
    client = create_client()

    session_state_response = client.get("/sessions/missing/state")
    state_response = client.get("/sessions/missing/meeting-state")
    transcript_response = client.get("/sessions/missing/transcript")
    pause_response = client.post("/sessions/missing/pause")
    summary_response = client.post("/sessions/missing/summary")
    evaluation_response = client.get("/sessions/missing/policy-evaluation")
    baseline_response = client.get("/sessions/missing/policy-baselines")
    export_response = client.get("/sessions/missing/policy-export")
    feedback_response = client.post(
        "/prompt-decisions/missing/feedback",
        json={"signal_type": "accept"},
    )
    memory_response = client.post("/memories/missing/confirm")

    assert session_state_response.status_code == 404
    assert state_response.status_code == 404
    assert transcript_response.status_code == 404
    assert pause_response.status_code == 404
    assert summary_response.status_code == 404
    assert evaluation_response.status_code == 404
    assert baseline_response.status_code == 404
    assert export_response.status_code == 404
    assert feedback_response.status_code == 404
    assert memory_response.status_code == 404


def test_duplicate_feedback_event_returns_409() -> None:
    client = create_client()
    create_session(client)
    step = append_gap_transcript(client)
    decision_id = step["prompts"][0]["decision_id"]

    assert client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "accept", "event_id": "fb_accept"},
    ).status_code == 200
    duplicate = client.post(
        f"/prompt-decisions/{decision_id}/feedback",
        json={"signal_type": "accept", "event_id": "fb_accept"},
    )

    assert duplicate.status_code == 409


def test_warmup_memory_endpoint_returns_cached_context_after_session_creation() -> None:
    """The warmup endpoint surfaces the cross-session memory that was
    retrieved at create_session time."""

    from proactive_assistant.memory import (
        InMemoryMemoryStore,
        MemoryRecord,
        MemoryScope,
        MemoryService,
        MemorySource,
        MemoryType,
    )

    memory_service = MemoryService(InMemoryMemoryStore())
    memory_service.store.add_memory(
        MemoryRecord(
            memory_id="mem_atlas_api",
            memory_type=MemoryType.PROJECT_CONTEXT,
            scope=MemoryScope.USER,
            text="Project Atlas 上周决定推迟到 Q4 发布。",
            org_id="org_001",
            user_id="user_001",
            source=MemorySource.MANUAL,
            confidence=0.9,
            importance=0.8,
            tags=["project_context"],
            metadata={"canonical_entity": "Project Atlas"},
        )
    )

    client_payload = FakeModelClient({"should_prompt": False, "content_granularity": 0, "confidence": 0.0, "privacy_level": "low", "privacy_risk": 0.0, "source_refs": []})
    prompt_service = PromptGenerationService(
        model_client=client_payload,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    service = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=PromptRuntimeService(),
        memory_service=memory_service,
    )
    app = create_app(service)
    client = TestClient(app)

    create_response = client.post(
        "/sessions",
        json={
            "session_id": "session_warmup_001",
            "config": {
                "title": "Project Atlas weekly sync",
                "metadata": {
                    "org_id": "org_001",
                    "subject_user_id": "user_001",
                    "project": "Project Atlas",
                },
            },
        },
    )
    assert create_response.status_code in (200, 201)

    warmup_response = client.get("/sessions/session_warmup_001/warmup-memory")
    assert warmup_response.status_code == 200
    body = warmup_response.json()

    assert body["session_id"] == "session_warmup_001"
    ids = [result["memory"]["memory_id"] for result in body["memory_context"]["results"]]
    assert "mem_atlas_api" in ids


def test_warmup_memory_endpoint_returns_404_for_unknown_session() -> None:
    client = create_client()
    response = client.get("/sessions/does_not_exist/warmup-memory")
    assert response.status_code == 404
