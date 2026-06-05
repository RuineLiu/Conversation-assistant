from typing import Any

from fastapi.testclient import TestClient

from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product.api import create_app
from proactive_assistant.product.service import ProductAssistantService
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


def create_client(response: dict[str, Any] | None = None) -> TestClient:
    model_client = FakeModelClient(response or valid_prompt_response())
    prompt_service = PromptGenerationService(
        model_client=model_client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    service = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=PromptRuntimeService(),
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


def test_get_meeting_state_endpoint_after_transcript() -> None:
    client = create_client()
    create_session(client)
    append_gap_transcript(client)

    response = client.get("/sessions/session_api_001/meeting-state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session_api_001"
    assert payload["meeting_state"]["utterances"][0]["text"].startswith("这个问题谁负责")


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


def test_confirmed_memory_context_is_returned_and_used_by_transcript_endpoint() -> None:
    client = create_client()
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


def test_reject_and_archive_memory_endpoints_hide_memory_context() -> None:
    client = create_client()
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

    state_response = client.get("/sessions/missing/meeting-state")
    feedback_response = client.post(
        "/prompt-decisions/missing/feedback",
        json={"signal_type": "accept"},
    )
    memory_response = client.post("/memories/missing/confirm")

    assert state_response.status_code == 404
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
