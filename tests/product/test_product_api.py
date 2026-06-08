from typing import Any

from fastapi.testclient import TestClient

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


def test_memory_snapshot_endpoint_can_preview_and_commit_meeting_state_memories() -> None:
    client = create_client()
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
    client = create_client(extraction_response=valid_memory_extraction_response())
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
