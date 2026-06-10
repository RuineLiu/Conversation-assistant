from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from proactive_assistant.memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemorySource,
    MemoryType,
)
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.persistence import SQLiteMemoryStore, SQLiteRuntimeStore, SQLiteSessionStore
from proactive_assistant.product import ProductAssistantService
from proactive_assistant.product.api import _build_storage_services
from proactive_assistant.prompting import PromptGenerationService
from proactive_assistant.repositories import MemoryRepository, RuntimeRepository, SessionRepository
from proactive_assistant.runtime import FeedbackSignalType, PromptRuntimeService
from proactive_assistant.runtime import MemoryCandidate, MemoryCandidateType, MemoryWritePolicy
from proactive_assistant.sessions import SessionConfig, SessionService, TranscriptSegmentInput


def test_sqlite_stores_satisfy_repository_protocols(tmp_path: Path) -> None:
    db_path = tmp_path / "proactive.db"

    assert isinstance(SQLiteSessionStore(db_path), SessionRepository)
    assert isinstance(SQLiteRuntimeStore(db_path), RuntimeRepository)
    assert isinstance(SQLiteMemoryStore(db_path), MemoryRepository)


def test_product_api_storage_env_switch_builds_sqlite_services(tmp_path: Path, monkeypatch: Any) -> None:
    db_path = tmp_path / "env_switch.db"
    monkeypatch.setenv("PROACTIVE_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("PROACTIVE_SQLITE_PATH", str(db_path))

    session_store, runtime_service, memory_service = _build_storage_services()

    assert isinstance(session_store, SQLiteSessionStore)
    assert isinstance(runtime_service.store, SQLiteRuntimeStore)
    assert memory_service is not None
    assert isinstance(memory_service.store, SQLiteMemoryStore)


def test_sqlite_session_store_persists_sessions_and_transcripts(tmp_path: Path) -> None:
    db_path = tmp_path / "proactive.db"
    service = SessionService(SQLiteSessionStore(db_path))
    service.create_session(SessionConfig(title="Launch risk sync"), session_id="session_sqlite_001")
    service.append_transcript(
        "session_sqlite_001",
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=900,
            text="这个问题谁负责，下周五 deadline 前能不能定？",
            asr_confidence=0.94,
        ),
        segment_id="seg_0",
    )

    reopened = SessionService(SQLiteSessionStore(db_path))

    assert reopened.get_session("session_sqlite_001").title == "Launch risk sync"
    transcript = reopened.get_transcript("session_sqlite_001")
    assert [segment.segment_id for segment in transcript] == ["seg_0"]
    assert transcript[0].text.startswith("这个问题谁负责")


def test_sqlite_memory_store_persists_and_searches_after_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "proactive.db"
    store = SQLiteMemoryStore(db_path)
    store.add_memory(
        MemoryRecord(
            memory_id="mem_action_001",
            memory_type=MemoryType.ACTION_ITEM,
            scope=MemoryScope.SESSION,
            text="Alex owns the launch deadline follow-up.",
            org_id="org_001",
            user_id="user_001",
            session_id="session_sqlite_001",
            source=MemorySource.MANUAL,
            source_ids=["manual:fixture"],
            confidence=0.9,
            importance=0.9,
            tags=["owner", "deadline"],
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
    )

    reopened = SQLiteMemoryStore(db_path)
    results = reopened.search_memories(
        MemoryQuery(
            org_id="org_001",
            user_id="user_001",
            session_id="session_sqlite_001",
            query_text="owner deadline",
            limit=10,
        )
    )

    assert [result.memory.memory_id for result in results] == ["mem_action_001"]
    assert results[0].matched_terms == ["deadline", "owner"]


def test_sqlite_memory_pending_update_persists_and_applies_after_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "proactive.db"
    service = MemoryService(SQLiteMemoryStore(db_path))
    created = service.upsert_candidate(
        _candidate(
            "张三负责客户报价确认，下周五截止。",
            metadata={"owner": "张三", "deadline": "下周五", "tags": ["meeting_state", "action_item"]},
        )
    )
    pending = service.upsert_candidate(
        _candidate(
            "李四负责客户报价确认，下周一截止。",
            metadata={"owner": "李四", "deadline": "下周一", "tags": ["meeting_state", "action_item"]},
        )
    )
    assert pending.pending_update is not None

    reopened = MemoryService(SQLiteMemoryStore(db_path))
    listed = reopened.list_pending_updates(memory_id=created.memory.memory_id)
    applied = reopened.apply_pending_update(pending.pending_update.update_id, reason="confirmed after reopen")

    assert [item.update_id for item in listed] == [pending.pending_update.update_id]
    assert applied.memory.memory_id == created.memory.memory_id
    assert applied.memory.metadata["owner"] == "李四"
    assert applied.pending_update is not None
    assert applied.pending_update.status == "applied"
    assert applied.pending_update.resolved_reason == "confirmed after reopen"


def test_sqlite_memory_forget_persists_redaction_and_pending_invalidation(tmp_path: Path) -> None:
    db_path = tmp_path / "proactive.db"
    service = MemoryService(SQLiteMemoryStore(db_path))
    created = service.upsert_candidate(
        _candidate(
            "张三负责客户报价确认，下周五截止。",
            metadata={"owner": "张三", "deadline": "下周五", "tags": ["meeting_state", "action_item"]},
        )
    )
    pending = service.upsert_candidate(
        _candidate(
            "李四负责客户报价确认，下周一截止。",
            metadata={"owner": "李四", "deadline": "下周一", "tags": ["meeting_state", "action_item"]},
        )
    )
    assert pending.pending_update is not None

    forgotten = service.forget_memory(created.memory.memory_id, reason="privacy deletion")
    reopened = MemoryService(SQLiteMemoryStore(db_path))
    stored_tombstones = reopened.store.list_memories(MemoryQuery(include_forgotten=True, limit=10))
    active_context = reopened.search_context(
        MemoryQuery(
            org_id="default_org",
            user_id="default_user",
            session_id="session_sqlite_001",
            query_text="客户报价确认",
            limit=10,
        )
    )
    updates = reopened.list_pending_updates(memory_id=created.memory.memory_id, status=None)

    assert forgotten.memory.write_status == "forgotten"
    assert forgotten.invalidated_pending_updates[0].status == "rejected"
    assert stored_tombstones[0].memory_id == created.memory.memory_id
    assert stored_tombstones[0].text == "[forgotten]"
    assert stored_tombstones[0].metadata["forget_reason"] == "privacy deletion"
    assert active_context.memory_refs == []
    assert updates[0].status == "rejected"
    assert updates[0].resolved_reason == "memory forgotten: privacy deletion"


def test_sqlite_product_flow_persists_runtime_and_memory_after_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "proactive.db"
    service = _product_service(db_path)
    service.create_session(
        SessionConfig(
            title="Launch risk sync",
            metadata={
                "org_id": "org_001",
                "subject_user_id": "user_001",
                "participants": ["Bao", "Alex"],
            },
        ),
        session_id="session_sqlite_001",
    )
    step = service.append_transcript_and_generate_prompts(
        "session_sqlite_001",
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=900,
            text="这个问题谁负责，下周五 deadline 前能不能定？",
            asr_confidence=0.94,
        ),
        segment_id="seg_0",
    )
    feedback = service.record_feedback(
        step.prompts[0].decision_id,
        FeedbackSignalType.ACCEPT,
        event_id="fb_accept",
    )
    memory = service.confirm_memory(feedback.memories[0].memory_id)

    reopened = _product_service(db_path)

    assert reopened.sessions.get_session("session_sqlite_001").title == "Launch risk sync"
    assert [decision.decision_id for decision in reopened.list_prompt_decisions(session_id="session_sqlite_001")] == [
        step.prompts[0].decision_id
    ]
    assert [candidate.memory_candidate_id for candidate in reopened.list_memory_candidates(session_id="session_sqlite_001")] == [
        feedback.memory_candidates[0].memory_candidate_id
    ]
    context = reopened.search_memory_context_for_session(
        "session_sqlite_001",
        query_text="owner deadline",
        limit=10,
    )
    assert context.memory_refs == [f"memory:{memory.memory_id}"]


def _product_service(db_path: Path, *, auto_memory_snapshot: bool = False) -> ProductAssistantService:
    prompt_service = PromptGenerationService(
        model_client=FakeModelClient(_valid_prompt_response()),
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    return ProductAssistantService(
        session_service=SessionService(SQLiteSessionStore(db_path)),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=PromptRuntimeService(SQLiteRuntimeStore(db_path)),
        memory_service=MemoryService(SQLiteMemoryStore(db_path)),
        auto_memory_snapshot=auto_memory_snapshot,
    )


def _candidate(text: str, *, metadata: dict[str, Any]) -> MemoryCandidate:
    return MemoryCandidate(
        memory_candidate_id="memcand_sqlite_action",
        decision_id="decision_sqlite_snapshot",
        session_id="session_sqlite_001",
        source_event_ids=[],
        candidate_type=MemoryCandidateType.ACTION_ITEM,
        text=text,
        confidence=0.8,
        write_policy=MemoryWritePolicy.ELIGIBLE,
        reason="sqlite pending update test",
        metadata=metadata,
    )


def _valid_prompt_response(**overrides: Any) -> dict[str, Any]:
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
