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


def _product_service(db_path: Path) -> ProductAssistantService:
    prompt_service = PromptGenerationService(
        model_client=FakeModelClient(_valid_prompt_response()),
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    return ProductAssistantService(
        session_service=SessionService(SQLiteSessionStore(db_path)),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=PromptRuntimeService(SQLiteRuntimeStore(db_path)),
        memory_service=MemoryService(SQLiteMemoryStore(db_path)),
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
