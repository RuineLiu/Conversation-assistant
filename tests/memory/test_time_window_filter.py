from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryType,
)
from proactive_assistant.memory.query_understanding import QueryUnderstandingService
from proactive_assistant.memory.retrieval import MemoryRetriever
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import PromptCategory


def test_time_window_filter_matches_event_fully_contained_in_query_window() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(_memory("mem_review", {"normalized_start_time": "2026-06-10", "normalized_end_time": "2026-06-12"}))

    results = store.list_memories(
        MemoryQuery(
            org_id="org_001",
            user_id="user_001",
            time_window_start="2026-06-09",
            time_window_end="2026-06-13",
        )
    )

    assert [item.memory_id for item in results] == ["mem_review"]


def test_time_window_filter_matches_partially_overlapping_event() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(_memory("mem_review", {"normalized_start_time": "2026-06-10", "normalized_end_time": "2026-06-12"}))

    results = store.list_memories(
        MemoryQuery(
            org_id="org_001",
            user_id="user_001",
            time_window_start="2026-06-12",
            time_window_end="2026-06-14",
        )
    )

    assert [item.memory_id for item in results] == ["mem_review"]


def test_time_window_filter_excludes_event_outside_query_window() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(_memory("mem_review", {"normalized_start_time": "2026-06-10", "normalized_end_time": "2026-06-12"}))

    results = store.list_memories(
        MemoryQuery(
            org_id="org_001",
            user_id="user_001",
            time_window_start="2026-06-13",
            time_window_end="2026-06-14",
        )
    )

    assert results == []


def test_time_window_filter_matches_undefined_end_after_event_start() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(_memory("mem_ongoing", {"normalized_start_time": "2026-06-10", "normalized_end_time": "未定"}))

    results = store.list_memories(
        MemoryQuery(
            org_id="org_001",
            user_id="user_001",
            time_window_start="2026-06-15",
            time_window_end="2026-06-16",
        )
    )

    assert [item.memory_id for item in results] == ["mem_ongoing"]


def test_query_without_time_window_does_not_filter_event_metadata() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(_memory("mem_review", {"normalized_start_time": "2026-06-10", "normalized_end_time": "2026-06-12"}))
    store.add_memory(_memory("mem_no_window", {"entity": "客户报价"}))

    results = store.list_memories(MemoryQuery(org_id="org_001", user_id="user_001", limit=10))

    assert {item.memory_id for item in results} == {"mem_review", "mem_no_window"}


def test_query_understanding_time_window_filters_schedule_retrieval() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(
        _memory(
            "mem_june_review",
            {
                "normalized_start_time": "2026-06-15",
                "normalized_end_time": "2026-06-15",
                "entity": "Project Atlas",
            },
            text="Project Atlas 6 月评审安排在 2026-06-15。",
        )
    )
    store.add_memory(
        _memory(
            "mem_july_review",
            {
                "normalized_start_time": "2026-07-01",
                "normalized_end_time": "2026-07-01",
                "entity": "Project Atlas",
            },
            text="Project Atlas 7 月评审安排在 2026-07-01。",
        )
    )

    retriever = MemoryRetriever(
        store,
        query_understanding=_query_understanding_service(
            {
                "intent": "lookup_schedule",
                "target_entity": "Project Atlas",
                "target_entity_confidence": 0.92,
                "anaphora_resolved": False,
                "time_window_start": "2026-06-15",
                "time_window_end": "2026-06-15",
                "time_is_relative": False,
                "referenced_speaker": "",
                "confidence": 0.91,
                "rationale": "用户询问指定时间窗口内的日程。",
            }
        ),
    )

    context = retriever.retrieve(
        MemoryQuery(
            query_text="Project Atlas 6 月评审是哪天？",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            prompt_category=PromptCategory.QUESTION_ANSWER,
            limit=10,
            reference_time=datetime(2026, 6, 10, tzinfo=UTC),
        )
    )

    assert context.memory_refs == ["memory:mem_june_review"]


def _query_understanding_service(response: dict[str, Any]) -> QueryUnderstandingService:
    return QueryUnderstandingService(
        model_client=FakeModelClient(response),
        settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
    )


def _memory(
    memory_id: str,
    metadata: dict[str, object],
    *,
    text: str = "Project Atlas 评审安排。",
) -> MemoryRecord:
    created = datetime(2026, 6, 5, tzinfo=UTC)
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=MemoryType.MEETING_FACT,
        scope=MemoryScope.USER,
        text=text,
        org_id="org_001",
        user_id="user_001",
        session_id=None,
        source=MemorySource.MEETING_STATE,
        source_ids=[f"meeting:{memory_id}"],
        confidence=0.9,
        importance=0.8,
        created_at=created,
        updated_at=created,
        metadata=metadata,
    )


def test_memory_with_only_deadline_visible_to_time_window_query() -> None:
    """P2-1: memories written outside the consolidation path may carry
    only normalized_deadline (no event window). They must still match
    time-bounded queries via the single-day fallback window."""

    store = InMemoryMemoryStore()
    store.add_memory(
        _memory(
            "mem_deadline_only",
            {
                "normalized_deadline": "2026-06-20",
                "entity": "客户报价",
            },
            text="张三负责客户报价确认，deadline 2026-06-20。",
        )
    )

    matched = store.list_memories(
        MemoryQuery(
            session_id="session_001",
            time_window_start="2026-06-01",
            time_window_end="2026-06-30",
            limit=10,
        )
    )
    excluded = store.list_memories(
        MemoryQuery(
            session_id="session_001",
            time_window_start="2026-07-01",
            time_window_end="2026-07-31",
            limit=10,
        )
    )

    assert [m.memory_id for m in matched] == ["mem_deadline_only"]
    assert excluded == []
