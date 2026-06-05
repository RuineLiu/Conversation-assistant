from __future__ import annotations

from datetime import UTC, datetime, timedelta

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalIntent,
    MemoryScope,
    MemoryService,
    MemorySource,
    MemoryType,
    MemoryUsePolicy,
    MemoryWriteStatus,
)
from proactive_assistant.prompting import PRDSurface, PrivacyLevel, PromptCategory


def memory(
    memory_id: str,
    text: str,
    *,
    memory_type: MemoryType = MemoryType.MEETING_FACT,
    scope: MemoryScope = MemoryScope.SESSION,
    session_id: str = "session_001",
    write_status: MemoryWriteStatus = MemoryWriteStatus.ACTIVE,
    privacy_level: PrivacyLevel = PrivacyLevel.LOW,
    metadata: dict[str, object] | None = None,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> MemoryRecord:
    created = created_at or datetime(2026, 6, 5, tzinfo=UTC)
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=memory_type,
        scope=scope,
        text=text,
        org_id="org_001",
        user_id="user_001",
        session_id=session_id if scope == MemoryScope.SESSION else None,
        source=MemorySource.MEETING_STATE,
        source_ids=[f"meeting:{memory_id}"],
        confidence=0.9,
        importance=0.85,
        privacy_level=privacy_level,
        write_status=write_status,
        tags=tags or [],
        created_at=created,
        updated_at=created,
        metadata=metadata or {},
    )


def test_exact_deadline_lookup_uses_target_entity_and_returns_provenance() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    store.add_memory(
        memory(
            "mem_launch_deadline",
            "Launch risk follow-up deadline is 2026-06-13.",
            memory_type=MemoryType.ACTION_ITEM,
            metadata={
                "normalized_entity": "launch risk",
                "normalized_deadline": "2026-06-13",
                "provenance": ["session:previous_sync:seg_12"],
            },
            tags=["launch", "risk", "deadline"],
            created_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
    )
    store.add_memory(
        memory(
            "mem_budget_deadline",
            "Budget review deadline is 2026-06-20.",
            memory_type=MemoryType.ACTION_ITEM,
            metadata={"normalized_entity": "budget review", "normalized_deadline": "2026-06-20"},
            tags=["budget", "deadline"],
            created_at=datetime(2026, 6, 4, tzinfo=UTC),
        )
    )

    context = service.search_context(
        MemoryQuery(
            query_text="之前这个 deadline 是什么时候？",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            active_entities=[{"canonical_name": "launch risk", "last_ts": 1200}],
            reference_time=datetime(2026, 6, 5, tzinfo=UTC),
        )
    )

    assert context.memory_refs == ["memory:mem_launch_deadline"]
    assert context.results[0].intent == MemoryRetrievalIntent.LOOKUP_DEADLINE.value
    assert context.results[0].reason == "exact_lookup_deadline"
    assert context.results[0].provenance == ["session:previous_sync:seg_12"]


def test_summary_gap_check_includes_pending_session_memory_by_default() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    store.add_memory(
        memory(
            "mem_pending_action",
            "Alex owns the deadline follow-up, pending confirmation.",
            memory_type=MemoryType.ACTION_ITEM,
            write_status=MemoryWriteStatus.PENDING_CONFIRMATION,
            tags=["owner", "deadline"],
        )
    )

    context = service.search_context(
        MemoryQuery(
            query_text="owner deadline",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            prompt_category=PromptCategory.SUMMARY_GAP_CHECK,
        )
    )

    assert context.memory_refs == ["memory:mem_pending_action"]
    assert context.results[0].rank_features["type_fit"] == 1.0


def test_high_privacy_memory_is_ref_only_on_glasses_surface() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    store.add_memory(
        memory(
            "mem_customer_price",
            "客户报价是 500 万，不能在眼镜端展开。",
            privacy_level=PrivacyLevel.HIGH,
            tags=["客户", "报价"],
            metadata={"provenance": ["session:sales_sync:seg_3"]},
        )
    )

    context = service.search_context(
        MemoryQuery(
            query_text="客户报价",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            prd_surface=PRDSurface.GLASSES_POPUP,
        )
    )

    assert context.memory_refs == ["memory:mem_customer_price"]
    assert context.results[0].use_policy == MemoryUsePolicy.DISPLAY_REF_ONLY.value
    assert "500 万" not in context.memory_context[0]
    assert "session:sales_sync:seg_3" in context.memory_context[0]


def test_open_recall_penalizes_recently_shown_memory() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    store.add_memory(
        memory(
            "mem_seen",
            "Alex owns the launch checklist.",
            memory_type=MemoryType.ACTION_ITEM,
            tags=["launch", "checklist"],
            created_at=now,
        )
    )
    store.add_memory(
        memory(
            "mem_unseen",
            "Bao owns the launch checklist review.",
            memory_type=MemoryType.ACTION_ITEM,
            tags=["launch", "checklist"],
            created_at=now - timedelta(days=1),
        )
    )

    context = service.search_context(
        MemoryQuery(
            query_text="launch checklist",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            shown_memory_ids=["mem_seen"],
            reference_time=now,
            limit=2,
        )
    )

    assert context.memory_refs[0] == "memory:mem_unseen"
    assert context.results[1].rank_features["redundancy"] == 1.0
