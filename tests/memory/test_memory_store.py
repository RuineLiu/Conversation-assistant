from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryQuery,
    MemoryRecord,
    MemoryRecordUpdate,
    MemoryScope,
    MemorySource,
    MemoryType,
    MemoryWriteStatus,
)
from proactive_assistant.prompting import PrivacyLevel


def memory(
    memory_id: str,
    text: str,
    *,
    memory_type: MemoryType = MemoryType.MEETING_FACT,
    scope: MemoryScope = MemoryScope.USER,
    org_id: str = "org_001",
    user_id: str = "user_001",
    session_id: str | None = None,
    privacy_level: PrivacyLevel = PrivacyLevel.LOW,
    write_status: MemoryWriteStatus = MemoryWriteStatus.ACTIVE,
    importance: float = 0.5,
    confidence: float = 0.7,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=memory_type,
        scope=scope,
        text=text,
        org_id=org_id,
        user_id=user_id,
        session_id=session_id,
        source=MemorySource.MANUAL,
        source_ids=["source_001"],
        confidence=confidence,
        importance=importance,
        privacy_level=privacy_level,
        write_status=write_status,
        tags=tags or [],
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_session_scoped_memory_requires_session_id() -> None:
    with pytest.raises(ValidationError):
        memory("mem_001", "Only visible inside one session.", scope=MemoryScope.SESSION)


def test_store_add_get_and_copy_boundaries() -> None:
    store = InMemoryMemoryStore()
    stored = store.add_memory(memory("mem_001", "用户偏好简短提示。", tags=["preference"]))

    stored.tags.append("mutated")
    fetched = store.get_memory("mem_001")
    fetched.tags.append("also_mutated")
    fetched_again = store.get_memory("mem_001")

    assert fetched_again.memory_id == "mem_001"
    assert fetched_again.last_accessed_at is not None
    assert fetched_again.tags == ["preference"]


def test_store_rejects_duplicate_and_unknown_ids() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(memory("mem_001", "first"))

    with pytest.raises(MemoryAlreadyExistsError):
        store.add_memory(memory("mem_001", "duplicate"))
    with pytest.raises(MemoryNotFoundError):
        store.get_memory("missing")


def test_list_memories_filters_scope_type_privacy_status_and_tags() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(
        memory(
            "mem_user_pref",
            "用户偏好一句话答案。",
            memory_type=MemoryType.USER_PREFERENCE,
            scope=MemoryScope.USER,
            tags=["prompt_style"],
            importance=0.8,
        )
    )
    store.add_memory(
        memory(
            "mem_archived",
            "旧的 action item。",
            memory_type=MemoryType.ACTION_ITEM,
            scope=MemoryScope.SESSION,
            session_id="session_001",
            write_status=MemoryWriteStatus.ARCHIVED,
            tags=["action"],
        )
    )
    store.add_memory(
        memory(
            "mem_private",
            "客户报价信息。",
            privacy_level=PrivacyLevel.HIGH,
            tags=["pricing"],
        )
    )

    filtered = store.list_memories(
        MemoryQuery(
            user_id="user_001",
            memory_types=[MemoryType.USER_PREFERENCE],
            privacy_levels=[PrivacyLevel.LOW],
            tags=["prompt_style"],
        )
    )

    assert [item.memory_id for item in filtered] == ["mem_user_pref"]
    assert [item.memory_id for item in store.list_memories(MemoryQuery(include_archived=True, limit=10))] == [
        "mem_user_pref",
        "mem_private",
        "mem_archived",
    ]


def test_session_query_includes_user_and_org_memories_but_filters_other_session_memories() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(memory("mem_user", "User-level preference.", scope=MemoryScope.USER))
    store.add_memory(memory("mem_session", "Current session action.", scope=MemoryScope.SESSION, session_id="session_001"))
    store.add_memory(memory("mem_other_session", "Other session action.", scope=MemoryScope.SESSION, session_id="session_002"))

    results = store.list_memories(MemoryQuery(session_id="session_001", limit=10))

    assert {item.memory_id for item in results} == {"mem_user", "mem_session"}


def test_search_memories_uses_keyword_overlap_and_ranking() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(
        memory(
            "mem_action",
            "Alex owns the launch deadline follow-up.",
            memory_type=MemoryType.ACTION_ITEM,
            importance=0.9,
            confidence=0.9,
            tags=["deadline", "owner"],
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
    )
    store.add_memory(
        memory(
            "mem_fact",
            "The launch meeting discussed pricing risk.",
            importance=0.7,
            confidence=0.8,
            tags=["risk"],
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )

    results = store.search_memories(MemoryQuery(query_text="owner deadline", limit=10))

    assert [result.memory.memory_id for result in results] == ["mem_action"]
    assert results[0].matched_terms == ["deadline", "owner"]
    assert results[0].score > 0.8


def test_update_memory_revalidates_scope_and_updates_metadata() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(memory("mem_001", "用户偏好简短提示。"))

    updated = store.update_memory(
        "mem_001",
        MemoryRecordUpdate(
            text="用户偏好非常简短的眼镜端提示。",
            importance=0.9,
            metadata={"updated_by": "test"},
        ),
    )

    assert updated.text == "用户偏好非常简短的眼镜端提示。"
    assert updated.importance == 0.9
    assert updated.metadata == {"updated_by": "test"}
    assert updated.updated_at > updated.created_at

    with pytest.raises(ValidationError):
        store.update_memory("mem_001", MemoryRecordUpdate(scope=MemoryScope.SESSION))


def test_archive_memory_hides_record_by_default() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(memory("mem_001", "Outdated preference.", created_at=datetime.now(UTC) - timedelta(days=1)))

    archived = store.archive_memory("mem_001", reason="user revoked")

    assert archived.write_status == "archived"
    assert archived.metadata["archive_reason"] == "user revoked"
    assert store.list_memories() == []
    assert store.list_memories(MemoryQuery(include_archived=True))[0].memory_id == "mem_001"


def test_forget_memory_redacts_record_and_hides_from_all_default_queries() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(
        memory(
            "mem_sensitive",
            "客户报价信息。",
            privacy_level=PrivacyLevel.HIGH,
            tags=["pricing", "customer"],
        )
    )

    forgotten = store.forget_memory("mem_sensitive", reason="user requested deletion")

    assert forgotten.write_status == MemoryWriteStatus.FORGOTTEN.value
    assert forgotten.text == "[forgotten]"
    assert forgotten.source_ids == []
    assert forgotten.tags == ["forgotten"]
    assert forgotten.confidence == 0.0
    assert forgotten.importance == 0.0
    assert forgotten.metadata["forgotten"] is True
    assert forgotten.metadata["forget_reason"] == "user requested deletion"
    assert "pricing" not in str(forgotten.metadata)
    assert store.list_memories(MemoryQuery(query_text="客户报价", include_archived=True, limit=10)) == []
    assert store.search_memories(MemoryQuery(query_text="客户报价", include_forgotten=True, limit=10)) == []
    assert store.list_memories(MemoryQuery(include_forgotten=True, limit=10))[0].memory_id == "mem_sensitive"
