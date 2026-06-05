from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryQuery,
    MemoryService,
    MemoryType,
    MemoryWriteStatus,
)
from proactive_assistant.prompting import PrivacyLevel
from proactive_assistant.runtime import MemoryCandidate, MemoryCandidateType, MemoryWritePolicy


def candidate(
    candidate_type: MemoryCandidateType,
    *,
    memory_candidate_id: str = "memcand_001",
    text: str = "用户偏好非常简短的眼镜端提示。",
    write_policy: MemoryWritePolicy = MemoryWritePolicy.ELIGIBLE,
    privacy_level: PrivacyLevel = PrivacyLevel.LOW,
    confidence: float = 0.8,
    metadata: dict[str, object] | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_candidate_id=memory_candidate_id,
        decision_id="dec_001",
        session_id="session_001",
        source_event_ids=["fb_001"],
        candidate_type=candidate_type,
        text=text,
        confidence=confidence,
        write_policy=write_policy,
        privacy_level=privacy_level,
        reason="test candidate",
        metadata=metadata or {},
    )


def test_propose_from_candidate_maps_user_preference_to_active_user_memory() -> None:
    service = MemoryService(InMemoryMemoryStore())

    memory = service.propose_from_candidate(
        candidate(
            MemoryCandidateType.USER_PREFERENCE,
            metadata={"org_id": "org_001", "subject_user_id": "user_001", "tags": ["prompt_style"]},
        )
    )

    assert memory.memory_id.startswith("mem_")
    assert memory.memory_type == "user_preference"
    assert memory.scope == "user"
    assert memory.org_id == "org_001"
    assert memory.user_id == "user_001"
    assert memory.session_id is None
    assert memory.source == "feedback_candidate"
    assert memory.source_ids == ["memory_candidate:memcand_001", "decision:dec_001", "feedback:fb_001"]
    assert memory.write_status == "active"
    assert memory.retention_policy == "until_revoked"
    assert memory.tags == ["feedback_candidate", "preference", "prompt_style", "user_preference"]


def test_commit_candidate_writes_pending_action_item_until_confirmed() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    pending = service.commit_candidate(
        candidate(
            MemoryCandidateType.ACTION_ITEM,
            memory_candidate_id="memcand_action",
            text="Alex owns the deadline follow-up.",
            write_policy=MemoryWritePolicy.NEEDS_CONFIRMATION,
            metadata={"tags": ["deadline", "owner"]},
        ),
        org_id="org_001",
        user_id="user_001",
    )

    assert pending.memory_type == "action_item"
    assert pending.scope == "session"
    assert pending.session_id == "session_001"
    assert pending.write_status == "pending_confirmation"
    assert pending.retention_policy == "90d"
    assert service.search_context(MemoryQuery(query_text="owner deadline", session_id="session_001")).results == []

    confirmed = service.confirm_memory(pending.memory_id)
    context = service.search_context(MemoryQuery(query_text="owner deadline", session_id="session_001"))

    assert confirmed.write_status == "active"
    assert context.memory_refs == [f"memory:{pending.memory_id}"]
    assert context.memory_context == [f"[memory:{pending.memory_id}] (action_item/session) Alex owns the deadline follow-up."]
    assert context.results[0].matched_terms == ["deadline", "owner"]


def test_blocked_candidate_is_written_as_rejected_and_not_retrieved() -> None:
    service = MemoryService(InMemoryMemoryStore())
    rejected = service.commit_candidate(
        candidate(
            MemoryCandidateType.MEETING_FACT,
            memory_candidate_id="memcand_blocked",
            text="Do not store this sensitive fact.",
            write_policy=MemoryWritePolicy.BLOCKED,
        )
    )

    assert rejected.write_status == "rejected"
    assert service.search_context(MemoryQuery(query_text="sensitive fact", session_id="session_001")).results == []


def test_privacy_preference_forces_high_privacy_user_memory() -> None:
    service = MemoryService(InMemoryMemoryStore())

    memory = service.commit_candidate(
        candidate(
            MemoryCandidateType.PRIVACY_PREFERENCE,
            memory_candidate_id="memcand_privacy",
            text="用户不希望在眼镜端显示客户报价细节。",
            privacy_level=PrivacyLevel.LOW,
        )
    )

    assert memory.memory_type == MemoryType.PRIVACY_PREFERENCE.value
    assert memory.scope == "user"
    assert memory.privacy_level == "high"
    assert memory.retention_policy == "until_revoked"
    assert {"privacy", "preference"}.issubset(set(memory.tags))


def test_reject_and_archive_memory_hide_records_from_context() -> None:
    service = MemoryService(InMemoryMemoryStore())
    first = service.commit_candidate(
        candidate(MemoryCandidateType.USER_PREFERENCE, memory_candidate_id="memcand_first", text="Prefer concise prompt style.")
    )
    second = service.commit_candidate(
        candidate(MemoryCandidateType.USER_PREFERENCE, memory_candidate_id="memcand_second", text="Prefer brief answers.")
    )

    rejected = service.reject_memory(first.memory_id, reason="user corrected")
    archived = service.archive_memory(second.memory_id, reason="superseded")

    assert rejected.write_status == MemoryWriteStatus.REJECTED.value
    assert rejected.metadata["reject_reason"] == "user corrected"
    assert archived.write_status == MemoryWriteStatus.ARCHIVED.value
    assert archived.metadata["archive_reason"] == "superseded"
    assert service.search_context(MemoryQuery(query_text="prefer", user_id="default_user")).results == []


def test_search_context_can_rank_active_memories_without_query_text() -> None:
    service = MemoryService(InMemoryMemoryStore())
    low = service.commit_candidate(
        candidate(
            MemoryCandidateType.MEETING_FACT,
            memory_candidate_id="memcand_low",
            text="The launch meeting discussed dashboard copy.",
            confidence=0.5,
        )
    )
    high = service.commit_candidate(
        candidate(
            MemoryCandidateType.ACTION_ITEM,
            memory_candidate_id="memcand_high",
            text="Bao owns the launch checklist.",
            confidence=0.95,
        )
    )

    context = service.search_context(MemoryQuery(session_id="session_001", limit=2))

    assert context.memory_refs == [f"memory:{high.memory_id}", f"memory:{low.memory_id}"]
    assert context.memory_context[0].endswith("Bao owns the launch checklist.")
    assert context.results[0].reason == "ranked by importance and confidence"
