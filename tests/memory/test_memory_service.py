import pytest

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryMergeAction,
    MemoryPendingUpdateStatus,
    MemoryPromotionStatus,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemorySource,
    MemoryType,
    MemoryUpdatePolicyDecision,
    MemoryUpsertStatus,
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


def test_upsert_candidate_creates_then_keeps_unchanged_memory() -> None:
    service = MemoryService(InMemoryMemoryStore())
    source = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_snapshot_action",
        text="张三负责客户报价确认，下周五截止。",
        metadata={"owner": "张三", "deadline": "下周五", "tags": ["meeting_state", "action_item"]},
    )

    created = service.upsert_candidate(source, org_id="org_001", user_id="user_001")
    unchanged = service.upsert_candidate(source, org_id="org_001", user_id="user_001")

    assert created.status == MemoryUpsertStatus.CREATED.value
    assert created.version == 1
    assert created.memory.metadata["memory_version"] == 1
    assert created.memory.metadata["memory_content_digest"] == created.new_digest
    assert unchanged.status == MemoryUpsertStatus.UNCHANGED.value
    assert unchanged.memory.memory_id == created.memory.memory_id
    assert unchanged.memory.updated_at == created.memory.updated_at
    assert unchanged.memory.metadata["memory_version"] == 1


def test_upsert_candidate_auto_updates_non_sensitive_existing_memory_with_new_version() -> None:
    service = MemoryService(InMemoryMemoryStore())
    first = candidate(
        MemoryCandidateType.MEETING_FACT,
        memory_candidate_id="memcand_snapshot_risk",
        text="Risk: 客户报价材料还没有完成。 Status: open.",
        metadata={"status": "open", "tags": ["meeting_state", "risk"]},
    )
    second = candidate(
        MemoryCandidateType.MEETING_FACT,
        memory_candidate_id="memcand_snapshot_risk",
        text="Risk: 客户报价材料已经完成。 Status: closed.",
        metadata={"status": "closed", "tags": ["meeting_state", "risk"]},
    )

    created = service.upsert_candidate(first, org_id="org_001", user_id="user_001")
    updated = service.upsert_candidate(second, org_id="org_001", user_id="user_001")
    stored = service.store.list_memories(MemoryQuery(session_id="session_001", include_pending=True, limit=20))

    assert updated.status == MemoryUpsertStatus.UPDATED.value
    assert updated.update_policy == MemoryUpdatePolicyDecision.AUTO_UPDATE.value
    assert updated.version == 2
    assert updated.previous_memory is not None
    assert updated.previous_memory.text == created.memory.text
    assert updated.memory.memory_id == created.memory.memory_id
    assert updated.memory.text == "Risk: 客户报价材料已经完成。 Status: closed."
    assert updated.memory.metadata["status"] == "closed"
    assert updated.memory.metadata["memory_version"] == 2
    assert updated.memory.metadata["memory_version_history"][0]["version"] == 2
    assert "text" in updated.changed_fields
    assert "metadata.status" in updated.changed_fields
    assert len(stored) == 1


def test_upsert_candidate_requires_confirmation_for_owner_and_deadline_changes() -> None:
    service = MemoryService(InMemoryMemoryStore())
    first = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_snapshot_action",
        text="张三负责客户报价确认，下周五截止。",
        metadata={"owner": "张三", "deadline": "下周五", "tags": ["meeting_state", "action_item"]},
    )
    second = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_snapshot_action",
        text="李四负责客户报价确认，下周一截止。",
        metadata={"owner": "李四", "deadline": "下周一", "tags": ["meeting_state", "action_item"]},
    )

    created = service.upsert_candidate(first, org_id="org_001", user_id="user_001")
    pending = service.upsert_candidate(second, org_id="org_001", user_id="user_001")
    stored = service.store.list_memories(MemoryQuery(session_id="session_001", include_pending=True, limit=20))

    assert pending.status == MemoryUpsertStatus.UNCHANGED.value
    assert pending.update_policy == MemoryUpdatePolicyDecision.NEEDS_CONFIRMATION.value
    assert pending.memory.memory_id == created.memory.memory_id
    assert pending.memory.text == "张三负责客户报价确认，下周五截止。"
    assert pending.proposed_memory is not None
    assert pending.proposed_memory.text == "李四负责客户报价确认，下周一截止。"
    assert pending.proposed_memory.metadata["owner"] == "李四"
    assert pending.proposed_memory.metadata["deadline"] == "下周一"
    assert pending.pending_update is not None
    assert pending.pending_update.status == MemoryPendingUpdateStatus.PENDING.value
    assert pending.pending_update.memory_id == created.memory.memory_id
    assert "sensitive_structured_field_changed" in pending.policy_reasons
    assert "metadata.owner" in pending.changed_fields
    assert len(stored) == 1
    assert stored[0].text == "张三负责客户报价确认，下周五截止。"

    listed = service.list_pending_updates(memory_id=created.memory.memory_id)
    applied = service.apply_pending_update(pending.pending_update.update_id, reason="user confirmed")
    resolved = service.get_pending_update(pending.pending_update.update_id)

    assert listed == [pending.pending_update]
    assert applied.status == MemoryUpsertStatus.UPDATED.value
    assert applied.pending_update is not None
    assert applied.pending_update.status == MemoryPendingUpdateStatus.APPLIED.value
    assert applied.pending_update.resolved_reason == "user confirmed"
    assert applied.memory.text == "李四负责客户报价确认，下周一截止。"
    assert applied.memory.metadata["memory_version"] == 2
    assert resolved.status == MemoryPendingUpdateStatus.APPLIED.value


def test_upsert_candidate_deduplicates_same_fact_from_different_candidate_id() -> None:
    service = MemoryService(InMemoryMemoryStore())
    first = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_action_a",
        text="张三负责客户报价确认，下周五截止。",
        metadata={"entity": "客户报价确认", "owner": "张三", "deadline": "下周五", "tags": ["action_item"]},
    )
    second = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_action_b",
        text="张三负责客户报价确认，下周五截止。",
        metadata={"entity": "客户报价确认", "owner": "张三", "deadline": "下周五", "tags": ["action_item"]},
    )

    created = service.upsert_candidate(first, org_id="org_001", user_id="user_001")
    duplicate = service.upsert_candidate(second, org_id="org_001", user_id="user_001")
    stored = service.store.list_memories(MemoryQuery(session_id="session_001", include_pending=True, limit=20))

    assert duplicate.status == MemoryUpsertStatus.UNCHANGED.value
    assert duplicate.memory.memory_id == created.memory.memory_id
    assert duplicate.merge_decision is not None
    assert duplicate.merge_decision.action == MemoryMergeAction.DUPLICATE.value
    assert duplicate.merge_decision.existing_memory_id == created.memory.memory_id
    assert len(stored) == 1


def test_upsert_candidate_reinforces_existing_memory_without_replacing_text() -> None:
    service = MemoryService(InMemoryMemoryStore())
    first = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_action_a",
        text="张三负责客户报价确认，下周五截止。",
        confidence=0.72,
        metadata={"entity": "客户报价确认", "owner": "张三", "deadline": "下周五", "tags": ["action_item"]},
    )
    second = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_action_b",
        text="客户报价确认还是张三负责，截止仍是下周五。",
        confidence=0.9,
        metadata={"entity": "客户报价确认", "owner": "张三", "deadline": "下周五", "tags": ["action_item", "meeting_end"]},
    )

    created = service.upsert_candidate(first, org_id="org_001", user_id="user_001")
    reinforced = service.upsert_candidate(second, org_id="org_001", user_id="user_001")
    stored = service.store.get_memory(created.memory.memory_id)

    assert reinforced.status == MemoryUpsertStatus.UPDATED.value
    assert reinforced.merge_decision is not None
    assert reinforced.merge_decision.action == MemoryMergeAction.REINFORCEMENT.value
    assert {
        "source_ids",
        "confidence",
        "importance",
        "tags",
        "metadata.memory_merge_history",
        "metadata.memory_last_merge_action",
        "metadata.memory_last_merge_resolution_status",
    }.issubset(set(reinforced.changed_fields))
    assert stored.text == "张三负责客户报价确认，下周五截止。"
    assert stored.confidence > created.memory.confidence
    assert "meeting_end" in stored.tags
    assert stored.metadata["memory_last_merge_action"] == MemoryMergeAction.REINFORCEMENT.value
    assert len(service.store.list_memories(MemoryQuery(session_id="session_001", include_pending=True, limit=20))) == 1


def test_upsert_candidate_conflict_from_different_candidate_id_creates_pending_update() -> None:
    service = MemoryService(InMemoryMemoryStore())
    first = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_action_a",
        text="张三负责客户报价确认，下周五截止。",
        metadata={"entity": "客户报价确认", "owner": "张三", "deadline": "下周五", "tags": ["action_item"]},
    )
    second = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_action_b",
        text="李四负责客户报价确认，下周一截止。",
        metadata={"entity": "客户报价确认", "owner": "李四", "deadline": "下周一", "tags": ["action_item"]},
    )

    created = service.upsert_candidate(first)
    pending = service.upsert_candidate(second)

    assert pending.status == MemoryUpsertStatus.UNCHANGED.value
    assert pending.merge_decision is not None
    assert pending.merge_decision.action == MemoryMergeAction.CONFLICT.value
    assert sorted(pending.merge_decision.conflict_fields) == [
        "metadata.assignee",
        "metadata.deadline",
        "metadata.normalized_deadline",
        "metadata.owner",
    ]
    assert pending.pending_update is not None
    assert pending.pending_update.memory_id == created.memory.memory_id
    assert "memory_merge_requires_confirmation" in pending.policy_reasons
    assert pending.proposed_memory is not None
    assert pending.proposed_memory.memory_id == created.memory.memory_id
    assert pending.proposed_memory.metadata["owner"] == "李四"


def test_upsert_candidate_supersede_signal_is_tracked_as_pending_update() -> None:
    service = MemoryService(InMemoryMemoryStore())
    created = service.upsert_candidate(
        candidate(
            MemoryCandidateType.PERSON_OR_FACT,
            memory_candidate_id="memcand_legal_a",
            text="李四是 Project Atlas 的法务接口。",
            metadata={"entity": "Project Atlas 法务接口", "owner": "李四", "tags": ["person_or_fact"]},
        )
    )
    supersede = service.upsert_candidate(
        candidate(
            MemoryCandidateType.PERSON_OR_FACT,
            memory_candidate_id="memcand_legal_b",
            text="更新为王五是 Project Atlas 的法务接口。",
            metadata={"entity": "Project Atlas 法务接口", "owner": "王五", "tags": ["person_or_fact"]},
        )
    )

    assert supersede.merge_decision is not None
    assert supersede.merge_decision.action == MemoryMergeAction.SUPERSEDE.value
    assert supersede.pending_update is not None
    assert supersede.pending_update.memory_id == created.memory.memory_id
    assert supersede.proposed_memory is not None
    assert supersede.proposed_memory.metadata["memory_last_merge_action"] == MemoryMergeAction.SUPERSEDE.value


def test_reject_pending_update_keeps_existing_memory_unchanged() -> None:
    service = MemoryService(InMemoryMemoryStore())
    first = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_snapshot_action",
        text="张三负责客户报价确认，下周五截止。",
        metadata={"owner": "张三", "deadline": "下周五", "tags": ["meeting_state", "action_item"]},
    )
    second = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_snapshot_action",
        text="李四负责客户报价确认，下周一截止。",
        metadata={"owner": "李四", "deadline": "下周一", "tags": ["meeting_state", "action_item"]},
    )

    created = service.upsert_candidate(first)
    pending = service.upsert_candidate(second)
    assert pending.pending_update is not None

    rejected = service.reject_pending_update(pending.pending_update.update_id, reason="wrong owner")
    stored = service.store.get_memory(created.memory.memory_id)

    assert rejected.status == MemoryPendingUpdateStatus.REJECTED.value
    assert rejected.resolved_reason == "wrong owner"
    assert stored.text == "张三负责客户报价确认，下周五截止。"
    assert stored.metadata["owner"] == "张三"


def test_forget_memory_redacts_record_invalidates_pending_update_and_blocks_rewrite() -> None:
    service = MemoryService(InMemoryMemoryStore())
    first = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_snapshot_action",
        text="张三负责客户报价确认，下周五截止。",
        metadata={"owner": "张三", "deadline": "下周五", "tags": ["meeting_state", "action_item"]},
    )
    second = candidate(
        MemoryCandidateType.ACTION_ITEM,
        memory_candidate_id="memcand_snapshot_action",
        text="李四负责客户报价确认，下周一截止。",
        metadata={"owner": "李四", "deadline": "下周一", "tags": ["meeting_state", "action_item"]},
    )

    created = service.upsert_candidate(first)
    pending = service.upsert_candidate(second)
    assert pending.pending_update is not None

    forgotten = service.forget_memory(created.memory.memory_id, reason="user said do not remember this")
    listed = service.list_pending_updates(memory_id=created.memory.memory_id, status=None)
    context = service.search_context(MemoryQuery(query_text="客户报价 张三 李四", session_id="session_001"))
    blocked = service.upsert_candidate(second)

    assert forgotten.memory.write_status == MemoryWriteStatus.FORGOTTEN.value
    assert forgotten.memory.text == "[forgotten]"
    assert forgotten.memory.metadata["forget_reason"] == "user said do not remember this"
    assert len(forgotten.invalidated_pending_updates) == 1
    assert forgotten.invalidated_pending_updates[0].status == MemoryPendingUpdateStatus.REJECTED.value
    assert forgotten.invalidated_pending_updates[0].resolved_reason == "memory forgotten: user said do not remember this"
    assert listed[0].status == MemoryPendingUpdateStatus.REJECTED.value
    assert context.results == []
    assert blocked.status == MemoryUpsertStatus.UNCHANGED.value
    assert blocked.update_policy == MemoryUpdatePolicyDecision.BLOCKED.value
    assert blocked.policy_reasons == ["existing_memory_is_forgotten"]
    with pytest.raises(ValueError, match="already rejected"):
        service.apply_pending_update(pending.pending_update.update_id)


def test_upsert_candidate_blocks_archived_memory_updates() -> None:
    service = MemoryService(InMemoryMemoryStore())
    source = candidate(
        MemoryCandidateType.MEETING_FACT,
        memory_candidate_id="memcand_snapshot_risk",
        text="Risk: 客户报价材料还没有完成。 Status: open.",
        metadata={"status": "open", "tags": ["meeting_state", "risk"]},
    )
    updated_source = candidate(
        MemoryCandidateType.MEETING_FACT,
        memory_candidate_id="memcand_snapshot_risk",
        text="Risk: 客户报价材料已经完成。 Status: closed.",
        metadata={"status": "closed", "tags": ["meeting_state", "risk"]},
    )

    created = service.upsert_candidate(source)
    service.archive_memory(created.memory.memory_id, reason="manual cleanup")
    blocked = service.upsert_candidate(updated_source)

    assert blocked.status == MemoryUpsertStatus.UNCHANGED.value
    assert blocked.update_policy == MemoryUpdatePolicyDecision.BLOCKED.value
    assert blocked.memory.write_status == MemoryWriteStatus.ARCHIVED.value
    assert blocked.policy_reasons == ["existing_memory_is_archived"]


def test_promote_project_context_memory_is_visible_across_sessions() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    source = store.add_memory(
        MemoryRecord(
            memory_id="mem_session_project_001",
            memory_type=MemoryType.MEETING_FACT,
            scope=MemoryScope.SESSION,
            text="Project Atlas uses customer pricing v2 as the launch blocker.",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            source=MemorySource.MEETING_STATE,
            confidence=0.86,
            importance=0.72,
            tags=["project_context", "promotion_candidate"],
            metadata={"tags": ["project_context"], "entity": "Project Atlas"},
        )
    )

    promoted = service.promote_memory(source.memory_id, reason="recurs across planning meetings")
    context = service.search_context(
        MemoryQuery(
            query_text="Project Atlas launch blocker pricing",
            org_id="org_001",
            user_id="user_001",
            session_id="session_002",
        )
    )

    assert promoted.status == MemoryPromotionStatus.PROMOTED.value
    assert promoted.promoted_memory is not None
    assert promoted.promoted_memory.scope == MemoryScope.ORG.value
    assert promoted.promoted_memory.memory_type == MemoryType.PROJECT_CONTEXT.value
    assert promoted.promoted_memory.session_id is None
    assert promoted.promoted_memory.source == MemorySource.PROMOTED.value
    assert promoted.promoted_memory.metadata["promoted_from_memory_id"] == source.memory_id
    assert context.memory_refs == [f"memory:{promoted.promoted_memory.memory_id}"]


def test_promote_action_item_requires_approval_before_long_term_write() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    source = service.confirm_memory(
        service.commit_candidate(
            candidate(
                MemoryCandidateType.ACTION_ITEM,
                memory_candidate_id="memcand_action_promote",
                text="张三负责客户报价确认，下周五截止。",
                write_policy=MemoryWritePolicy.NEEDS_CONFIRMATION,
                metadata={"owner": "张三", "deadline": "下周五", "tags": ["meeting_state", "action_item"]},
            ),
            org_id="org_001",
            user_id="user_001",
        ).memory_id
    )

    proposed = service.promote_memory(source.memory_id, target_scope=MemoryScope.ORG, reason="manual review needed")
    approved = service.promote_memory(
        source.memory_id,
        target_scope=MemoryScope.ORG,
        reason="manually approved",
        approved=True,
    )

    assert proposed.status == MemoryPromotionStatus.NEEDS_CONFIRMATION.value
    assert proposed.promoted_memory is None
    assert proposed.proposed_memory is not None
    assert "action_items_are_session_bound_by_default" in proposed.policy_reasons
    assert approved.status == MemoryPromotionStatus.PROMOTED.value
    assert approved.promoted_memory is not None
    assert approved.promoted_memory.scope == MemoryScope.ORG.value


def test_promote_high_privacy_memory_needs_confirmation() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    source = store.add_memory(
        MemoryRecord(
            memory_id="mem_sensitive_001",
            memory_type=MemoryType.MEETING_FACT,
            scope=MemoryScope.SESSION,
            text="Customer ACME private pricing is under legal review.",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            source=MemorySource.MEETING_STATE,
            confidence=0.82,
            importance=0.8,
            privacy_level=PrivacyLevel.HIGH,
            tags=["project_context"],
            metadata={"privacy_risk": 0.81, "tags": ["project_context"]},
        )
    )

    result = service.promote_memory(source.memory_id)

    assert result.status == MemoryPromotionStatus.NEEDS_CONFIRMATION.value
    assert result.promoted_memory is None
    assert result.proposed_memory is not None
    assert "high_privacy_memory_needs_confirmation" in result.policy_reasons
