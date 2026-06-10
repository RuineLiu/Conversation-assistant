from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha1
from typing import Any

from proactive_assistant.memory.contracts import (
    MemoryContext,
    MemoryForgetResult,
    MemoryMergeAction,
    MemoryMergeDecision,
    MemoryPendingUpdate,
    MemoryPendingUpdateStatus,
    MemoryPromotionResult,
    MemoryPromotionStatus,
    MemoryQuery,
    MemoryRecord,
    MemoryRecordUpdate,
    MemoryScope,
    MemorySource,
    MemoryType,
    MemoryUpdatePolicyDecision,
    MemoryUpsertResult,
    MemoryUpsertStatus,
    MemoryWriteStatus,
    RetentionPolicy,
)
from proactive_assistant.memory.consolidation import consolidated_memory_metadata, consolidated_memory_tags
from proactive_assistant.memory.merge import detect_memory_merge
from proactive_assistant.memory.query_understanding import QueryUnderstandingService
from proactive_assistant.memory.retrieval import MemoryRetriever
from proactive_assistant.memory.store import MemoryRepository
from proactive_assistant.memory.store import MemoryAlreadyExistsError, MemoryPendingUpdateAlreadyExistsError
from proactive_assistant.memory.vector_store import MemoryVectorStore
from proactive_assistant.model_gateway.embeddings import EmbeddingClient
from proactive_assistant.prompting import PrivacyLevel
from proactive_assistant.runtime import MemoryCandidate, MemoryCandidateType, MemoryWritePolicy


class MemoryService:
    """Lifecycle service for converting candidates into long-term memories."""

    def __init__(
        self,
        store: MemoryRepository,
        *,
        embedding_client: EmbeddingClient | None = None,
        vector_store: MemoryVectorStore | None = None,
        embedding_model: str | None = None,
        query_understanding: QueryUnderstandingService | None = None,
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.query_understanding = query_understanding

    def propose_from_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        org_id: str = "default_org",
        user_id: str = "default_user",
    ) -> MemoryRecord:
        candidate_type = MemoryCandidateType(candidate.candidate_type)
        write_policy = MemoryWritePolicy(candidate.write_policy)
        now = datetime.now(UTC)
        metadata = consolidated_memory_metadata(candidate)
        return MemoryRecord(
            memory_id=_memory_id_for_candidate(candidate),
            memory_type=_memory_type_for_candidate(candidate_type),
            scope=_scope_for_candidate(candidate_type),
            text=candidate.text,
            org_id=str(candidate.metadata.get("org_id", org_id)),
            user_id=str(candidate.metadata.get("user_id", candidate.metadata.get("subject_user_id", user_id))),
            session_id=candidate.session_id if _scope_for_candidate(candidate_type) == MemoryScope.SESSION else None,
            source=MemorySource.FEEDBACK_CANDIDATE,
            source_ids=_source_ids(candidate),
            confidence=candidate.confidence,
            importance=_importance_for_candidate(candidate_type, candidate.confidence),
            privacy_level=_privacy_for_candidate(candidate),
            retention_policy=_retention_for_candidate(candidate_type),
            write_status=_write_status_for_policy(write_policy),
            tags=_tags_for_candidate(candidate_type, candidate),
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )

    def commit_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        org_id: str = "default_org",
        user_id: str = "default_user",
    ) -> MemoryRecord:
        memory = self.propose_from_candidate(candidate, org_id=org_id, user_id=user_id)
        return self.store.add_memory(memory)

    def commit_candidate_once(
        self,
        candidate: MemoryCandidate,
        *,
        org_id: str = "default_org",
        user_id: str = "default_user",
    ) -> MemoryRecord:
        memory = self.propose_from_candidate(candidate, org_id=org_id, user_id=user_id)
        try:
            return self.store.add_memory(memory)
        except MemoryAlreadyExistsError:
            return self.store.get_memory(memory.memory_id)

    def upsert_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        org_id: str = "default_org",
        user_id: str = "default_user",
    ) -> MemoryUpsertResult:
        proposed = self.propose_from_candidate(candidate, org_id=org_id, user_id=user_id)
        merge_decision = detect_memory_merge(proposed, self._merge_candidate_pool(proposed))
        if merge_decision.action != MemoryMergeAction.CREATE.value:
            existing = self.store.get_memory(str(merge_decision.existing_memory_id))
            return self._apply_merge_decision(existing, proposed, merge_decision)

        prepared, digest = _memory_with_version_metadata(
            proposed,
            version=1,
            digest=_memory_content_digest(proposed),
            previous_digest=None,
            changed_fields=[],
            previous_history=[],
        )
        try:
            created = self.store.add_memory(prepared)
            return MemoryUpsertResult(
                status=MemoryUpsertStatus.CREATED,
                memory=created,
                previous_memory=None,
                changed_fields=[],
                previous_digest=None,
                new_digest=digest,
                version=1,
                merge_decision=merge_decision,
            )
        except MemoryAlreadyExistsError:
            existing = self.store.get_memory(proposed.memory_id)
            merge_decision = MemoryMergeDecision(
                action=MemoryMergeAction.UPDATE,
                existing_memory_id=existing.memory_id,
                proposed_memory_id=proposed.memory_id,
                similarity=1.0,
                evidence_refs=_merge_unique(existing.source_ids, proposed.source_ids),
                reasons=["same_memory_id"],
            )

        return self._upsert_existing_memory(existing, proposed, merge_decision, record_merge=False)

    def _apply_merge_decision(
        self,
        existing: MemoryRecord,
        proposed: MemoryRecord,
        merge_decision: MemoryMergeDecision,
    ) -> MemoryUpsertResult:
        action = MemoryMergeAction(merge_decision.action)
        if action == MemoryMergeAction.BLOCKED:
            return MemoryUpsertResult(
                status=MemoryUpsertStatus.UNCHANGED,
                memory=existing,
                previous_memory=existing,
                proposed_memory=proposed,
                changed_fields=[],
                previous_digest=_memory_content_digest(existing),
                new_digest=_memory_content_digest(proposed),
                version=_memory_version(existing),
                update_policy=MemoryUpdatePolicyDecision.BLOCKED,
                policy_reasons=list(merge_decision.reasons),
                merge_decision=merge_decision,
            )
        if action == MemoryMergeAction.DUPLICATE:
            return MemoryUpsertResult(
                status=MemoryUpsertStatus.UNCHANGED,
                memory=existing,
                previous_memory=existing,
                proposed_memory=proposed,
                changed_fields=[],
                previous_digest=_memory_content_digest(existing),
                new_digest=_memory_content_digest(existing),
                version=_memory_version(existing),
                update_policy=MemoryUpdatePolicyDecision.AUTO_UPDATE,
                policy_reasons=list(merge_decision.reasons),
                merge_decision=merge_decision,
            )
        if action == MemoryMergeAction.REINFORCEMENT:
            return self._reinforce_existing_memory(existing, proposed, merge_decision)
        return self._upsert_existing_memory(existing, proposed, merge_decision)

    def _reinforce_existing_memory(
        self,
        existing: MemoryRecord,
        proposed: MemoryRecord,
        merge_decision: MemoryMergeDecision,
    ) -> MemoryUpsertResult:
        previous_digest = _memory_content_digest(existing)
        reinforced = _reinforced_memory(existing, proposed, merge_decision)
        changed_fields = _changed_fields(existing, reinforced)
        if not changed_fields:
            return MemoryUpsertResult(
                status=MemoryUpsertStatus.UNCHANGED,
                memory=existing,
                previous_memory=existing,
                proposed_memory=proposed,
                changed_fields=[],
                previous_digest=previous_digest,
                new_digest=previous_digest,
                version=_memory_version(existing),
                update_policy=MemoryUpdatePolicyDecision.AUTO_UPDATE,
                policy_reasons=list(merge_decision.reasons),
                merge_decision=merge_decision,
            )
        updated_memory, new_digest = _memory_with_version_metadata(
            reinforced,
            version=_memory_version(existing) + 1,
            digest=_memory_content_digest(reinforced),
            previous_digest=previous_digest,
            changed_fields=changed_fields,
            previous_history=_memory_version_history(existing),
        )
        updated = self.store.update_memory(existing.memory_id, _record_update_from_memory(updated_memory))
        return MemoryUpsertResult(
            status=MemoryUpsertStatus.UPDATED,
            memory=updated,
            previous_memory=existing,
            proposed_memory=proposed,
            changed_fields=changed_fields,
            previous_digest=previous_digest,
            new_digest=new_digest,
            version=_memory_version(updated),
            update_policy=MemoryUpdatePolicyDecision.AUTO_UPDATE,
            policy_reasons=list(merge_decision.reasons),
            merge_decision=merge_decision,
        )

    def _upsert_existing_memory(
        self,
        existing: MemoryRecord,
        proposed: MemoryRecord,
        merge_decision: MemoryMergeDecision,
        *,
        record_merge: bool = True,
    ) -> MemoryUpsertResult:
        desired = _merge_upsert_memory(existing, proposed).model_copy(update={"memory_id": existing.memory_id})
        if record_merge:
            desired = _memory_with_merge_metadata(desired, merge_decision)
        previous_digest = _memory_content_digest(existing)
        new_digest = _memory_content_digest(desired)
        changed_fields = _changed_fields(existing, desired)
        policy, policy_reasons = _evaluate_update_policy(existing, desired, changed_fields)
        if MemoryMergeAction(merge_decision.action) in {MemoryMergeAction.CONFLICT, MemoryMergeAction.SUPERSEDE}:
            policy = MemoryUpdatePolicyDecision.NEEDS_CONFIRMATION
            policy_reasons = sorted(set([*policy_reasons, *merge_decision.reasons, "memory_merge_requires_confirmation"]))
        if policy == MemoryUpdatePolicyDecision.BLOCKED:
            return MemoryUpsertResult(
                status=MemoryUpsertStatus.UNCHANGED,
                memory=existing,
                previous_memory=existing,
                proposed_memory=desired,
                changed_fields=changed_fields,
                previous_digest=previous_digest,
                new_digest=new_digest,
                version=_memory_version(existing),
                update_policy=policy,
                policy_reasons=policy_reasons,
                merge_decision=merge_decision,
            )
        if policy == MemoryUpdatePolicyDecision.NEEDS_CONFIRMATION:
            proposed_versioned, proposed_digest = _memory_with_version_metadata(
                desired,
                version=_memory_version(existing) + 1,
                digest=new_digest,
                previous_digest=previous_digest,
                changed_fields=changed_fields,
                previous_history=_memory_version_history(existing),
            )
            pending_update = self._store_pending_update(
                _pending_update_for_proposal(
                    existing,
                    proposed_versioned,
                    previous_digest=previous_digest,
                    new_digest=proposed_digest,
                    changed_fields=changed_fields,
                    policy_reasons=policy_reasons,
                )
            )
            return MemoryUpsertResult(
                status=MemoryUpsertStatus.UNCHANGED,
                memory=existing,
                previous_memory=existing,
                proposed_memory=proposed_versioned,
                changed_fields=changed_fields,
                previous_digest=previous_digest,
                new_digest=proposed_digest,
                version=_memory_version(existing),
                update_policy=policy,
                policy_reasons=policy_reasons,
                pending_update=pending_update,
                merge_decision=merge_decision,
            )
        needs_version_seed = "memory_content_digest" not in existing.metadata or "memory_version" not in existing.metadata
        if previous_digest == new_digest and not changed_fields and not needs_version_seed:
            return MemoryUpsertResult(
                status=MemoryUpsertStatus.UNCHANGED,
                memory=existing,
                previous_memory=existing,
                changed_fields=[],
                previous_digest=previous_digest,
                new_digest=new_digest,
                version=_memory_version(existing),
                update_policy=policy,
                policy_reasons=policy_reasons,
                merge_decision=merge_decision,
            )

        if needs_version_seed and not changed_fields:
            changed_fields = ["version_metadata"]

        next_version = _memory_version(existing) + (0 if changed_fields == ["version_metadata"] else 1)
        updated_memory, new_digest = _memory_with_version_metadata(
            desired,
            version=next_version,
            digest=new_digest,
            previous_digest=previous_digest,
            changed_fields=changed_fields,
            previous_history=_memory_version_history(existing),
        )
        updated = self.store.update_memory(
            existing.memory_id,
            MemoryRecordUpdate(
                memory_type=updated_memory.memory_type,
                scope=updated_memory.scope,
                text=updated_memory.text,
                org_id=updated_memory.org_id,
                user_id=updated_memory.user_id,
                session_id=updated_memory.session_id,
                source=updated_memory.source,
                source_ids=updated_memory.source_ids,
                confidence=updated_memory.confidence,
                importance=updated_memory.importance,
                privacy_level=updated_memory.privacy_level,
                retention_policy=updated_memory.retention_policy,
                write_status=updated_memory.write_status,
                tags=updated_memory.tags,
                metadata=updated_memory.metadata,
            ),
        )
        return MemoryUpsertResult(
            status=MemoryUpsertStatus.UPDATED,
            memory=updated,
            previous_memory=existing,
            changed_fields=changed_fields,
            previous_digest=previous_digest,
            new_digest=new_digest,
            version=next_version,
            update_policy=policy,
            policy_reasons=policy_reasons,
            merge_decision=merge_decision,
        )

    def confirm_memory(self, memory_id: str) -> MemoryRecord:
        memory = self.store.get_memory(memory_id)
        _ensure_not_forgotten(memory, operation="confirm")
        return self.store.update_memory(memory_id, MemoryRecordUpdate(write_status=MemoryWriteStatus.ACTIVE))

    def reject_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        memory = self.store.get_memory(memory_id)
        _ensure_not_forgotten(memory, operation="reject")
        metadata = dict(memory.metadata)
        if reason:
            metadata["reject_reason"] = reason
        return self.store.update_memory(
            memory_id,
            MemoryRecordUpdate(write_status=MemoryWriteStatus.REJECTED, metadata=metadata),
        )

    def archive_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        memory = self.store.get_memory(memory_id)
        _ensure_not_forgotten(memory, operation="archive")
        return self.store.archive_memory(memory_id, reason=reason)

    def forget_memory(self, memory_id: str, *, reason: str = "") -> MemoryForgetResult:
        pending_updates = self.store.list_pending_updates(
            memory_id=memory_id,
            status=MemoryPendingUpdateStatus.PENDING,
        )
        forgotten = self.store.forget_memory(memory_id, reason=reason)
        invalidated: list[MemoryPendingUpdate] = []
        for pending_update in pending_updates:
            invalidated.append(
                self.store.update_pending_update_status(
                    pending_update.update_id,
                    MemoryPendingUpdateStatus.REJECTED,
                    reason=_pending_rejection_reason_for_forget(reason),
                )
            )
        return MemoryForgetResult(
            memory=forgotten,
            invalidated_pending_updates=invalidated,
            reason=reason,
            redacted=True,
        )

    def list_pending_updates(
        self,
        *,
        memory_id: str | None = None,
        status: MemoryPendingUpdateStatus | str | None = MemoryPendingUpdateStatus.PENDING,
    ) -> list[MemoryPendingUpdate]:
        return self.store.list_pending_updates(memory_id=memory_id, status=status)

    def get_pending_update(self, update_id: str) -> MemoryPendingUpdate:
        return self.store.get_pending_update(update_id)

    def apply_pending_update(self, update_id: str, *, reason: str = "") -> MemoryUpsertResult:
        pending_update = self.store.get_pending_update(update_id)
        if MemoryPendingUpdateStatus(pending_update.status) != MemoryPendingUpdateStatus.PENDING:
            raise ValueError(f"pending memory update is already {pending_update.status}: {update_id}")
        current = self.store.get_memory(pending_update.memory_id)
        _ensure_not_forgotten(current, operation="apply pending update")
        current_digest = _memory_content_digest(current)
        if current_digest != pending_update.previous_digest:
            raise ValueError(f"pending memory update is stale: {update_id}")
        updated = self.store.update_memory(
            pending_update.memory_id,
            _record_update_from_memory(pending_update.proposed_memory),
        )
        resolved = self.store.update_pending_update_status(
            update_id,
            MemoryPendingUpdateStatus.APPLIED,
            reason=reason,
        )
        return MemoryUpsertResult(
            status=MemoryUpsertStatus.UPDATED,
            memory=updated,
            previous_memory=current,
            proposed_memory=pending_update.proposed_memory,
            changed_fields=list(pending_update.changed_fields),
            previous_digest=pending_update.previous_digest,
            new_digest=pending_update.new_digest,
            version=_memory_version(updated),
            update_policy=pending_update.update_policy,
            policy_reasons=list(pending_update.policy_reasons),
            pending_update=resolved,
        )

    def reject_pending_update(self, update_id: str, *, reason: str = "") -> MemoryPendingUpdate:
        pending_update = self.store.get_pending_update(update_id)
        if MemoryPendingUpdateStatus(pending_update.status) != MemoryPendingUpdateStatus.PENDING:
            raise ValueError(f"pending memory update is already {pending_update.status}: {update_id}")
        return self.store.update_pending_update_status(
            update_id,
            MemoryPendingUpdateStatus.REJECTED,
            reason=reason,
        )

    def promote_memory(
        self,
        memory_id: str,
        *,
        target_scope: MemoryScope | str | None = None,
        target_memory_type: MemoryType | str | None = None,
        reason: str = "",
        approved: bool = False,
    ) -> MemoryPromotionResult:
        source = self.store.get_memory(memory_id)
        resolved_scope, resolved_type = _resolve_promotion_target(
            source,
            target_scope=target_scope,
            target_memory_type=target_memory_type,
        )
        proposed = _promoted_memory_from_source(
            source,
            target_scope=resolved_scope,
            target_memory_type=resolved_type,
            reason=reason,
        )
        status, policy_reasons = _evaluate_promotion_policy(source, proposed)

        if status == MemoryPromotionStatus.BLOCKED:
            return MemoryPromotionResult(
                status=status,
                source_memory=source,
                promoted_memory=None,
                proposed_memory=proposed,
                target_scope=resolved_scope,
                target_memory_type=resolved_type,
                policy_reasons=policy_reasons,
                approved=approved,
            )

        if status == MemoryPromotionStatus.NEEDS_CONFIRMATION and not approved:
            return MemoryPromotionResult(
                status=status,
                source_memory=source,
                promoted_memory=None,
                proposed_memory=proposed,
                target_scope=resolved_scope,
                target_memory_type=resolved_type,
                policy_reasons=policy_reasons,
                approved=False,
            )

        try:
            promoted = self.store.add_memory(proposed)
            result_status = MemoryPromotionStatus.PROMOTED
        except MemoryAlreadyExistsError:
            promoted = self.store.get_memory(proposed.memory_id)
            result_status = MemoryPromotionStatus.UNCHANGED

        return MemoryPromotionResult(
            status=result_status,
            source_memory=source,
            promoted_memory=promoted,
            proposed_memory=proposed,
            target_scope=resolved_scope,
            target_memory_type=resolved_type,
            policy_reasons=policy_reasons,
            approved=approved,
        )

    def search_context(self, query: MemoryQuery) -> MemoryContext:
        return MemoryRetriever(
            self.store,
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            query_understanding=self.query_understanding,
        ).retrieve(query)

    def _merge_candidate_pool(self, proposed: MemoryRecord) -> list[MemoryRecord]:
        return self.store.list_memories(
            MemoryQuery(
                org_id=proposed.org_id,
                user_id=proposed.user_id,
                session_id=proposed.session_id,
                memory_types=[MemoryType(proposed.memory_type)],
                include_archived=True,
                include_forgotten=True,
                include_pending=True,
                limit=200,
            )
        )

    def _store_pending_update(self, pending_update: MemoryPendingUpdate) -> MemoryPendingUpdate:
        try:
            return self.store.add_pending_update(pending_update)
        except MemoryPendingUpdateAlreadyExistsError:
            return self.store.get_pending_update(pending_update.update_id)


def _memory_id_for_candidate(candidate: MemoryCandidate) -> str:
    digest = sha1(
        ":".join([candidate.session_id, candidate.decision_id, candidate.memory_candidate_id]).encode("utf-8")
    ).hexdigest()[:12]
    return f"mem_{digest}"


def _resolve_promotion_target(
    source: MemoryRecord,
    *,
    target_scope: MemoryScope | str | None,
    target_memory_type: MemoryType | str | None,
) -> tuple[MemoryScope, MemoryType]:
    if target_scope is not None:
        resolved_scope = MemoryScope(target_scope)
    elif MemoryType(source.memory_type) in {
        MemoryType.USER_PREFERENCE,
        MemoryType.NEGATIVE_PREFERENCE,
        MemoryType.PRIVACY_PREFERENCE,
    }:
        resolved_scope = MemoryScope.USER
    else:
        resolved_scope = MemoryScope.ORG

    if target_memory_type is not None:
        resolved_type = MemoryType(target_memory_type)
    elif MemoryType(source.memory_type) == MemoryType.MEETING_FACT and _has_tag_or_metadata(source, "project_context"):
        resolved_type = MemoryType.PROJECT_CONTEXT
    else:
        resolved_type = MemoryType(source.memory_type)

    return resolved_scope, resolved_type


def _promoted_memory_from_source(
    source: MemoryRecord,
    *,
    target_scope: MemoryScope,
    target_memory_type: MemoryType,
    reason: str,
) -> MemoryRecord:
    now = datetime.now(UTC)
    metadata = _non_version_metadata(source.metadata)
    metadata.update(
        {
            "promoted_from_memory_id": source.memory_id,
            "promoted_from_session_id": source.session_id,
            "promotion_reason": reason,
            "promotion_target_scope": target_scope.value,
            "promotion_target_memory_type": target_memory_type.value,
            "promoted_at": now.isoformat(),
            "memory_schema_version": "memory_promotion_v1",
        }
    )
    source_ids = _merge_unique(source.source_ids, [f"memory:{source.memory_id}"])
    tags = _merge_unique(source.tags, ["promoted", f"scope:{target_scope.value}", target_memory_type.value])
    return MemoryRecord(
        memory_id=_promoted_memory_id(source.memory_id, target_scope, target_memory_type),
        memory_type=target_memory_type,
        scope=target_scope,
        text=source.text,
        org_id=source.org_id,
        user_id=source.user_id,
        session_id=source.session_id if target_scope == MemoryScope.SESSION else None,
        source=MemorySource.PROMOTED,
        source_ids=source_ids,
        confidence=source.confidence,
        importance=round(min(1.0, max(source.importance, source.importance + 0.08)), 4),
        privacy_level=source.privacy_level,
        retention_policy=_promotion_retention_policy(target_scope, target_memory_type),
        write_status=MemoryWriteStatus.ACTIVE,
        tags=tags,
        created_at=now,
        updated_at=now,
        metadata=metadata,
    )


def _evaluate_promotion_policy(
    source: MemoryRecord,
    proposed: MemoryRecord,
) -> tuple[MemoryPromotionStatus, list[str]]:
    source_status = MemoryWriteStatus(source.write_status)
    if source_status != MemoryWriteStatus.ACTIVE:
        return MemoryPromotionStatus.BLOCKED, [f"source_memory_is_{source_status.value}"]
    if MemoryScope(source.scope) != MemoryScope.SESSION:
        return MemoryPromotionStatus.BLOCKED, ["source_memory_is_already_long_lived"]
    if MemoryType(source.memory_type) == MemoryType.ACTION_ITEM:
        return MemoryPromotionStatus.NEEDS_CONFIRMATION, ["action_items_are_session_bound_by_default"]
    if MemoryType(source.memory_type) == MemoryType.SUMMARY:
        return MemoryPromotionStatus.NEEDS_CONFIRMATION, ["meeting_summaries_need_user_confirmation"]
    if _privacy_rank(source.privacy_level) >= _privacy_rank(PrivacyLevel.HIGH):
        return MemoryPromotionStatus.NEEDS_CONFIRMATION, ["high_privacy_memory_needs_confirmation"]
    if _privacy_risk(source) >= 0.7:
        return MemoryPromotionStatus.NEEDS_CONFIRMATION, ["high_privacy_risk_memory_needs_confirmation"]
    if MemoryType(proposed.memory_type) in {
        MemoryType.USER_PREFERENCE,
        MemoryType.NEGATIVE_PREFERENCE,
        MemoryType.PRIVACY_PREFERENCE,
        MemoryType.PROJECT_CONTEXT,
        MemoryType.PERSON_OR_FACT,
    }:
        return MemoryPromotionStatus.PROMOTED, []
    if _has_tag_or_metadata(source, "promotion_candidate") or _has_tag_or_metadata(source, "long_term"):
        return MemoryPromotionStatus.PROMOTED, []
    return MemoryPromotionStatus.NEEDS_CONFIRMATION, ["memory_type_is_not_auto_promotable"]


def _promotion_retention_policy(target_scope: MemoryScope, target_memory_type: MemoryType) -> RetentionPolicy:
    if target_memory_type in {
        MemoryType.USER_PREFERENCE,
        MemoryType.NEGATIVE_PREFERENCE,
        MemoryType.PRIVACY_PREFERENCE,
    }:
        return RetentionPolicy.UNTIL_REVOKED
    if target_scope in {MemoryScope.USER, MemoryScope.ORG, MemoryScope.GLOBAL}:
        return RetentionPolicy.PERSISTENT
    return RetentionPolicy.NINETY_DAYS


def _promoted_memory_id(source_memory_id: str, target_scope: MemoryScope, target_memory_type: MemoryType) -> str:
    digest = sha1(":".join([source_memory_id, target_scope.value, target_memory_type.value]).encode("utf-8")).hexdigest()[:12]
    return f"memprom_{digest}"


def _has_tag_or_metadata(memory: MemoryRecord, value: str) -> bool:
    tags = {str(tag) for tag in memory.tags}
    metadata_values = {str(item) for item in memory.metadata.get("tags", []) if str(item)}
    metadata_values.add(str(memory.metadata.get("promotion_target", "")))
    metadata_values.add(str(memory.metadata.get("promotion_target_type", "")))
    return value in tags or value in metadata_values


def _pending_update_for_proposal(
    existing: MemoryRecord,
    proposed: MemoryRecord,
    *,
    previous_digest: str,
    new_digest: str,
    changed_fields: list[str],
    policy_reasons: list[str],
) -> MemoryPendingUpdate:
    return MemoryPendingUpdate(
        update_id=_pending_update_id(existing.memory_id, previous_digest, new_digest),
        memory_id=existing.memory_id,
        proposed_memory=proposed,
        previous_digest=previous_digest,
        new_digest=new_digest,
        changed_fields=changed_fields,
        update_policy=MemoryUpdatePolicyDecision.NEEDS_CONFIRMATION,
        policy_reasons=policy_reasons,
        metadata={
            "memory_version": _memory_version(proposed),
            "source_memory_version": _memory_version(existing),
        },
    )


def _pending_update_id(memory_id: str, previous_digest: str, new_digest: str) -> str:
    digest = sha1(":".join([memory_id, previous_digest, new_digest]).encode("utf-8")).hexdigest()[:12]
    return f"memupd_{digest}"


def _record_update_from_memory(memory: MemoryRecord) -> MemoryRecordUpdate:
    return MemoryRecordUpdate(
        memory_type=memory.memory_type,
        scope=memory.scope,
        text=memory.text,
        org_id=memory.org_id,
        user_id=memory.user_id,
        session_id=memory.session_id,
        source=memory.source,
        source_ids=memory.source_ids,
        confidence=memory.confidence,
        importance=memory.importance,
        privacy_level=memory.privacy_level,
        retention_policy=memory.retention_policy,
        write_status=memory.write_status,
        tags=memory.tags,
        metadata=memory.metadata,
    )


def _memory_type_for_candidate(candidate_type: MemoryCandidateType) -> MemoryType:
    return {
        MemoryCandidateType.USER_PREFERENCE: MemoryType.USER_PREFERENCE,
        MemoryCandidateType.NEGATIVE_PREFERENCE: MemoryType.NEGATIVE_PREFERENCE,
        MemoryCandidateType.PRIVACY_PREFERENCE: MemoryType.PRIVACY_PREFERENCE,
        MemoryCandidateType.MEETING_FACT: MemoryType.MEETING_FACT,
        MemoryCandidateType.ACTION_ITEM: MemoryType.ACTION_ITEM,
        MemoryCandidateType.DECISION: MemoryType.DECISION,
        MemoryCandidateType.PERSON_OR_FACT: MemoryType.PERSON_OR_FACT,
        MemoryCandidateType.PROJECT_CONTEXT: MemoryType.PROJECT_CONTEXT,
        MemoryCandidateType.SUMMARY: MemoryType.SUMMARY,
    }[candidate_type]


def _scope_for_candidate(candidate_type: MemoryCandidateType) -> MemoryScope:
    if candidate_type in {
        MemoryCandidateType.USER_PREFERENCE,
        MemoryCandidateType.NEGATIVE_PREFERENCE,
        MemoryCandidateType.PRIVACY_PREFERENCE,
    }:
        return MemoryScope.USER
    return MemoryScope.SESSION


def _write_status_for_policy(policy: MemoryWritePolicy) -> MemoryWriteStatus:
    return {
        MemoryWritePolicy.ELIGIBLE: MemoryWriteStatus.ACTIVE,
        MemoryWritePolicy.NEEDS_CONFIRMATION: MemoryWriteStatus.PENDING_CONFIRMATION,
        MemoryWritePolicy.BLOCKED: MemoryWriteStatus.REJECTED,
    }[policy]


def _retention_for_candidate(candidate_type: MemoryCandidateType) -> RetentionPolicy:
    if candidate_type in {
        MemoryCandidateType.USER_PREFERENCE,
        MemoryCandidateType.NEGATIVE_PREFERENCE,
        MemoryCandidateType.PRIVACY_PREFERENCE,
    }:
        return RetentionPolicy.UNTIL_REVOKED
    if candidate_type == MemoryCandidateType.SUMMARY:
        return RetentionPolicy.THIRTY_DAYS
    return RetentionPolicy.NINETY_DAYS


def _privacy_for_candidate(candidate: MemoryCandidate) -> PrivacyLevel:
    candidate_type = MemoryCandidateType(candidate.candidate_type)
    if candidate_type == MemoryCandidateType.PRIVACY_PREFERENCE:
        return PrivacyLevel.HIGH
    return PrivacyLevel(candidate.privacy_level)


def _importance_for_candidate(candidate_type: MemoryCandidateType, confidence: float) -> float:
    base = {
        MemoryCandidateType.USER_PREFERENCE: 0.82,
        MemoryCandidateType.NEGATIVE_PREFERENCE: 0.86,
        MemoryCandidateType.PRIVACY_PREFERENCE: 0.95,
        MemoryCandidateType.MEETING_FACT: 0.62,
        MemoryCandidateType.ACTION_ITEM: 0.74,
        MemoryCandidateType.DECISION: 0.78,
        MemoryCandidateType.PERSON_OR_FACT: 0.68,
        MemoryCandidateType.PROJECT_CONTEXT: 0.76,
        MemoryCandidateType.SUMMARY: 0.7,
    }[candidate_type]
    return round(min(1.0, max(0.0, 0.7 * base + 0.3 * confidence)), 4)


def _source_ids(candidate: MemoryCandidate) -> list[str]:
    ids = [
        f"memory_candidate:{candidate.memory_candidate_id}",
        f"decision:{candidate.decision_id}",
    ]
    ids.extend(f"feedback:{event_id}" for event_id in candidate.source_event_ids)
    return ids


def _tags_for_candidate(candidate_type: MemoryCandidateType, candidate: MemoryCandidate) -> list[str]:
    tags = {"feedback_candidate", candidate_type.value}
    if candidate_type in {MemoryCandidateType.USER_PREFERENCE, MemoryCandidateType.NEGATIVE_PREFERENCE}:
        tags.add("preference")
    if candidate_type == MemoryCandidateType.PRIVACY_PREFERENCE:
        tags.update({"preference", "privacy"})
    if candidate_type == MemoryCandidateType.ACTION_ITEM:
        tags.add("action")
    if candidate_type == MemoryCandidateType.DECISION:
        tags.add("decision")
    if candidate_type == MemoryCandidateType.PERSON_OR_FACT:
        tags.add("person_or_fact")
    if candidate_type == MemoryCandidateType.PROJECT_CONTEXT:
        tags.update({"project_context", "promotion_candidate"})
    if candidate_type == MemoryCandidateType.SUMMARY:
        tags.add("summary")
    tags.update(str(tag) for tag in candidate.metadata.get("tags", []) if str(tag))
    return consolidated_memory_tags(candidate, tags)


_VERSION_METADATA_KEYS = {
    "memory_content_digest",
    "memory_last_candidate_id",
    "memory_last_decision_id",
    "memory_last_upserted_at",
    "memory_previous_digest",
    "memory_upsert_source",
    "memory_version",
    "memory_version_history",
}


def _merge_upsert_memory(existing: MemoryRecord, proposed: MemoryRecord) -> MemoryRecord:
    metadata = _non_version_metadata(existing.metadata)
    metadata.update(_non_version_metadata(proposed.metadata))
    return proposed.model_copy(
        update={
            "created_at": existing.created_at,
            "last_accessed_at": existing.last_accessed_at,
            "source_ids": _merge_unique(existing.source_ids, proposed.source_ids),
            "write_status": _resolved_write_status(existing, proposed),
            "tags": _merge_unique(existing.tags, proposed.tags),
            "metadata": metadata,
        }
    )


def _reinforced_memory(
    existing: MemoryRecord,
    proposed: MemoryRecord,
    merge_decision: MemoryMergeDecision,
) -> MemoryRecord:
    reinforced = existing.model_copy(
        update={
            "source_ids": _merge_unique(existing.source_ids, proposed.source_ids),
            "confidence": round(min(1.0, max(existing.confidence, proposed.confidence) + 0.03), 4),
            "importance": round(min(1.0, max(existing.importance, proposed.importance) + 0.03), 4),
            "tags": _merge_unique(existing.tags, proposed.tags),
            "metadata": _merge_metadata_with_decision(existing.metadata, proposed.metadata, merge_decision),
        }
    )
    return MemoryRecord.model_validate(reinforced.model_dump(mode="python"))


def _memory_with_merge_metadata(memory: MemoryRecord, merge_decision: MemoryMergeDecision) -> MemoryRecord:
    metadata = _merge_metadata_with_decision(memory.metadata, {}, merge_decision)
    return memory.model_copy(update={"metadata": metadata})


def _merge_metadata_with_decision(
    base_metadata: dict[str, Any],
    proposed_metadata: dict[str, Any],
    merge_decision: MemoryMergeDecision,
) -> dict[str, Any]:
    metadata = _non_version_metadata(base_metadata)
    for key, value in _non_version_metadata(proposed_metadata).items():
        if key not in metadata:
            metadata[key] = deepcopy(value)
    history = metadata.get("memory_merge_history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "action": str(merge_decision.action),
            "existing_memory_id": merge_decision.existing_memory_id,
            "proposed_memory_id": merge_decision.proposed_memory_id,
            "similarity": merge_decision.similarity,
            "changed_fields": list(merge_decision.changed_fields),
            "conflict_fields": list(merge_decision.conflict_fields),
            "evidence_refs": list(merge_decision.evidence_refs),
            "reasons": list(merge_decision.reasons),
            "resolution_status": merge_decision.resolution_status,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    metadata["memory_merge_history"] = history[-20:]
    metadata["memory_last_merge_action"] = str(merge_decision.action)
    metadata["memory_last_merge_resolution_status"] = merge_decision.resolution_status
    return metadata


def _resolved_write_status(existing: MemoryRecord, proposed: MemoryRecord) -> str:
    existing_status = MemoryWriteStatus(existing.write_status)
    proposed_status = MemoryWriteStatus(proposed.write_status)
    if existing_status == MemoryWriteStatus.ACTIVE and proposed_status == MemoryWriteStatus.PENDING_CONFIRMATION:
        return MemoryWriteStatus.ACTIVE.value
    return proposed_status.value


def _evaluate_update_policy(
    existing: MemoryRecord,
    desired: MemoryRecord,
    changed_fields: list[str],
) -> tuple[MemoryUpdatePolicyDecision, list[str]]:
    if not changed_fields:
        return MemoryUpdatePolicyDecision.AUTO_UPDATE, []

    reasons: list[str] = []
    existing_status = MemoryWriteStatus(existing.write_status)
    desired_status = MemoryWriteStatus(desired.write_status)
    if existing_status in {MemoryWriteStatus.ARCHIVED, MemoryWriteStatus.REJECTED, MemoryWriteStatus.FORGOTTEN}:
        return MemoryUpdatePolicyDecision.BLOCKED, [f"existing_memory_is_{existing_status.value}"]
    if desired_status == MemoryWriteStatus.REJECTED:
        return MemoryUpdatePolicyDecision.BLOCKED, ["proposed_update_is_rejected"]

    sensitive_fields = {
        "metadata.owner",
        "metadata.assignee",
        "metadata.deadline",
        "metadata.normalized_deadline",
        "metadata.privacy_level",
        "privacy_level",
    }
    if sensitive_fields.intersection(changed_fields):
        reasons.append("sensitive_structured_field_changed")
    if MemoryType(existing.memory_type) in {MemoryType.USER_PREFERENCE, MemoryType.PRIVACY_PREFERENCE}:
        reasons.append("long_lived_preference_update")
    if _privacy_rank(desired.privacy_level) >= _privacy_rank(PrivacyLevel.HIGH):
        reasons.append("high_privacy_update")
    if _privacy_risk(desired) >= 0.7:
        reasons.append("high_privacy_risk_update")
    if existing_status == MemoryWriteStatus.ACTIVE and desired_status == MemoryWriteStatus.PENDING_CONFIRMATION:
        reasons.append("active_memory_would_be_downgraded_to_pending")

    if reasons:
        return MemoryUpdatePolicyDecision.NEEDS_CONFIRMATION, sorted(set(reasons))
    return MemoryUpdatePolicyDecision.AUTO_UPDATE, []


def _privacy_rank(value: PrivacyLevel | str) -> int:
    level = PrivacyLevel(value)
    return {
        PrivacyLevel.LOW: 0,
        PrivacyLevel.MEDIUM: 1,
        PrivacyLevel.HIGH: 2,
    }[level]


def _ensure_not_forgotten(memory: MemoryRecord, *, operation: str) -> None:
    if MemoryWriteStatus(memory.write_status) == MemoryWriteStatus.FORGOTTEN:
        raise ValueError(f"cannot {operation} forgotten memory: {memory.memory_id}")


def _pending_rejection_reason_for_forget(reason: str) -> str:
    return f"memory forgotten: {reason}" if reason else "memory forgotten"


def _privacy_risk(memory: MemoryRecord) -> float:
    raw = memory.metadata.get("privacy_risk", memory.metadata.get("privacy_risk_score", 0.0))
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def _memory_with_version_metadata(
    memory: MemoryRecord,
    *,
    version: int,
    digest: str,
    previous_digest: str | None,
    changed_fields: list[str],
    previous_history: list[dict[str, Any]],
) -> tuple[MemoryRecord, str]:
    metadata = _non_version_metadata(memory.metadata)
    history = list(previous_history)
    if changed_fields and changed_fields != ["version_metadata"]:
        history.append(
            {
                "version": version,
                "changed_at": datetime.now(UTC).isoformat(),
                "changed_fields": list(changed_fields),
                "previous_digest": previous_digest,
                "new_digest": digest,
            }
        )
    metadata.update(
        {
            "memory_version": version,
            "memory_content_digest": digest,
            "memory_previous_digest": previous_digest,
            "memory_version_history": history[-20:],
            "memory_last_upserted_at": datetime.now(UTC).isoformat(),
            "memory_last_candidate_id": memory.source_ids[0] if memory.source_ids else "",
            "memory_last_decision_id": _decision_source_id(memory.source_ids),
            "memory_upsert_source": "candidate",
        }
    )
    return memory.model_copy(update={"metadata": metadata}), digest


def _memory_content_digest(memory: MemoryRecord) -> str:
    payload = {
        "memory_type": str(memory.memory_type),
        "scope": str(memory.scope),
        "text": memory.text,
        "org_id": memory.org_id,
        "user_id": memory.user_id,
        "session_id": memory.session_id,
        "source": str(memory.source),
        "source_ids": sorted(memory.source_ids),
        "confidence": round(memory.confidence, 6),
        "importance": round(memory.importance, 6),
        "privacy_level": str(memory.privacy_level),
        "retention_policy": str(memory.retention_policy),
        "write_status": str(memory.write_status),
        "tags": sorted(memory.tags),
        "metadata": _canonicalize(_non_version_metadata(memory.metadata)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha1(encoded.encode("utf-8")).hexdigest()


def _changed_fields(existing: MemoryRecord, desired: MemoryRecord) -> list[str]:
    changed: list[str] = []
    scalar_fields = (
        "memory_type",
        "scope",
        "text",
        "org_id",
        "user_id",
        "session_id",
        "source",
        "source_ids",
        "confidence",
        "importance",
        "privacy_level",
        "retention_policy",
        "write_status",
        "tags",
    )
    for field in scalar_fields:
        if getattr(existing, field) != getattr(desired, field):
            changed.append(field)
    existing_metadata = _non_version_metadata(existing.metadata)
    desired_metadata = _non_version_metadata(desired.metadata)
    for key in sorted(set(existing_metadata) | set(desired_metadata)):
        if existing_metadata.get(key) != desired_metadata.get(key):
            changed.append(f"metadata.{key}")
    return changed


def _non_version_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in metadata.items() if key not in _VERSION_METADATA_KEYS}


def _memory_version(memory: MemoryRecord) -> int:
    raw_version = memory.metadata.get("memory_version", 1)
    try:
        return max(1, int(raw_version))
    except (TypeError, ValueError):
        return 1


def _memory_version_history(memory: MemoryRecord) -> list[dict[str, Any]]:
    raw_history = memory.metadata.get("memory_version_history", [])
    if not isinstance(raw_history, list):
        return []
    return [dict(item) for item in raw_history if isinstance(item, dict)]


def _decision_source_id(source_ids: list[str]) -> str:
    for source_id in source_ids:
        if source_id.startswith("decision:"):
            return source_id
    return ""


def _merge_unique(first: list[str], second: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in [*first, *second]:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
