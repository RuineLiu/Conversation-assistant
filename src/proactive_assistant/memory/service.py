from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha1

from proactive_assistant.memory.contracts import (
    MemoryContext,
    MemoryQuery,
    MemoryRecord,
    MemoryRecordUpdate,
    MemoryScope,
    MemorySearchResult,
    MemorySource,
    MemoryType,
    MemoryWriteStatus,
    RetentionPolicy,
)
from proactive_assistant.memory.store import MemoryStore
from proactive_assistant.prompting import PrivacyLevel
from proactive_assistant.runtime import MemoryCandidate, MemoryCandidateType, MemoryWritePolicy


class MemoryService:
    """Lifecycle service for converting candidates into long-term memories."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

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
            metadata={
                "candidate_id": candidate.memory_candidate_id,
                "candidate_type": str(candidate.candidate_type),
                "decision_id": candidate.decision_id,
                "source_event_ids": list(candidate.source_event_ids),
                "write_policy": str(candidate.write_policy),
                "reason": candidate.reason,
                "candidate_metadata": candidate.metadata,
            },
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

    def confirm_memory(self, memory_id: str) -> MemoryRecord:
        return self.store.update_memory(memory_id, MemoryRecordUpdate(write_status=MemoryWriteStatus.ACTIVE))

    def reject_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        memory = self.store.get_memory(memory_id)
        metadata = dict(memory.metadata)
        if reason:
            metadata["reject_reason"] = reason
        return self.store.update_memory(
            memory_id,
            MemoryRecordUpdate(write_status=MemoryWriteStatus.REJECTED, metadata=metadata),
        )

    def archive_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        return self.store.archive_memory(memory_id, reason=reason)

    def search_context(self, query: MemoryQuery) -> MemoryContext:
        results = self.store.search_memories(query)
        return MemoryContext(
            memory_context=[_context_line(result) for result in results],
            memory_refs=[f"memory:{result.memory.memory_id}" for result in results],
            results=results,
        )


def _memory_id_for_candidate(candidate: MemoryCandidate) -> str:
    digest = sha1(
        ":".join([candidate.session_id, candidate.decision_id, candidate.memory_candidate_id]).encode("utf-8")
    ).hexdigest()[:12]
    return f"mem_{digest}"


def _memory_type_for_candidate(candidate_type: MemoryCandidateType) -> MemoryType:
    return {
        MemoryCandidateType.USER_PREFERENCE: MemoryType.USER_PREFERENCE,
        MemoryCandidateType.NEGATIVE_PREFERENCE: MemoryType.NEGATIVE_PREFERENCE,
        MemoryCandidateType.PRIVACY_PREFERENCE: MemoryType.PRIVACY_PREFERENCE,
        MemoryCandidateType.MEETING_FACT: MemoryType.MEETING_FACT,
        MemoryCandidateType.ACTION_ITEM: MemoryType.ACTION_ITEM,
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
    tags.update(str(tag) for tag in candidate.metadata.get("tags", []) if str(tag))
    return sorted(tags)


def _context_line(result: MemorySearchResult) -> str:
    memory = result.memory
    return f"[memory:{memory.memory_id}] ({memory.memory_type}/{memory.scope}) {memory.text}"
