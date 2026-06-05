from __future__ import annotations

import re
from copy import deepcopy
from datetime import UTC, datetime
from typing import Protocol

from proactive_assistant.memory.contracts import (
    MemoryQuery,
    MemoryRecord,
    MemoryRecordUpdate,
    MemoryScope,
    MemorySearchResult,
    MemoryWriteStatus,
)


class MemoryStoreError(RuntimeError):
    """Base error for memory store operations."""


class MemoryAlreadyExistsError(MemoryStoreError):
    """Raised when inserting a duplicate memory id."""


class MemoryNotFoundError(MemoryStoreError):
    """Raised when a memory id does not exist."""


class MemoryStore(Protocol):
    def add_memory(self, memory: MemoryRecord) -> MemoryRecord: ...

    def get_memory(self, memory_id: str) -> MemoryRecord: ...

    def list_memories(self, query: MemoryQuery | None = None) -> list[MemoryRecord]: ...

    def search_memories(self, query: MemoryQuery) -> list[MemorySearchResult]: ...

    def update_memory(self, memory_id: str, update: MemoryRecordUpdate) -> MemoryRecord: ...

    def archive_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord: ...


class InMemoryMemoryStore:
    """Deterministic in-memory memory store for early product integration."""

    def __init__(self) -> None:
        self._memories: dict[str, MemoryRecord] = {}
        self._order: list[str] = []

    def add_memory(self, memory: MemoryRecord) -> MemoryRecord:
        if memory.memory_id in self._memories:
            raise MemoryAlreadyExistsError(f"memory already exists: {memory.memory_id}")
        self._memories[memory.memory_id] = deepcopy(memory)
        self._order.append(memory.memory_id)
        return deepcopy(memory)

    def get_memory(self, memory_id: str) -> MemoryRecord:
        memory = self._get(memory_id)
        now = datetime.now(UTC)
        memory = memory.model_copy(update={"last_accessed_at": now})
        self._memories[memory_id] = memory
        return deepcopy(memory)

    def list_memories(self, query: MemoryQuery | None = None) -> list[MemoryRecord]:
        resolved_query = query or MemoryQuery(limit=200)
        memories = [self._memories[memory_id] for memory_id in self._order]
        filtered = [memory for memory in memories if _matches_query(memory, resolved_query)]
        return [deepcopy(memory) for memory in _sort_records(filtered)[: resolved_query.limit]]

    def search_memories(self, query: MemoryQuery) -> list[MemorySearchResult]:
        query_terms = _terms(query.query_text)
        results: list[MemorySearchResult] = []
        for memory in self.list_memories(query):
            matched_terms = sorted(query_terms & _terms(" ".join([memory.text, *memory.tags])))
            if query_terms and not matched_terms:
                continue
            score = _score_memory(memory, matched_terms, query_terms)
            results.append(
                MemorySearchResult(
                    memory=memory,
                    score=score,
                    matched_terms=matched_terms,
                    reason=_score_reason(matched_terms, query_terms),
                )
            )
        return sorted(
            results,
            key=lambda item: (
                -item.score,
                -item.memory.importance,
                -item.memory.confidence,
                -_created_ts(item.memory),
                item.memory.memory_id,
            ),
        )[: query.limit]

    def update_memory(self, memory_id: str, update: MemoryRecordUpdate) -> MemoryRecord:
        memory = self._get(memory_id)
        updates = update.model_dump(exclude_none=True, mode="python")
        if not updates:
            return deepcopy(memory)
        updates["updated_at"] = datetime.now(UTC)
        updated = memory.model_copy(update=updates)
        # Re-validate cross-field constraints such as session scope requiring session_id.
        updated = MemoryRecord.model_validate(updated.model_dump(mode="python"))
        self._memories[memory_id] = updated
        return deepcopy(updated)

    def archive_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        memory = self._get(memory_id)
        metadata = dict(memory.metadata)
        if reason:
            metadata["archive_reason"] = reason
        return self.update_memory(
            memory_id,
            MemoryRecordUpdate(write_status=MemoryWriteStatus.ARCHIVED, metadata=metadata),
        )

    def _get(self, memory_id: str) -> MemoryRecord:
        try:
            return self._memories[memory_id]
        except KeyError as exc:
            raise MemoryNotFoundError(f"memory not found: {memory_id}") from exc


def _matches_query(memory: MemoryRecord, query: MemoryQuery) -> bool:
    if not query.include_archived and memory.write_status == MemoryWriteStatus.ARCHIVED.value:
        return False
    if not query.include_pending and memory.write_status == MemoryWriteStatus.PENDING_CONFIRMATION.value:
        return False
    if memory.write_status == MemoryWriteStatus.REJECTED.value:
        return False
    if query.org_id is not None and memory.org_id != query.org_id:
        return False
    if query.user_id is not None and memory.user_id != query.user_id:
        return False
    if query.session_id is not None and not _visible_to_session(memory, query.session_id):
        return False
    if query.memory_types and memory.memory_type not in set(query.memory_types):
        return False
    if query.scopes and memory.scope not in set(query.scopes):
        return False
    if query.privacy_levels and memory.privacy_level not in set(query.privacy_levels):
        return False
    if query.source_ids and not set(query.source_ids).intersection(memory.source_ids):
        return False
    if query.tags and not set(query.tags).issubset(set(memory.tags)):
        return False
    return True


def _visible_to_session(memory: MemoryRecord, session_id: str) -> bool:
    if memory.scope == MemoryScope.SESSION.value:
        return memory.session_id == session_id
    return True


def _sort_records(memories: list[MemoryRecord]) -> list[MemoryRecord]:
    return sorted(
        memories,
        key=lambda memory: (
            _status_rank(memory),
            -memory.importance,
            -memory.confidence,
            -_created_ts(memory),
            memory.memory_id,
        ),
    )


def _created_ts(memory: MemoryRecord) -> float:
    return memory.created_at.timestamp()


def _status_rank(memory: MemoryRecord) -> int:
    return {
        MemoryWriteStatus.ACTIVE.value: 0,
        MemoryWriteStatus.PENDING_CONFIRMATION.value: 1,
        MemoryWriteStatus.ARCHIVED.value: 2,
        MemoryWriteStatus.REJECTED.value: 3,
    }.get(memory.write_status, 4)


def _score_memory(memory: MemoryRecord, matched_terms: list[str], query_terms: set[str]) -> float:
    lexical = len(matched_terms) / max(len(query_terms), 1)
    score = 0.55 * lexical + 0.25 * memory.importance + 0.20 * memory.confidence
    return round(score, 4)


def _score_reason(matched_terms: list[str], query_terms: set[str]) -> str:
    if not query_terms:
        return "ranked by importance and confidence"
    return "keyword_overlap" if matched_terms else "no_keyword_overlap"


def _terms(text: str) -> set[str]:
    normalized = text.lower()
    ascii_terms = re.findall(r"[a-z0-9_]+", normalized)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    return {term for term in [*ascii_terms, *chinese_terms] if term}
