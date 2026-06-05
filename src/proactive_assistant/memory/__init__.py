"""Long-term memory contracts and stores for proactive assistant."""

from proactive_assistant.memory.contracts import (
    MemoryContext,
    MemoryQuery,
    MemoryRecord,
    MemoryRecordUpdate,
    MemoryRetrievalIntent,
    MemoryScope,
    MemorySearchResult,
    MemorySource,
    MemoryType,
    MemoryUsePolicy,
    MemoryWriteStatus,
    RetentionPolicy,
)
from proactive_assistant.memory.retrieval import MemoryRetriever, build_structured_query
from proactive_assistant.memory.service import MemoryService
from proactive_assistant.memory.snapshot import memory_candidates_from_meeting_state
from proactive_assistant.memory.store import (
    InMemoryMemoryStore,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryRepository,
    MemoryStore,
    MemoryStoreError,
)

__all__ = [
    "InMemoryMemoryStore",
    "MemoryAlreadyExistsError",
    "MemoryContext",
    "MemoryNotFoundError",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryRecordUpdate",
    "MemoryRetrievalIntent",
    "MemoryRetriever",
    "MemoryRepository",
    "MemoryScope",
    "MemorySearchResult",
    "MemoryService",
    "MemorySource",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryType",
    "MemoryUsePolicy",
    "MemoryWriteStatus",
    "RetentionPolicy",
    "build_structured_query",
    "memory_candidates_from_meeting_state",
]
