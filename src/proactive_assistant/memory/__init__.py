"""Long-term memory contracts and stores for proactive assistant."""

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
from proactive_assistant.memory.service import MemoryService
from proactive_assistant.memory.store import (
    InMemoryMemoryStore,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
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
    "MemoryScope",
    "MemorySearchResult",
    "MemoryService",
    "MemorySource",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryType",
    "MemoryWriteStatus",
    "RetentionPolicy",
]
