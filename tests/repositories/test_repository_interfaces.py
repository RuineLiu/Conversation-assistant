from proactive_assistant.memory import InMemoryMemoryStore, MemoryRepository, MemoryStore
from proactive_assistant.repositories import (
    MemoryRepository as ExportedMemoryRepository,
    RuntimeRepository as ExportedRuntimeRepository,
    SessionRepository as ExportedSessionRepository,
)
from proactive_assistant.runtime import InMemoryRuntimeStore, RuntimeRepository, RuntimeStore
from proactive_assistant.sessions import InMemorySessionStore, SessionRepository, SessionStore


def test_in_memory_stores_satisfy_repository_protocols() -> None:
    assert isinstance(InMemorySessionStore(), SessionRepository)
    assert isinstance(InMemoryRuntimeStore(), RuntimeRepository)
    assert isinstance(InMemoryMemoryStore(), MemoryRepository)


def test_legacy_store_aliases_point_to_repository_protocols() -> None:
    assert SessionStore is SessionRepository
    assert RuntimeStore is RuntimeRepository
    assert MemoryStore is MemoryRepository


def test_repository_package_exports_protocols() -> None:
    assert ExportedSessionRepository is SessionRepository
    assert ExportedRuntimeRepository is RuntimeRepository
    assert ExportedMemoryRepository is MemoryRepository
