from __future__ import annotations

from datetime import UTC, datetime

from proactive_assistant.memory import InMemoryMemoryVectorStore, MemoryVectorRecord
from proactive_assistant.persistence import SQLiteMemoryVectorStore


def test_in_memory_vector_store_upserts_and_deletes_embedding() -> None:
    store = InMemoryMemoryVectorStore()
    record = _vector_record("mem_001", embedding=[0.1, 0.2, 0.3])

    stored = store.upsert_embedding(record)
    deleted = store.delete_embedding("mem_001", "embedding-test")

    assert stored.memory_id == "mem_001"
    assert store.get_embedding("mem_001", "embedding-test") is None
    assert deleted == 1


def test_sqlite_vector_store_persists_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "vectors.db"
    SQLiteMemoryVectorStore(db_path).upsert_embedding(_vector_record("mem_001", embedding=[0.1, 0.2, 0.3]))

    reopened = SQLiteMemoryVectorStore(db_path)
    stored = reopened.get_embedding("mem_001", "embedding-test")

    assert stored is not None
    assert stored.embedding == [0.1, 0.2, 0.3]
    assert stored.content_hash == "hash_001"


def _vector_record(memory_id: str, *, embedding: list[float]) -> MemoryVectorRecord:
    return MemoryVectorRecord(
        memory_id=memory_id,
        embedding_model="embedding-test",
        dimensions=len(embedding),
        embedding=embedding,
        content_hash="hash_001",
        updated_at=datetime(2026, 6, 5, tzinfo=UTC),
    )
