from __future__ import annotations

from datetime import UTC, datetime

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    InMemoryMemoryVectorStore,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemorySource,
    MemoryType,
)
from proactive_assistant.model_gateway import FakeEmbeddingClient


def test_semantic_retrieval_recalls_memory_without_keyword_overlap() -> None:
    memory_store = InMemoryMemoryStore()
    vector_store = InMemoryMemoryVectorStore()
    service = MemoryService(
        memory_store,
        embedding_client=FakeEmbeddingClient(_semantic_fixture_embedding, dimensions=4),
        vector_store=vector_store,
        embedding_model="fixture-embedding",
    )
    memory_store.add_memory(
        _memory(
            "mem_legal_contact",
            "李四是法务接口，负责客户数据合规风险确认。",
            memory_type=MemoryType.PERSON_OR_FACT,
            tags=["法务", "客户数据", "合规"],
        )
    )
    memory_store.add_memory(
        _memory(
            "mem_parking",
            "访客停车需要在前台登记车牌。",
            memory_type=MemoryType.MEETING_FACT,
            tags=["行政"],
        )
    )

    context = service.search_context(
        MemoryQuery(
            query_text="合同隐私条款归属",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            use_semantic_retrieval=True,
            semantic_min_score=0.5,
            limit=2,
        )
    )

    assert context.memory_refs[0] == "memory:mem_legal_contact"
    assert context.results[0].rank_features["semantic"] == 1.0
    assert context.results[0].reason == "open_recall: semantic hybrid ranking"
    assert vector_store.get_embedding("mem_legal_contact", "fixture-embedding") is not None


def test_semantic_retrieval_reuses_cached_memory_embeddings() -> None:
    memory_store = InMemoryMemoryStore()
    vector_store = InMemoryMemoryVectorStore()
    embedding_client = FakeEmbeddingClient(_semantic_fixture_embedding, dimensions=4)
    service = MemoryService(
        memory_store,
        embedding_client=embedding_client,
        vector_store=vector_store,
        embedding_model="fixture-embedding",
    )
    memory_store.add_memory(
        _memory(
            "mem_decision",
            "MVP 版本先保留手动查看入口，自动推送后续再开。",
            memory_type=MemoryType.DECISION,
            tags=["MVP", "手动入口"],
        )
    )
    query = MemoryQuery(
        query_text="为什么现在不自动弹出？",
        org_id="org_001",
        user_id="user_001",
        session_id="session_001",
        use_semantic_retrieval=True,
    )

    assert service.search_context(query).memory_refs == ["memory:mem_decision"]
    assert service.search_context(query).memory_refs == ["memory:mem_decision"]

    # First call embeds query + memory. Second call embeds query only because
    # the memory vector is persisted in the vector store.
    assert [len(texts) for _, texts in embedding_client.requests] == [1, 1, 1]


def _memory(
    memory_id: str,
    text: str,
    *,
    memory_type: MemoryType,
    tags: list[str],
) -> MemoryRecord:
    created = datetime(2026, 6, 5, tzinfo=UTC)
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=memory_type,
        scope=MemoryScope.SESSION,
        text=text,
        org_id="org_001",
        user_id="user_001",
        session_id="session_001",
        source=MemorySource.MANUAL,
        source_ids=[f"manual:{memory_id}"],
        confidence=0.9,
        importance=0.85,
        tags=tags,
        created_at=created,
        updated_at=created,
    )


def _semantic_fixture_embedding(text: str) -> list[float]:
    lowered = text.lower()
    vector = [0.0, 0.0, 0.0, 0.0]
    if any(term in text for term in ["法务", "合规", "合同", "隐私", "条款", "客户数据"]):
        vector[0] = 1.0
    if any(term in text for term in ["停车", "车牌", "前台"]):
        vector[1] = 1.0
    if any(term in text for term in ["手动", "自动", "弹出", "推送", "入口"]) or "mvp" in lowered:
        vector[2] = 1.0
    if not any(vector):
        vector[3] = 1.0
    return vector
