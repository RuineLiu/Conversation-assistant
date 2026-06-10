from proactive_assistant.detection.vocabulary import PersonalVocabularyService
from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemorySource,
    MemoryType,
)


def make_memory_service() -> MemoryService:
    return MemoryService(InMemoryMemoryStore())


def add_user_scope_term(
    service: MemoryService,
    *,
    memory_id: str,
    term: str,
    org_id: str = "org_001",
    user_id: str = "user_001",
    memory_type: MemoryType = MemoryType.PERSON_OR_FACT,
    extra_tags: list[str] | None = None,
) -> None:
    service.store.add_memory(
        MemoryRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            scope=MemoryScope.USER,
            text=f"{term} 是用户已知的术语。",
            org_id=org_id,
            user_id=user_id,
            source=MemorySource.MANUAL,
            confidence=0.9,
            importance=0.7,
            tags=extra_tags or [],
            metadata={"canonical_entity": term, "entity": term},
        )
    )


def add_session_explanation(
    service: MemoryService,
    *,
    memory_id: str,
    term: str,
    explanation: str,
    session_id: str = "session_001",
) -> None:
    service.store.add_memory(
        MemoryRecord(
            memory_id=memory_id,
            memory_type=MemoryType.MEETING_FACT,
            scope=MemoryScope.SESSION,
            text=explanation,
            session_id=session_id,
            source=MemorySource.TRANSCRIPT,
            confidence=0.8,
            importance=0.6,
            tags=["term_explanation"],
            metadata={"canonical_entity": term, "entity": term},
        )
    )


def test_returns_empty_when_no_memory_service() -> None:
    vocab = PersonalVocabularyService(None)

    assert vocab.get_known_vocabulary(session_id="session_001") == []
    assert vocab.get_explained_terms_for_session(session_id="session_001") == []


def test_returns_user_scope_known_terms() -> None:
    service = make_memory_service()
    add_user_scope_term(service, memory_id="mem_okr", term="OKR")
    add_user_scope_term(service, memory_id="mem_kpi", term="KPI")
    vocab = PersonalVocabularyService(service)

    known = vocab.get_known_vocabulary(
        session_id="session_001",
        org_id="org_001",
        user_id="user_001",
    )

    assert "OKR" in known
    assert "KPI" in known


def test_known_vocabulary_dedupes_normalized_entities() -> None:
    service = make_memory_service()
    add_user_scope_term(service, memory_id="mem_a", term="OKR")
    add_user_scope_term(service, memory_id="mem_b", term="okr")
    vocab = PersonalVocabularyService(service)

    known = vocab.get_known_vocabulary(
        session_id="session_001",
        org_id="org_001",
        user_id="user_001",
    )

    normalized = [term.lower() for term in known]
    assert normalized.count("okr") == 1


def test_session_scope_explanations_only_show_for_matching_session() -> None:
    service = make_memory_service()
    add_session_explanation(
        service, memory_id="mem_a", term="GMV", explanation="商品交易总额。", session_id="session_001"
    )
    add_session_explanation(
        service, memory_id="mem_b", term="ARR", explanation="年度经常性收入。", session_id="session_other"
    )
    vocab = PersonalVocabularyService(service)

    explained_a = vocab.get_explained_terms_for_session(session_id="session_001")
    explained_other = vocab.get_explained_terms_for_session(session_id="session_other")

    assert explained_a == ["GMV"]
    assert explained_other == ["ARR"]


def test_known_vocabulary_isolated_across_users() -> None:
    service = make_memory_service()
    add_user_scope_term(service, memory_id="mem_u1", term="OKR", user_id="user_001")
    add_user_scope_term(service, memory_id="mem_u2", term="KPI", user_id="user_002")
    vocab = PersonalVocabularyService(service)

    user1_terms = vocab.get_known_vocabulary(
        session_id="session_001", user_id="user_001", org_id="org_001"
    )
    user2_terms = vocab.get_known_vocabulary(
        session_id="session_001", user_id="user_002", org_id="org_001"
    )

    assert "OKR" in user1_terms and "KPI" not in user1_terms
    assert "KPI" in user2_terms and "OKR" not in user2_terms
