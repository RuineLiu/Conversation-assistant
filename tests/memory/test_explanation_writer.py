import pytest

from proactive_assistant.detection.vocabulary import PersonalVocabularyService
from proactive_assistant.memory import (
    ExplanationMemoryWriter,
    InMemoryMemoryStore,
    MemoryQuery,
    MemoryService,
    MemoryType,
)
from proactive_assistant.prompting import PrivacyLevel


def make_service() -> tuple[MemoryService, ExplanationMemoryWriter]:
    service = MemoryService(InMemoryMemoryStore())
    writer = ExplanationMemoryWriter(service)
    return service, writer


def test_commit_explanation_persists_session_scope_meeting_fact_with_term_metadata() -> None:
    service, writer = make_service()

    record = writer.commit_explanation(
        session_id="session_001",
        org_id="org_001",
        user_id="user_001",
        term="GMV",
        explanation="商品交易总额，电商核心指标。",
        term_type="acronym",
        source_segment_id="seg_0",
        detector_candidate_id="unkterm_abc",
        confidence=0.86,
    )

    assert record.memory_type == MemoryType.MEETING_FACT.value
    assert record.scope == "session"
    assert record.session_id == "session_001"
    assert record.org_id == "org_001"
    assert record.user_id == "user_001"
    assert record.text == "商品交易总额，电商核心指标。"
    # Consolidation allowlist fields land at metadata top-level.
    assert record.metadata["canonical_entity"] == "GMV"
    assert record.metadata["entity"] == "GMV"
    assert record.metadata["topic"] == "GMV"
    assert record.metadata["normalized_entity"] == "gmv"
    assert record.metadata["source_capture_ref"] == "transcript:seg_0"
    # Non-allowlisted fields ride through under candidate_metadata.
    candidate_meta = record.metadata["candidate_metadata"]
    assert candidate_meta["term"] == "GMV"
    assert candidate_meta["term_type"] == "acronym"
    assert candidate_meta["memory_source"] == "unknown_term_explanation_v1"
    assert candidate_meta["detector_candidate_id"] == "unkterm_abc"
    assert "term_explanation" in record.tags
    assert "term_type:acronym" in record.tags


def test_commit_explanation_is_idempotent_on_session_and_normalized_term() -> None:
    service, writer = make_service()

    first = writer.commit_explanation(
        session_id="session_001",
        org_id="org_001",
        user_id="user_001",
        term="GMV",
        explanation="商品交易总额，电商核心指标。",
    )
    second = writer.commit_explanation(
        session_id="session_001",
        org_id="org_001",
        user_id="user_001",
        term="gmv",  # different casing should still hit the same record
        explanation="不同的解释也不会覆盖第一次的内容。",
    )

    assert first.memory_id == second.memory_id
    assert second.text == first.text  # original explanation preserved
    listed = service.store.list_memories(
        MemoryQuery(session_id="session_001", limit=10)
    )
    assert len(listed) == 1


def test_commit_explanation_different_terms_get_different_records() -> None:
    service, writer = make_service()

    writer.commit_explanation(
        session_id="session_001",
        org_id="org_001",
        user_id="user_001",
        term="GMV",
        explanation="商品交易总额。",
    )
    writer.commit_explanation(
        session_id="session_001",
        org_id="org_001",
        user_id="user_001",
        term="ARR",
        explanation="年度经常性收入。",
    )

    listed = service.store.list_memories(
        MemoryQuery(session_id="session_001", limit=10)
    )
    assert len(listed) == 2
    terms = sorted(record.metadata["canonical_entity"] for record in listed)
    assert terms == ["ARR", "GMV"]


def test_commit_explanation_different_sessions_get_different_records() -> None:
    service, writer = make_service()

    writer.commit_explanation(
        session_id="session_a",
        org_id="org_001",
        user_id="user_001",
        term="GMV",
        explanation="商品交易总额。",
    )
    writer.commit_explanation(
        session_id="session_b",
        org_id="org_001",
        user_id="user_001",
        term="GMV",
        explanation="商品交易总额。",
    )

    listed_a = service.store.list_memories(MemoryQuery(session_id="session_a", limit=10))
    listed_b = service.store.list_memories(MemoryQuery(session_id="session_b", limit=10))
    assert len(listed_a) == 1
    assert len(listed_b) == 1
    assert listed_a[0].memory_id != listed_b[0].memory_id


def test_commit_explanation_rejects_empty_term_or_explanation() -> None:
    _, writer = make_service()

    with pytest.raises(ValueError, match="term"):
        writer.commit_explanation(
            session_id="session_001",
            org_id="org_001",
            user_id="user_001",
            term="   ",
            explanation="something",
        )
    with pytest.raises(ValueError, match="explanation"):
        writer.commit_explanation(
            session_id="session_001",
            org_id="org_001",
            user_id="user_001",
            term="GMV",
            explanation="",
        )


def test_explanation_is_visible_to_personal_vocabulary_within_session() -> None:
    """The whole point of B2: writer output becomes part of session
    vocabulary, so the next detector cycle can drop the term from its
    candidate set."""

    service, writer = make_service()
    writer.commit_explanation(
        session_id="session_001",
        org_id="org_001",
        user_id="user_001",
        term="GMV",
        explanation="商品交易总额。",
    )

    vocab = PersonalVocabularyService(service)
    explained = vocab.get_explained_terms_for_session(session_id="session_001")

    assert "GMV" in explained
