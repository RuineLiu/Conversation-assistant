from __future__ import annotations

from datetime import UTC, datetime

from proactive_assistant.memory import InMemoryMemoryStore, MemoryQuery, MemoryRetrievalIntent, MemoryService
from proactive_assistant.prompting import PRDSurface, PromptCategory
from proactive_assistant.runtime import MemoryCandidate, MemoryCandidateType, MemoryWritePolicy


def candidate(
    candidate_type: MemoryCandidateType,
    text: str,
    *,
    memory_candidate_id: str = "memcand_consolidation",
    write_policy: MemoryWritePolicy = MemoryWritePolicy.ELIGIBLE,
    metadata: dict[str, object] | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_candidate_id=memory_candidate_id,
        decision_id="dec_001",
        session_id="session_001",
        source_event_ids=["fb_001"],
        candidate_type=candidate_type,
        text=text,
        confidence=0.86,
        write_policy=write_policy,
        reason="test consolidation",
        metadata=metadata or {},
    )


def test_action_item_candidate_writes_normalized_owner_deadline_entity_and_provenance() -> None:
    service = MemoryService(InMemoryMemoryStore())

    memory = service.commit_candidate(
        candidate(
            MemoryCandidateType.ACTION_ITEM,
            "Alex owns the launch risk follow-up by 2026-06-13.",
            metadata={
                "prompt_category": "summary_gap_check",
                "prd_surface": "glasses_popup",
                "source_refs": ["transcript:seg_12"],
            },
        ),
        org_id="org_001",
        user_id="user_001",
    )

    assert memory.metadata["owner"] == "Alex"
    assert memory.metadata["assignee"] == "Alex"
    assert memory.metadata["normalized_deadline"] == "2026-06-13"
    assert memory.metadata["normalized_entity"] == "the launch risk"
    assert memory.metadata["prompt_category"] == "summary_gap_check"
    assert memory.metadata["prd_surface"] == "glasses_popup"
    assert memory.metadata["provenance"] == ["transcript:seg_12", "decision:dec_001", "feedback:fb_001"]
    assert {"owner", "deadline", "entity", "summary_gap_check", "glasses_popup"}.issubset(set(memory.tags))


def test_consolidated_action_item_supports_exact_deadline_and_owner_lookup() -> None:
    service = MemoryService(InMemoryMemoryStore())
    memory = service.commit_candidate(
        candidate(
            MemoryCandidateType.ACTION_ITEM,
            "Alex owns the launch risk follow-up by 2026-06-13.",
        ),
        org_id="org_001",
        user_id="user_001",
    )

    deadline_context = service.search_context(
        MemoryQuery(
            query_text="之前这个 deadline 是什么时候？",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            active_entities=[{"canonical_name": "launch risk", "last_ts": 1000}],
        )
    )
    owner_context = service.search_context(
        MemoryQuery(
            query_text="之前这个是谁负责？",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            active_entities=[{"canonical_name": "launch risk", "last_ts": 1000}],
        )
    )

    assert deadline_context.memory_refs == [f"memory:{memory.memory_id}"]
    assert deadline_context.results[0].intent == MemoryRetrievalIntent.LOOKUP_DEADLINE.value
    assert owner_context.memory_refs == [f"memory:{memory.memory_id}"]
    assert owner_context.results[0].intent == MemoryRetrievalIntent.LOOKUP_OWNER.value


def test_explicit_structured_candidate_metadata_is_not_overwritten() -> None:
    service = MemoryService(InMemoryMemoryStore())

    memory = service.commit_candidate(
        candidate(
            MemoryCandidateType.ACTION_ITEM,
            "Alex owns the launch risk follow-up by 2026-06-13.",
            metadata={
                "owner": "Bao",
                "normalized_deadline": "2026-06-14",
                "normalized_entity": "launch risk exception",
                "status": "blocked",
                "provenance": ["manual:override"],
            },
        ),
        org_id="org_001",
        user_id="user_001",
    )

    assert memory.metadata["owner"] == "Bao"
    assert memory.metadata["assignee"] == "Bao"
    assert memory.metadata["normalized_deadline"] == "2026-06-14"
    assert memory.metadata["normalized_entity"] == "launch risk exception"
    assert memory.metadata["status"] == "blocked"
    assert memory.metadata["provenance"] == ["manual:override"]


def test_privacy_preference_keeps_policy_metadata_for_retrieval() -> None:
    service = MemoryService(InMemoryMemoryStore())
    memory = service.commit_candidate(
        candidate(
            MemoryCandidateType.PRIVACY_PREFERENCE,
            "Avoid showing customer pricing details on glasses popup.",
            metadata={
                "prompt_category": PromptCategory.SUMMARY_GAP_CHECK.value,
                "prd_surface": PRDSurface.GLASSES_POPUP.value,
            },
        )
    )

    assert memory.metadata["feedback_affinity"] == 0.9
    assert memory.metadata["prompt_category"] == "summary_gap_check"
    assert memory.metadata["prd_surface"] == "glasses_popup"
    assert memory.metadata["memory_schema_version"] == "memory_consolidation_v1"


def test_relative_deadline_is_normalized_against_reference_time() -> None:
    service = MemoryService(InMemoryMemoryStore())

    memory = service.commit_candidate(
        candidate(
            MemoryCandidateType.ACTION_ITEM,
            "张三负责客户报价确认，下周五截止。",
            metadata={"reference_time": datetime(2026, 6, 5, tzinfo=UTC)},
        ),
        org_id="org_001",
        user_id="user_001",
    )

    assert memory.metadata["owner"] == "张三"
    assert memory.metadata["normalized_deadline"] == "2026-06-12"
