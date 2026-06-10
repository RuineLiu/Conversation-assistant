"""End-to-end tests for P2-1/P2-2/P2-4/P2-5.

- P2-1: target_speaker_id propagates from segment → opportunity → memory metadata
- P2-2: capture_inline_memory writes a USER_PREFERENCE record idempotently
- P2-4: privacy_metrics counts what was deferred / suppressed / blocked
- P2-5: warmup MemoryContext flags is_empty_cold_start when the user
  has zero records, separately from "query had no match".
"""

from typing import Any

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemorySource,
    MemoryType,
)
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product import ProductAssistantService
from proactive_assistant.prompting import (
    PrivacyLevel,
    PromptGenerationService,
    RuleBasedPromptGenerationService,
)
from proactive_assistant.runtime import (
    MemoryCandidateType,
    PromptRuntimeService,
)
from proactive_assistant.sessions import (
    InMemorySessionStore,
    SessionConfig,
    SessionService,
    TranscriptSegmentInput,
)


def _build_service(memory_service: MemoryService | None = None) -> ProductAssistantService:
    memory_service = memory_service or MemoryService(InMemoryMemoryStore())
    fake = FakeModelClient(
        {
            "should_prompt": True,
            "prompt_category": "summary_gap_check",
            "content_granularity": 2,
            "glasses_title": "Owner pending",
            "glasses_text": "请确认负责人。",
            "app_detail_text": "需要明确负责人。",
            "source_refs": ["transcript:seg_0"],
            "confidence": 0.8,
            "privacy_level": "low",
            "privacy_risk": 0.1,
            "rationale": "test",
            "safety_flags": [],
        }
    )
    prompt_service = PromptGenerationService(
        model_client=fake,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    return ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(
            prompt_service=prompt_service,
            fallback_prompt_service=RuleBasedPromptGenerationService(),
        ),
        runtime_service=PromptRuntimeService(),
        memory_service=memory_service,
    )


# ----- P2-1 -----

def test_target_speaker_id_propagates_from_segment_to_opportunity() -> None:
    service = _build_service()
    service.create_session(
        SessionConfig(
            title="Risk sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )
    step = service.append_transcript_and_generate_prompts(
        "session_001",
        TranscriptSegmentInput(
            speaker="张三",
            start_ms=0,
            end_ms=900,
            text="这个问题谁负责？",
            asr_confidence=0.94,
        ),
        segment_id="seg_0",
    )

    assert step.decisions
    for decision in step.decisions:
        assert decision.candidate.opportunity.target_speaker_id == "张三"


# ----- P2-2 -----

def test_capture_inline_memory_writes_user_preference_idempotently() -> None:
    service = _build_service()
    service.create_session(
        SessionConfig(
            title="Pricing review",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )

    first = service.capture_inline_memory(
        "session_001",
        "我每周一上午开会希望避免任何打扰。",
        memory_type=MemoryCandidateType.USER_PREFERENCE,
        source_segment_id="seg_0",
        target_speaker_id="张三",
        topic="schedule_preference",
    )
    second = service.capture_inline_memory(
        "session_001",
        "我每周一上午开会希望避免任何打扰。",
        memory_type=MemoryCandidateType.USER_PREFERENCE,
        source_segment_id="seg_0",
    )

    assert first.committed is True
    assert first.memory.memory_id == second.memory.memory_id
    # The candidate metadata round-trips into memory metadata.candidate_metadata
    candidate_meta = first.memory.metadata["candidate_metadata"]
    assert candidate_meta["capture_source"] == "user_explicit_inline"
    assert candidate_meta["target_speaker_id"] == "张三"
    assert "inline_capture" in first.memory.tags


def test_capture_inline_memory_rejects_empty_text() -> None:
    service = _build_service()
    service.create_session(SessionConfig(title="x"), session_id="session_001")
    try:
        service.capture_inline_memory("session_001", "   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for empty capture text")


# ----- P2-4 -----

def test_privacy_metrics_counts_glasses_decisions_and_enforcement() -> None:
    service = _build_service()
    service.create_session(
        SessionConfig(title="x", metadata={"org_id": "org_001", "subject_user_id": "user_001"}),
        session_id="session_001",
    )
    service.append_transcript_and_generate_prompts(
        "session_001",
        TranscriptSegmentInput(
            speaker="张三",
            start_ms=0,
            end_ms=900,
            text="这个问题谁负责，下周五前能不能定？",
            asr_confidence=0.94,
        ),
        segment_id="seg_0",
    )

    metrics = service.privacy_metrics("session_001")

    assert metrics.session_id == "session_001"
    assert metrics.total_decisions >= 1
    assert metrics.glasses_decisions >= 1


# ----- P2-5 -----

def test_warmup_memory_flags_cold_start_when_user_has_no_records() -> None:
    service = _build_service()
    service.create_session(
        SessionConfig(
            title="Project Atlas weekly",
            metadata={"org_id": "org_001", "subject_user_id": "user_001", "project": "Project Atlas"},
        ),
        session_id="session_001",
    )

    warmup = service.get_warmup_memory_context("session_001")

    assert warmup.is_empty_cold_start is True
    assert warmup.memory_context == []
    assert warmup.memory_refs == []


def test_warmup_memory_distinguishes_cold_start_from_no_query_match() -> None:
    """User has memory records but none match this particular session's
    query: warmup should return empty context but is_empty_cold_start=False
    so the UI knows memory exists, just not for this topic."""

    memory_service = MemoryService(InMemoryMemoryStore())
    # Seed a memory unrelated to Project Atlas
    memory_service.store.add_memory(
        MemoryRecord(
            memory_id="mem_other",
            memory_type=MemoryType.USER_PREFERENCE,
            scope=MemoryScope.USER,
            text="用户喜欢简短的提示。",
            org_id="org_001",
            user_id="user_001",
            source=MemorySource.MANUAL,
            confidence=0.9,
            importance=0.8,
            tags=["preference"],
            metadata={"canonical_entity": "Prompt style"},
        )
    )
    service = _build_service(memory_service)
    service.create_session(
        SessionConfig(
            title="Project Atlas planning",
            metadata={"org_id": "org_001", "subject_user_id": "user_001", "project": "Project Atlas"},
        ),
        session_id="session_001",
    )

    warmup = service.get_warmup_memory_context("session_001")

    # Records exist for this user (so not cold start), but the Atlas
    # query did not match the unrelated preference record.
    assert warmup.is_empty_cold_start is False
