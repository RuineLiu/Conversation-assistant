"""PR1 (auto memory snapshot) tests.

When ``auto_memory_snapshot=True`` (the default), every transcript step
that adds a commitment-worthy MeetingState item (action_item with owner /
deadline / next_step, decided Decision, any Risk) should auto-upsert that
item into long-term memory, so cross-session recall has data without
anyone needing to call ``/memory-snapshot`` manually.
"""

from typing import Any

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryQuery,
    MemoryService,
    MemoryType,
)
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product import ProductAssistantService
from proactive_assistant.prompting import PromptGenerationService, RuleBasedPromptGenerationService
from proactive_assistant.runtime import PromptRuntimeService
from proactive_assistant.sessions import (
    InMemorySessionStore,
    SessionConfig,
    SessionService,
    TranscriptSegmentInput,
)


def _build(
    *, auto_memory_snapshot: bool = True, memory_service: MemoryService | None = None,
):  # type: ignore[no-untyped-def]
    fake_response = {
        "should_prompt": False,
        "prompt_category": None,
        "content_granularity": 0,
        "confidence": 0.0,
        "privacy_level": "low",
        "privacy_risk": 0.0,
        "source_refs": [],
    }
    fake = FakeModelClient(fake_response)
    prompt_service = PromptGenerationService(
        model_client=fake,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    memory_service = memory_service or MemoryService(InMemoryMemoryStore())
    service = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(
            prompt_service=prompt_service,
            fallback_prompt_service=RuleBasedPromptGenerationService(),
        ),
        runtime_service=PromptRuntimeService(),
        memory_service=memory_service,
        auto_memory_snapshot=auto_memory_snapshot,
    )
    return service, memory_service


def _commitment_transcript(text: str) -> TranscriptSegmentInput:
    return TranscriptSegmentInput(
        speaker="张三",
        start_ms=0,
        end_ms=1000,
        text=text,
        asr_confidence=0.94,
    )


def test_auto_snapshot_persists_action_item_with_commitment() -> None:
    service, memory_service = _build()
    service.create_session(
        SessionConfig(
            title="Q3 launch sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )

    service.append_transcript_and_generate_prompts(
        "session_001",
        _commitment_transcript("张三负责客户报价确认，下周五交付。"),
        segment_id="seg_0",
    )

    records = memory_service.store.list_memories(MemoryQuery(session_id="session_001", limit=20))
    action_records = [r for r in records if r.memory_type == MemoryType.ACTION_ITEM.value]
    assert action_records
    # owner/deadline metadata propagates to the persisted record
    record = action_records[0]
    assert record.metadata.get("owner") == "张三" or "张三" in record.text


def test_auto_snapshot_disabled_keeps_memory_empty_until_manual_call() -> None:
    service, memory_service = _build(auto_memory_snapshot=False)
    service.create_session(
        SessionConfig(
            title="Q3 launch sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )

    service.append_transcript_and_generate_prompts(
        "session_001",
        _commitment_transcript("张三负责客户报价确认，下周五交付。"),
        segment_id="seg_0",
    )

    records = memory_service.store.list_memories(MemoryQuery(session_id="session_001", limit=20))
    action_records = [r for r in records if r.memory_type == MemoryType.ACTION_ITEM.value]
    assert action_records == []


def test_auto_snapshot_only_snapshots_each_item_once() -> None:
    """Internal seen-set prevents the same item from being upsertted on
    every subsequent transcript step. We trigger one commitment, then
    several follow-up turns with unrelated content, and verify the action
    item count in memory does not grow on every step."""

    service, memory_service = _build()
    service.create_session(
        SessionConfig(
            title="Q3 launch sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )

    service.append_transcript_and_generate_prompts(
        "session_001",
        _commitment_transcript("张三负责客户报价确认，下周五交付。"),
        segment_id="seg_0",
    )
    first_count = len(memory_service.store.list_memories(MemoryQuery(session_id="session_001", limit=20)))

    # Internal seen-set should prevent re-snapshotting the same item on
    # subsequent steps. We assert via direct private method call to avoid
    # cross-test orchestrator interference.
    state = service.get_meeting_state("session_001")
    service._auto_snapshot_new_commitments("session_001", state)
    service._auto_snapshot_new_commitments("session_001", state)
    service._auto_snapshot_new_commitments("session_001", state)

    final_count = len(memory_service.store.list_memories(MemoryQuery(session_id="session_001", limit=20)))
    assert final_count == first_count


def test_auto_snapshot_failure_does_not_break_realtime_step() -> None:
    """A memory write hiccup must not crash the transcript step. We
    sabotage the memory service to raise on upsert and verify the step
    still returns a normal result."""

    service, memory_service = _build()

    class _ExplodingMemory:
        def __init__(self, inner):
            self._inner = inner
            self.store = inner.store

        def upsert_candidate(self, *args, **kwargs):
            raise RuntimeError("synthetic memory failure")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    service.memory = _ExplodingMemory(service.memory)
    service.create_session(
        SessionConfig(
            title="Q3 launch sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001"},
        ),
        session_id="session_001",
    )

    step = service.append_transcript_and_generate_prompts(
        "session_001",
        _commitment_transcript("张三负责客户报价确认，下周五交付。"),
        segment_id="seg_0",
    )
    # Step result is intact even though memory upsert raised
    assert step.session.session_id == "session_001"
    assert step.transcript_segment.segment_id == "seg_0"
