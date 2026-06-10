"""B1 tests: cross-session memory warmup at create_session.

The warmup query is built from session metadata (title + project +
participants + tags) and retrieves USER / ORG / GLOBAL scope memories.
The result is cached on the product service so every subsequent
transcript step gets the warmup context prepended to its prompt input,
producing the "I remember from last week..." demo effect.
"""

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
    PromptGenerationService,
    RuleBasedPromptGenerationService,
)
from proactive_assistant.runtime import PromptRuntimeService
from proactive_assistant.sessions import (
    InMemorySessionStore,
    SessionConfig,
    SessionService,
    TranscriptSegmentInput,
)


def _build_service(memory_service: MemoryService) -> ProductAssistantService:
    client = FakeModelClient({"should_prompt": False, "content_granularity": 0, "confidence": 0.0, "privacy_level": "low", "privacy_risk": 0.0, "source_refs": []})
    prompt_service = PromptGenerationService(
        model_client=client,
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


def _seed_cross_session_memory(
    memory_service: MemoryService,
    *,
    memory_id: str,
    text: str,
    org_id: str = "org_001",
    user_id: str = "user_001",
    scope: MemoryScope = MemoryScope.USER,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> MemoryRecord:
    return memory_service.store.add_memory(
        MemoryRecord(
            memory_id=memory_id,
            memory_type=MemoryType.PROJECT_CONTEXT,
            scope=scope,
            text=text,
            org_id=org_id,
            user_id=user_id,
            source=MemorySource.MANUAL,
            confidence=0.9,
            importance=0.8,
            tags=tags or [],
            metadata=metadata or {},
        )
    )


def test_warmup_retrieves_user_scope_memory_matching_session_title() -> None:
    memory_service = MemoryService(InMemoryMemoryStore())
    _seed_cross_session_memory(
        memory_service,
        memory_id="mem_atlas",
        text="Project Atlas 上周决定推迟到 Q4 发布。",
        tags=["project_context", "atlas"],
        metadata={"canonical_entity": "Project Atlas", "entity": "Project Atlas"},
    )
    service = _build_service(memory_service)

    session = service.create_session(
        SessionConfig(
            title="Project Atlas weekly sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001", "project": "Project Atlas"},
        ),
        session_id="session_001",
    )

    warmup = service.get_warmup_memory_context(session.session_id)

    assert warmup.memory_refs == [f"memory:{warmup.results[0].memory.memory_id}"]
    assert warmup.results[0].memory.memory_id == "mem_atlas"
    assert any("Project Atlas" in line for line in warmup.memory_context)


def test_warmup_excludes_other_sessions_session_scope_memories() -> None:
    memory_service = MemoryService(InMemoryMemoryStore())
    # SESSION-scope memory from a DIFFERENT session — must not warm up here.
    memory_service.store.add_memory(
        MemoryRecord(
            memory_id="mem_other_session",
            memory_type=MemoryType.MEETING_FACT,
            scope=MemoryScope.SESSION,
            text="Project Atlas 上次会议拍板的话题。",
            org_id="org_001",
            user_id="user_001",
            session_id="session_other",
            source=MemorySource.TRANSCRIPT,
            confidence=0.85,
            importance=0.6,
            tags=["meeting_state"],
            metadata={"canonical_entity": "Project Atlas"},
        )
    )
    # USER-scope memory — should appear in warmup.
    _seed_cross_session_memory(
        memory_service,
        memory_id="mem_user_scope",
        text="Project Atlas 的发布节奏长期由 Bao 负责。",
        metadata={"canonical_entity": "Project Atlas"},
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
    ids = [result.memory.memory_id for result in warmup.results]

    assert "mem_user_scope" in ids
    assert "mem_other_session" not in ids


def test_warmup_returns_empty_when_metadata_yields_no_query() -> None:
    memory_service = MemoryService(InMemoryMemoryStore())
    _seed_cross_session_memory(
        memory_service,
        memory_id="mem_atlas",
        text="Some long-term context.",
    )
    service = _build_service(memory_service)

    # No title, no metadata — warmup query is empty, should return empty.
    service.create_session(SessionConfig(), session_id="session_001")
    warmup = service.get_warmup_memory_context("session_001")

    assert warmup.memory_context == []
    assert warmup.memory_refs == []
    assert warmup.results == []


def test_warmup_context_is_visible_in_subsequent_transcript_step() -> None:
    memory_service = MemoryService(InMemoryMemoryStore())
    _seed_cross_session_memory(
        memory_service,
        memory_id="mem_atlas",
        text="Project Atlas 上周决定推迟到 Q4 发布。",
        metadata={"canonical_entity": "Project Atlas"},
    )
    service = _build_service(memory_service)
    session = service.create_session(
        SessionConfig(
            title="Project Atlas weekly sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001", "project": "Project Atlas"},
        ),
        session_id="session_001",
    )

    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=900,
            text="今天先快速过一下进展。",
            asr_confidence=0.94,
        ),
        segment_id="seg_0",
    )

    # The snapshot the orchestrator saw must include warmup memory refs.
    assert "memory:mem_atlas" in step.snapshot.memory_refs


def test_warmup_disabled_when_flag_is_false() -> None:
    memory_service = MemoryService(InMemoryMemoryStore())
    _seed_cross_session_memory(
        memory_service,
        memory_id="mem_atlas",
        text="Some cross-session context.",
        metadata={"canonical_entity": "Project Atlas"},
    )
    service = _build_service(memory_service)

    service.create_session(
        SessionConfig(
            title="Project Atlas weekly sync",
            metadata={"org_id": "org_001", "subject_user_id": "user_001", "project": "Project Atlas"},
        ),
        session_id="session_001",
        warmup_memory=False,
    )

    warmup = service.get_warmup_memory_context("session_001")
    assert warmup.memory_context == []
    assert warmup.memory_refs == []
