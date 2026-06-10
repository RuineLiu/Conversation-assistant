from __future__ import annotations

from datetime import UTC, datetime

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemorySource,
    MemoryType,
)
from proactive_assistant.memory.query_understanding import QueryUnderstandingService
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product import ProductAssistantService
from proactive_assistant.prompting import RuleBasedPromptGenerationService
from proactive_assistant.runtime import PromptRuntimeService
from proactive_assistant.sessions import (
    InMemorySessionStore,
    SessionConfig,
    SessionService,
    TranscriptSegmentInput,
)


def test_product_flow_uses_query_understanding_for_cross_session_memory_retrieval() -> None:
    query_client = FakeModelClient(
        {
            "intent": "lookup_deadline",
            "target_entity": "客户报价",
            "target_entity_confidence": 0.94,
            "anaphora_resolved": False,
            "time_window_start": "2026-06-01",
            "time_window_end": "2026-06-30",
            "time_is_relative": False,
            "referenced_speaker": "张三",
            "confidence": 0.91,
            "rationale": "用户询问客户报价行动项的截止时间。",
        }
    )
    memory_service = MemoryService(
        InMemoryMemoryStore(),
        query_understanding=QueryUnderstandingService(
            model_client=query_client,
            settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
        ),
    )
    memory_service.store.add_memory(
        _action_memory(
            "mem_customer_quote_deadline",
            "上次会议张三负责客户报价确认，deadline 是 2026-06-20。",
            metadata={
                "normalized_entity": "客户报价",
                "owner": "张三",
                "normalized_deadline": "2026-06-20",
                "normalized_start_time": "2026-06-01",
                "normalized_end_time": "2026-06-30",
                "status": "open",
            },
        )
    )
    memory_service.store.add_memory(
        _action_memory(
            "mem_july_quote_deadline",
            "客户报价复盘安排在 2026-07-08。",
            metadata={
                "normalized_entity": "客户报价",
                "normalized_deadline": "2026-07-08",
                "normalized_start_time": "2026-07-01",
                "normalized_end_time": "2026-07-31",
                "status": "open",
            },
        )
    )
    service = _build_product_service(memory_service)
    session = service.create_session(
        SessionConfig(metadata={"org_id": "org_001", "subject_user_id": "user_001"}),
        session_id="session_current",
        warmup_memory=False,
    )

    # The transcript step itself runs category-less base retrieval, which
    # the P1-2 cost gate intentionally keeps on the rule path (no LLM).
    step = service.append_transcript_and_generate_prompts(
        session.session_id,
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=1200,
            text="上次张三说的客户报价 deadline 哪天？",
            asr_confidence=0.96,
        ),
        segment_id="seg_query",
        use_memory=True,
        memory_limit=5,
    )
    assert step.transcript_segment.segment_id == "seg_query"
    base_retrieval_calls = len(query_client.requests)

    # Question-shaped retrieval (the per-opportunity path and direct API
    # queries carry prompt_category) goes through LLM understanding and
    # hits the cross-session deadline memory with time-window filtering.
    context = service.search_memory_context_for_session(
        session.session_id,
        query_text="上次张三说的客户报价 deadline 哪天？",
        limit=5,
        prompt_category="question_answer",
    )

    assert context.memory_refs == ["memory:mem_customer_quote_deadline"]
    assert context.results[0].target_entity == "客户报价"
    assert len(query_client.requests) > base_retrieval_calls


def _build_product_service(memory_service: MemoryService) -> ProductAssistantService:
    return ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=PromptOrchestrator(
            prompt_service=RuleBasedPromptGenerationService(),
        ),
        runtime_service=PromptRuntimeService(),
        memory_service=memory_service,
    )


def _action_memory(memory_id: str, text: str, *, metadata: dict[str, object]) -> MemoryRecord:
    created = datetime(2026, 6, 1, tzinfo=UTC)
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=MemoryType.ACTION_ITEM,
        scope=MemoryScope.USER,
        text=text,
        org_id="org_001",
        user_id="user_001",
        source=MemorySource.MEETING_STATE,
        source_ids=[f"meeting:{memory_id}"],
        confidence=0.9,
        importance=0.85,
        tags=["客户", "报价", "deadline"],
        created_at=created,
        updated_at=created,
        metadata=metadata,
    )
