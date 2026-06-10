from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalIntent,
    MemoryScope,
    MemorySource,
    MemoryType,
)
from proactive_assistant.memory.query_understanding import QueryUnderstandingRequest, QueryUnderstandingService
from proactive_assistant.memory.retrieval import MemoryRetriever, build_structured_query
from proactive_assistant.model_gateway import FakeModelClient, ModelGatewayError, ModelRequest
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import PromptCategory


def test_query_understanding_result_drives_retriever_intent_and_target_entity() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(
        _memory(
            "mem_customer_quote_owner",
            "客户报价确认由张三负责。",
            memory_type=MemoryType.ACTION_ITEM,
            metadata={"normalized_entity": "客户报价", "owner": "张三", "status": "open"},
            tags=["客户", "报价", "负责人"],
        )
    )
    retriever = MemoryRetriever(
        store,
        query_understanding=_service(
            {
                "intent": "lookup_owner",
                "target_entity": "客户报价",
                "target_entity_confidence": 0.92,
                "anaphora_resolved": False,
                "time_window_start": "",
                "time_window_end": "",
                "time_is_relative": False,
                "referenced_speaker": "",
                "confidence": 0.91,
                "rationale": "用户询问客户报价负责人。",
            }
        ),
    )

    context = retriever.retrieve(
        MemoryQuery(
            query_text="客户报价谁跟？",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            prompt_category=PromptCategory.PERSON_OR_FACT,
            reference_time=datetime(2026, 6, 10, tzinfo=UTC),
        )
    )

    assert context.memory_refs == ["memory:mem_customer_quote_owner"]
    assert context.results[0].intent == MemoryRetrievalIntent.LOOKUP_OWNER.value
    assert context.results[0].target_entity == "客户报价"
    assert context.results[0].reason == "exact_lookup_owner"


def test_retriever_sends_query_text_and_recent_transcript_as_separate_fields() -> None:
    def respond(request: ModelRequest) -> dict[str, Any]:
        payload = json.loads(request.input_text)
        assert payload["query_text"] == "2026年6月客户报价 deadline 哪天？"
        assert "张三上次提到客户报价还没有确认" in payload["recent_transcript_text"]
        return {
            "intent": "lookup_deadline",
            "target_entity": "客户报价",
            "target_entity_confidence": 0.95,
            "anaphora_resolved": False,
            "time_window_start": "2026-06-01",
            "time_window_end": "2026-06-30",
            "time_is_relative": False,
            "referenced_speaker": "",
            "confidence": 0.96,
            "rationale": "用户询问 6 月客户报价 deadline。",
        }

    store = InMemoryMemoryStore()
    store.add_memory(
        _memory(
            "mem_customer_quote_deadline",
            "张三负责客户报价确认，deadline 是 2026-06-20。",
            memory_type=MemoryType.ACTION_ITEM,
            metadata={
                "normalized_entity": "客户报价",
                "normalized_deadline": "2026-06-20",
                "normalized_start_time": "2026-06-01",
                "normalized_end_time": "2026-06-30",
            },
            tags=["客户", "报价", "deadline"],
        )
    )
    client = FakeModelClient(respond)

    context = MemoryRetriever(
        store,
        query_understanding=QueryUnderstandingService(
            model_client=client,
            settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
        ),
    ).retrieve(
        MemoryQuery(
            query_text="2026年6月客户报价 deadline 哪天？",
            recent_transcript_text="Bao: 我们继续 Project Atlas 风险同步。Mia: 张三上次提到客户报价还没有确认。",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            prompt_category=PromptCategory.QUESTION_ANSWER,
            reference_time=datetime(2026, 6, 10, tzinfo=UTC),
        )
    )

    assert context.memory_refs == ["memory:mem_customer_quote_deadline"]
    assert len(client.requests) == 1


def test_structured_query_infers_month_window_when_llm_omits_it() -> None:
    understood = _query_understanding_result(
        intent=MemoryRetrievalIntent.LOOKUP_DEADLINE,
        target_entity="2026年6月客户报价",
        time_window_start=None,
        time_window_end=None,
    )

    structured = build_structured_query(
        MemoryQuery(
            query_text="2026年6月客户报价 deadline 哪天？",
            reference_time=datetime(2026, 6, 10, tzinfo=UTC),
        ),
        understood=understood,
    )

    assert structured.intent == MemoryRetrievalIntent.LOOKUP_DEADLINE
    assert structured.target_entity == "客户报价"
    assert structured.time_window_start == "2026-06-01"
    assert structured.time_window_end == "2026-06-30"


def test_query_understanding_low_confidence_returns_none() -> None:
    service = _service(
        {
            "intent": "lookup_owner",
            "target_entity": "客户报价",
            "target_entity_confidence": 0.9,
            "anaphora_resolved": False,
            "time_window_start": "",
            "time_window_end": "",
            "time_is_relative": False,
            "referenced_speaker": "",
            "confidence": 0.42,
            "rationale": "不确定。",
        }
    )

    result = service.understand(QueryUnderstandingRequest(query_text="客户报价谁跟？"))

    assert result is None


def test_query_understanding_model_gateway_error_returns_none() -> None:
    def fail(_request: ModelRequest) -> dict[str, Any]:
        raise ModelGatewayError("temporary outage")

    service = QueryUnderstandingService(
        model_client=FakeModelClient(fail),
        settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
    )

    result = service.understand(QueryUnderstandingRequest(query_text="客户报价谁跟？"))

    assert result is None


def test_query_understanding_cache_returns_cached_result_without_second_model_call() -> None:
    client = FakeModelClient(
        {
            "intent": "lookup_deadline",
            "target_entity": "客户报价",
            "target_entity_confidence": 0.9,
            "anaphora_resolved": False,
            "time_window_start": "",
            "time_window_end": "",
            "time_is_relative": False,
            "referenced_speaker": "",
            "confidence": 0.88,
            "rationale": "用户询问截止时间。",
        }
    )
    service = QueryUnderstandingService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
    )
    request = QueryUnderstandingRequest(query_text="客户报价 deadline 哪天？", reference_date_iso="2026-06-10")

    first = service.understand(request)
    second = service.understand(request)

    assert first is not None
    assert first.cached is False
    assert second is not None
    assert second.cached is True
    assert len(client.requests) == 1


def test_query_understanding_cache_key_includes_active_entities() -> None:
    def respond(request: ModelRequest) -> dict[str, Any]:
        payload = json.loads(request.input_text)
        entity = payload["active_entities"][0]
        return {
            "intent": "lookup_owner",
            "target_entity": entity,
            "target_entity_confidence": 0.9,
            "anaphora_resolved": True,
            "time_window_start": "",
            "time_window_end": "",
            "time_is_relative": False,
            "referenced_speaker": "",
            "confidence": 0.88,
            "rationale": "根据当前实体消解指代。",
        }

    client = FakeModelClient(respond)
    service = QueryUnderstandingService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
    )

    first = service.understand(
        QueryUnderstandingRequest(query_text="这个谁负责？", active_entities=["客户报价"], reference_date_iso="2026-06-10")
    )
    second = service.understand(
        QueryUnderstandingRequest(query_text="这个谁负责？", active_entities=["发布计划"], reference_date_iso="2026-06-10")
    )

    assert first is not None
    assert first.target_entity == "客户报价"
    assert second is not None
    assert second.target_entity == "发布计划"
    assert second.cached is False
    assert len(client.requests) == 2


def test_lookup_task_list_intent_matches_open_action_items() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(
        _memory(
            "mem_open_task",
            "张三需要推进客户报价确认。",
            memory_type=MemoryType.ACTION_ITEM,
            metadata={"entity": "客户报价", "status": "open"},
            tags=["客户", "报价", "待办"],
        )
    )
    store.add_memory(
        _memory(
            "mem_done_task",
            "李四已经完成会后纪要。",
            memory_type=MemoryType.ACTION_ITEM,
            metadata={"entity": "会后纪要", "status": "done"},
            tags=["纪要"],
        )
    )

    context = MemoryRetriever(
        store,
        query_understanding=_service(
            {
                "intent": "lookup_task_list",
                "target_entity": "",
                "target_entity_confidence": 0.0,
                "anaphora_resolved": False,
                "time_window_start": "",
                "time_window_end": "",
                "time_is_relative": False,
                "referenced_speaker": "",
                "confidence": 0.9,
                "rationale": "用户询问待办列表。",
            }
        ),
    ).retrieve(
        MemoryQuery(
            query_text="我还有哪些待办？",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            prompt_category=PromptCategory.QUESTION_ANSWER,
            limit=2,
        )
    )

    assert context.memory_refs[0] == "memory:mem_open_task"
    assert context.results[0].intent == MemoryRetrievalIntent.LOOKUP_TASK_LIST.value
    assert context.results[0].rank_features["exact_feature"] == 1.0
    assert context.results[1].memory.memory_id == "mem_done_task"
    assert context.results[1].rank_features["exact_feature"] == 0.3


def test_lookup_schedule_intent_matches_memories_with_event_window() -> None:
    store = InMemoryMemoryStore()
    store.add_memory(
        _memory(
            "mem_project_review",
            "Project Atlas 评审安排在 2026-06-15。",
            memory_type=MemoryType.MEETING_FACT,
            metadata={"normalized_start_time": "2026-06-15", "normalized_end_time": "2026-06-15", "entity": "Project Atlas"},
            tags=["Project Atlas", "评审"],
        )
    )

    context = MemoryRetriever(
        store,
        query_understanding=_service(
            {
                "intent": "lookup_schedule",
                "target_entity": "Project Atlas",
                "target_entity_confidence": 0.88,
                "anaphora_resolved": False,
                "time_window_start": "2026-06-15",
                "time_window_end": "2026-06-15",
                "time_is_relative": False,
                "referenced_speaker": "",
                "confidence": 0.9,
                "rationale": "用户询问日程。",
            }
        ),
    ).retrieve(
        MemoryQuery(
            query_text="Project Atlas 评审是哪天？",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            prompt_category=PromptCategory.QUESTION_ANSWER,
            limit=2,
        )
    )

    assert context.memory_refs == ["memory:mem_project_review"]
    assert context.results[0].intent == MemoryRetrievalIntent.LOOKUP_SCHEDULE.value
    assert context.results[0].rank_features["exact_feature"] == 1.0


def _service(response: dict[str, Any]) -> QueryUnderstandingService:
    return QueryUnderstandingService(
        model_client=FakeModelClient(response),
        settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
    )


def _query_understanding_result(
    *,
    intent: MemoryRetrievalIntent,
    target_entity: str | None,
    time_window_start: str | None,
    time_window_end: str | None,
) -> Any:
    from proactive_assistant.memory.query_understanding import QueryUnderstandingResult

    return QueryUnderstandingResult(
        intent=intent,
        target_entity=target_entity,
        target_entity_confidence=0.9,
        anaphora_resolved=False,
        time_window_start=time_window_start,
        time_window_end=time_window_end,
        time_is_relative=False,
        referenced_speaker=None,
        confidence=0.9,
        rationale="test",
    )


def _memory(
    memory_id: str,
    text: str,
    *,
    memory_type: MemoryType,
    metadata: dict[str, object] | None = None,
    tags: list[str] | None = None,
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
        source=MemorySource.MEETING_STATE,
        source_ids=[f"meeting:{memory_id}"],
        confidence=0.9,
        importance=0.85,
        tags=tags or [],
        created_at=created,
        updated_at=created,
        metadata=metadata or {},
    )


def test_invalid_llm_dates_are_dropped_instead_of_emptying_candidate_pool() -> None:
    """P1-1: LLM may disobey the ISO-date instruction (e.g. "下周").
    Non-ISO values must be dropped at the service boundary; otherwise the
    store-level time filter would treat the unparseable window as
    matching nothing and silently empty the result set."""

    store = InMemoryMemoryStore()
    store.add_memory(
        _memory(
            "mem_quote_owner",
            "客户报价确认由张三负责。",
            memory_type=MemoryType.ACTION_ITEM,
            metadata={"normalized_entity": "客户报价", "owner": "张三", "status": "open"},
            tags=["客户", "报价"],
        )
    )

    context = MemoryRetriever(
        store,
        query_understanding=_service(
            {
                "intent": "lookup_owner",
                "target_entity": "客户报价",
                "target_entity_confidence": 0.9,
                "anaphora_resolved": False,
                "time_window_start": "下周",  # invalid: not ISO
                "time_window_end": "尽快",    # invalid: not ISO
                "time_is_relative": True,
                "referenced_speaker": "",
                "confidence": 0.9,
                "rationale": "test",
            }
        ),
    ).retrieve(
        MemoryQuery(
            query_text="客户报价谁负责？",
            org_id="org_001",
            user_id="user_001",
            session_id="session_001",
            prompt_category=PromptCategory.PERSON_OR_FACT,
            reference_time=datetime(2026, 6, 10, tzinfo=UTC),
        )
    )

    # Invalid dates dropped -> no time filter applied -> memory still found.
    assert context.memory_refs == ["memory:mem_quote_owner"]


def test_query_understanding_skipped_for_non_question_categories() -> None:
    """P1-2 gate: summary/suggestion/concept categories and category-less
    base retrieval do not consume an LLM call."""

    store = InMemoryMemoryStore()
    client = FakeModelClient(
        {
            "intent": "open_recall",
            "target_entity": "",
            "target_entity_confidence": 0.0,
            "anaphora_resolved": False,
            "time_window_start": "",
            "time_window_end": "",
            "time_is_relative": False,
            "referenced_speaker": "",
            "confidence": 0.9,
            "rationale": "test",
        }
    )
    retriever = MemoryRetriever(
        store,
        query_understanding=QueryUnderstandingService(
            model_client=client,
            settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
        ),
    )

    # Category-less base retrieval: no LLM call.
    retriever.retrieve(MemoryQuery(query_text="我们继续推进吧", session_id="session_001"))
    assert client.requests == []

    # Non-question category: no LLM call.
    retriever.retrieve(
        MemoryQuery(
            query_text="会后总结一下",
            session_id="session_001",
            prompt_category=PromptCategory.SUMMARY_GAP_CHECK,
        )
    )
    assert client.requests == []

    # Question category: LLM call issued.
    retriever.retrieve(
        MemoryQuery(
            query_text="上次的 deadline 是哪天？",
            session_id="session_001",
            prompt_category=PromptCategory.QUESTION_ANSWER,
        )
    )
    assert len(client.requests) == 1
