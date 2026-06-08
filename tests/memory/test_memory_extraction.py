import pytest

from proactive_assistant.memory import MemoryExtractionRequest, MemoryExtractionService
from proactive_assistant.model_gateway import FakeModelClient, ModelOutputValidationError
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import TranscriptWindowItem
from proactive_assistant.runtime import MemoryCandidateType, MemoryWritePolicy


def test_memory_extraction_service_converts_structured_model_output_to_candidates() -> None:
    client = FakeModelClient(_valid_extraction_response())
    service = MemoryExtractionService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test", max_output_tokens=512),
    )

    result = service.extract_candidates(_request())

    assert len(result.candidates) == 2
    action = result.candidates[0]
    project = result.candidates[1]
    assert action.memory_candidate_id.startswith("memcand_llm_")
    assert action.decision_id == "llm_memory_extraction:session_001"
    assert action.candidate_type == MemoryCandidateType.ACTION_ITEM.value
    assert action.write_policy == MemoryWritePolicy.NEEDS_CONFIRMATION.value
    assert action.metadata["owner"] == "张三"
    assert action.metadata["deadline"] == "下周五"
    assert action.metadata["source_utterance_id"] == "seg_0"
    assert action.metadata["memory_extraction_source"] == "llm_memory_extraction_v1"
    assert "deadline" in action.metadata["tags"]
    assert project.candidate_type == MemoryCandidateType.PROJECT_CONTEXT.value
    assert project.metadata["promotion_target"] == "long_term"
    assert result.model_usage is not None
    assert result.model_usage.model == "gpt-test"
    assert client.requests[0].response_schema_name == "MemoryExtractionResult"
    assert client.requests[0].metadata["contract"] == "memory_extraction_v1"
    assert "JSON object" in client.requests[0].instructions
    assert "张三负责客户报价确认" in client.requests[0].input_text


def test_memory_extraction_service_normalizes_compatible_model_field_drift() -> None:
    client = FakeModelClient(
        {
            "candidates": [
                {
                    "type": "action_item",
                    "text": "张三负责客户报价确认，下周五截止。",
                    "write_policy": "eligible",
                    "privacy_level": "medium",
                    "source_ref": "transcript:seg_0",
                    "owner": "张三",
                    "deadline": "下周五",
                    "tags": "action_item",
                }
            ],
        }
    )
    service = MemoryExtractionService(model_client=client)

    result = service.extract_candidates(_request())

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.candidate_type == MemoryCandidateType.ACTION_ITEM.value
    assert candidate.confidence == 0.72
    assert candidate.write_policy == MemoryWritePolicy.NEEDS_CONFIRMATION.value
    assert candidate.metadata["source_refs"] == ["transcript:seg_0"]
    assert candidate.metadata["tags"] == ["action_item", "deadline", "llm_extracted", "owner"]


def test_memory_extraction_service_normalizes_null_optional_text_fields() -> None:
    client = FakeModelClient(
        {
            "candidates": [
                {
                    "type": "person_or_fact",
                    "text": "李四是法务接口。",
                    "write_policy": "eligible",
                    "privacy_level": "low",
                    "source_refs": ["transcript:seg_0"],
                    "entity": "李四",
                    "owner": None,
                    "deadline": None,
                    "status": None,
                    "topic": None,
                    "reason": None,
                }
            ],
        }
    )
    service = MemoryExtractionService(model_client=client)

    result = service.extract_candidates(_request())

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.candidate_type == MemoryCandidateType.PERSON_OR_FACT.value
    assert "deadline" not in candidate.metadata
    assert "owner" not in candidate.metadata
    assert candidate.reason == "llm memory extraction"


def test_memory_extraction_service_absorbs_project_person_role_extras() -> None:
    client = FakeModelClient(
        {
            "candidates": [
                {
                    "type": "person_or_fact",
                    "text": "李四是 Project Atlas 的法务接口。",
                    "write_policy": "eligible",
                    "privacy_level": "low",
                    "source_refs": ["transcript:seg_0"],
                    "project": "Project Atlas",
                    "person": "李四",
                    "role": "法务接口",
                }
            ],
        }
    )
    service = MemoryExtractionService(model_client=client)

    result = service.extract_candidates(_request())

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.candidate_type == MemoryCandidateType.PERSON_OR_FACT.value
    assert candidate.metadata["entity"] == "李四 (法务接口)"
    assert candidate.metadata["topic"] == "Project Atlas"
    assert "Project Atlas" in candidate.metadata["tags"]
    assert "法务接口" in candidate.metadata["tags"]


def test_memory_extraction_service_filters_blocked_model_candidates() -> None:
    payload = _valid_extraction_response(
        candidates=[
            {
                **_candidate_payload(),
                "confidence": 0.2,
                "write_policy": "eligible",
            },
            {
                **_candidate_payload(text="低置信度候选。"),
                "write_policy": "blocked",
            },
        ]
    )
    service = MemoryExtractionService(model_client=FakeModelClient(payload))

    result = service.extract_candidates(_request())

    assert result.candidates == []


def test_memory_extraction_service_rejects_invalid_model_output() -> None:
    service = MemoryExtractionService(model_client=FakeModelClient({"candidates": [{"text": "missing fields"}]}))

    with pytest.raises(ModelOutputValidationError):
        service.extract_candidates(_request())


def _request() -> MemoryExtractionRequest:
    return MemoryExtractionRequest(
        session_id="session_001",
        transcript_window=[
            TranscriptWindowItem(
                transcript_id="seg_0",
                speaker="Bao",
                text="张三负责客户报价确认，下周五截止。这个是 Project Atlas 上线前的阻塞项。",
                timestamp_ms=0,
            )
        ],
        session_context={
            "org_id": "org_001",
            "subject_user_id": "user_001",
            "metadata": {"participants": ["Bao", "张三"]},
        },
        privacy_constraints=["avoid unnecessary customer data"],
    )


def _valid_extraction_response(**overrides):  # type: ignore[no-untyped-def]
    payload = {
        "candidates": [
            _candidate_payload(),
            _candidate_payload(
                candidate_type="project_context",
                text="Project Atlas 上线前的关键阻塞项是客户报价确认。",
                confidence=0.79,
                write_policy="eligible",
                privacy_level="medium",
                privacy_risk=0.4,
                entity="Project Atlas",
                owner="",
                deadline="",
                status="open",
                topic="Project Atlas 上线",
                tags=["project_context"],
                promotion_candidate=True,
            ),
        ],
        "extraction_notes": "extracted action and project context",
        "safety_flags": [],
    }
    payload.update(overrides)
    return payload


def _candidate_payload(**overrides):  # type: ignore[no-untyped-def]
    payload = {
        "candidate_type": "action_item",
        "text": "张三负责客户报价确认，下周五截止。",
        "confidence": 0.84,
        "write_policy": "eligible",
        "privacy_level": "medium",
        "privacy_risk": 0.35,
        "source_refs": ["transcript:seg_0"],
        "reason": "明确出现负责人和截止时间。",
        "entity": "客户报价确认",
        "owner": "张三",
        "deadline": "下周五",
        "status": "open",
        "topic": "客户报价",
        "tags": ["action_item"],
        "promotion_candidate": False,
    }
    payload.update(overrides)
    return payload
