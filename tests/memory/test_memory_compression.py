import pytest

from proactive_assistant.memory import MemoryCompressionRequest, MemoryCompressionService
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import TranscriptWindowItem
from proactive_assistant.runtime import MemoryCandidateType, MemoryWritePolicy


def test_memory_compression_service_returns_summary_quality_and_candidates() -> None:
    client = FakeModelClient(_valid_compression_response())
    service = MemoryCompressionService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test", max_output_tokens=800),
    )

    result = service.compress_chunk(_request())

    assert result.session_id == "session_001"
    assert result.chunk_id == "chunk_001"
    assert "客户报价" in result.chunk_summary
    assert result.compression_quality == 0.82
    assert result.coverage_score == 0.78
    assert result.loss_risk_score == 0.18
    assert result.source_refs == ["transcript:seg_0", "transcript:seg_1"]
    assert len(result.candidate_memories) == 2
    action = result.candidate_memories[0]
    assert action.memory_candidate_id.startswith("memcand_llm_")
    assert action.decision_id == "llm_memory_compression:session_001:chunk_001"
    assert action.candidate_type == MemoryCandidateType.ACTION_ITEM.value
    assert action.write_policy == MemoryWritePolicy.NEEDS_CONFIRMATION.value
    assert action.metadata["source_refs"] == ["transcript:seg_0"]
    assert action.metadata["memory_extraction_source"] == "llm_memory_extraction_v1"
    assert result.model_usage is not None
    assert result.model_usage.model == "gpt-test"
    assert client.requests[0].response_schema_name == "MemoryCompressionResult"
    assert client.requests[0].metadata["contract"] == "memory_compression_v1"
    assert client.requests[0].metadata["chunk_id"] == "chunk_001"
    assert "compress meeting transcript chunks" in client.requests[0].instructions
    assert "张三负责客户报价确认" in client.requests[0].input_text


def test_memory_compression_service_normalizes_compatible_field_drift() -> None:
    client = FakeModelClient(
        {
            "summary": "本段确认 Project Atlas 的法务接口和手动入口决策。",
            "key_points": "李四是法务接口",
            "open_questions": None,
            "candidates": [
                {
                    "type": "person_or_fact",
                    "text": "李四是 Project Atlas 的法务接口。",
                    "write_policy": "eligible",
                    "privacy_level": "low",
                    "source_ref": "transcript:seg_1",
                    "project": "Project Atlas",
                    "person": "李四",
                    "role": "法务接口",
                }
            ],
            "source_ref": "transcript:seg_1",
        }
    )
    service = MemoryCompressionService(model_client=client)

    result = service.compress_chunk(_request())

    assert result.key_points == ["李四是法务接口"]
    assert result.open_questions == []
    assert result.source_refs == ["transcript:seg_1"]
    assert result.compression_quality >= 0.9
    candidate = result.candidate_memories[0]
    assert candidate.candidate_type == MemoryCandidateType.PERSON_OR_FACT.value
    assert candidate.metadata["entity"] == "李四 (法务接口)"
    assert candidate.metadata["topic"] == "Project Atlas"


def test_memory_compression_normalizes_dict_points_and_refs() -> None:
    service = MemoryCompressionService(
        model_client=FakeModelClient(
            _valid_compression_response(
                chunk_summary={"text": "本段确认报价负责人与法务接口。"},
                key_points=[
                    {"point": "张三负责客户报价确认。", "source_refs": ["transcript:seg_0"]},
                ],
                open_questions=[
                    {"question": "法务审批截止时间未确认。", "source_refs": ["transcript:seg_1"]},
                ],
                source_refs=[],
            )
        )
    )

    result = service.compress_chunk(_request())

    assert result.chunk_summary == "本段确认报价负责人与法务接口。"
    assert result.key_points == ["张三负责客户报价确认。"]
    assert result.open_questions == ["法务审批截止时间未确认。"]
    assert result.source_refs == ["transcript:seg_0", "transcript:seg_1"]


def test_memory_compression_accepts_candidate_text_aliases() -> None:
    service = MemoryCompressionService(
        model_client=FakeModelClient(
            _valid_compression_response(
                candidate_memories=[
                    {
                        "type": "decision",
                        "content": "MVP 版本先不做自动推送，只保留手动查看入口。",
                        "write_policy": "eligible",
                        "privacy_level": "low",
                        "source_refs": ["transcript:seg_1"],
                        "status": "decided",
                    }
                ],
            )
        )
    )

    result = service.compress_chunk(_request())

    assert result.candidate_memories[0].text == "MVP 版本先不做自动推送，只保留手动查看入口。"
    assert result.candidate_memories[0].candidate_type == MemoryCandidateType.DECISION.value


def test_memory_compression_accepts_quality_score_objects() -> None:
    service = MemoryCompressionService(
        model_client=FakeModelClient(
            _valid_compression_response(
                compression_quality={"score": 0.81},
                coverage_score={"value": 0.77},
                loss_risk_score={"rating": 0.19},
            )
        )
    )

    result = service.compress_chunk(_request())

    assert result.compression_quality == 0.81
    assert result.coverage_score == 0.77
    assert result.loss_risk_score == 0.19


def test_memory_compression_builds_candidate_text_from_structured_fields() -> None:
    service = MemoryCompressionService(
        model_client=FakeModelClient(
            _valid_compression_response(
                candidate_memories=[
                    {
                        "candidate_type": "action_item",
                        "confidence": 0.82,
                        "write_policy": "eligible",
                        "privacy_level": "medium",
                        "source_ref": "transcript:seg_0",
                        "topic": "客户报价确认",
                        "owner": "张三",
                        "deadline": "下周五",
                        "status": "open",
                    }
                ],
            )
        )
    )

    result = service.compress_chunk(_request())

    assert result.candidate_memories[0].text == "客户报价确认；负责人：张三；截止：下周五；状态：open。"


def test_memory_compression_drops_empty_model_candidates() -> None:
    service = MemoryCompressionService(
        model_client=FakeModelClient(
            _valid_compression_response(
                candidate_memories=[
                    {"candidate_type": "summary", "source_refs": ["transcript:seg_0"]},
                    _candidate_payload(),
                ],
            )
        )
    )

    result = service.compress_chunk(_request())

    assert [candidate.candidate_type for candidate in result.candidate_memories] == [
        MemoryCandidateType.ACTION_ITEM.value
    ]


def test_memory_compression_allows_low_value_chunk_without_candidates() -> None:
    service = MemoryCompressionService(
        model_client=FakeModelClient(
            {
                "chunk_summary": "参会者进行寒暄，没有形成可写入长期记忆的事实。",
                "key_points": [],
                "open_questions": [],
                "candidate_memories": [],
                "source_refs": ["transcript:seg_0"],
                "compression_quality": 0.7,
                "coverage_score": 0.85,
                "loss_risk_score": 0.1,
            }
        )
    )

    result = service.compress_chunk(_request(max_candidate_memories=0))

    assert result.candidate_memories == []
    assert result.chunk_summary.startswith("参会者")


def test_memory_compression_merges_candidate_refs_into_top_level_refs() -> None:
    service = MemoryCompressionService(
        model_client=FakeModelClient(
            _valid_compression_response(
                source_refs=["transcript:seg_0"],
                candidate_memories=[
                    _candidate_payload(source_refs=["transcript:seg_99"]),
                ],
            )
        )
    )

    result = service.compress_chunk(_request())

    assert "transcript:seg_99" in result.source_refs


def test_memory_compression_uses_request_transcript_refs_when_model_omits_refs() -> None:
    service = MemoryCompressionService(
        model_client=FakeModelClient(
            {
                "chunk_summary": "本段讨论客户报价确认和法务审批。",
                "key_points": ["张三负责客户报价确认。"],
                "open_questions": [],
                "candidate_memories": [],
                "compression_quality": 0.7,
                "coverage_score": 0.7,
                "loss_risk_score": 0.2,
            }
        )
    )

    result = service.compress_chunk(_request(max_candidate_memories=0))

    assert result.source_refs == ["transcript:seg_0", "transcript:seg_1"]


def _request(max_candidate_memories: int = 8) -> MemoryCompressionRequest:
    return MemoryCompressionRequest(
        session_id="session_001",
        chunk_id="chunk_001",
        transcript_window=[
            TranscriptWindowItem(
                transcript_id="seg_0",
                speaker="Bao",
                text="张三负责客户报价确认，下周五截止。",
                timestamp_ms=0,
            ),
            TranscriptWindowItem(
                transcript_id="seg_1",
                speaker="Mia",
                text="李四是法务接口，她会确认合同隐私条款是否影响报价方案。",
                timestamp_ms=1200,
            ),
        ],
        session_context={
            "org_id": "org_001",
            "subject_user_id": "user_001",
            "metadata": {"participants": ["Bao", "Mia", "张三", "李四"]},
        },
        privacy_constraints=["不要在眼镜端直接展示客户敏感报价"],
        max_candidate_memories=max_candidate_memories,
    )


def _valid_compression_response(**overrides):  # type: ignore[no-untyped-def]
    payload = {
        "chunk_summary": "本段明确 Project Atlas 的客户报价确认负责人、截止时间和法务隐私条款接口。",
        "key_points": [
            "张三负责客户报价确认，下周五截止。",
            "李四是法务接口，负责确认合同隐私条款影响。",
        ],
        "open_questions": ["报价口径是否需要额外审批仍未确认。"],
        "candidate_memories": [
            _candidate_payload(),
            _candidate_payload(
                candidate_type="person_or_fact",
                text="李四是法务接口，负责确认合同隐私条款是否影响报价方案。",
                confidence=0.83,
                write_policy="eligible",
                privacy_level="medium",
                privacy_risk=0.45,
                source_refs=["transcript:seg_1"],
                entity="李四",
                owner="",
                deadline="",
                status="active",
                topic="合同隐私条款",
                tags=["person_or_fact", "legal"],
            ),
        ],
        "source_refs": ["transcript:seg_0", "transcript:seg_1"],
        "compression_quality": 0.82,
        "coverage_score": 0.78,
        "loss_risk_score": 0.18,
        "compression_notes": "保留了行动项和法务接口事实。",
        "safety_flags": [],
    }
    payload.update(overrides)
    return payload


def _candidate_payload(**overrides):  # type: ignore[no-untyped-def]
    payload = {
        "candidate_type": "action_item",
        "text": "张三负责客户报价确认，下周五截止。",
        "confidence": 0.86,
        "write_policy": "eligible",
        "privacy_level": "medium",
        "privacy_risk": 0.38,
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
