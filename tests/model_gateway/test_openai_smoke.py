import pytest

from proactive_assistant.model_gateway import (
    FakeModelClient,
    ModelGatewayError,
    ModelGatewaySettings,
)
from proactive_assistant.model_gateway.smoke import run_openai_prompt_smoke_test
from proactive_assistant.model_gateway.smoke import run_openai_memory_extraction_smoke_test
from proactive_assistant.model_gateway.smoke import run_openai_memory_compression_smoke_test


def valid_prompt_response() -> dict[str, object]:
    return {
        "should_prompt": True,
        "prompt_category": "summary_gap_check",
        "content_granularity": 2,
        "glasses_title": "负责人待确认",
        "glasses_text": "这个风险还没有明确 owner 和截止时间。",
        "app_detail_text": "会议中出现 owner/deadline gap，需要确认负责人、截止时间和下一步。",
        "source_refs": ["transcript:transcript_smoke_001"],
        "confidence": 0.84,
        "privacy_level": "low",
        "privacy_risk": 0.08,
        "rationale": "检测到未确认的负责人和 deadline。",
        "safety_flags": [],
    }


def valid_memory_extraction_response() -> dict[str, object]:
    return {
        "candidates": [
            {
                "candidate_type": "action_item",
                "text": "张三负责客户报价确认，下周五截止。",
                "confidence": 0.86,
                "write_policy": "eligible",
                "privacy_level": "medium",
                "privacy_risk": 0.35,
                "source_refs": ["transcript:transcript_smoke_mem_002"],
                "reason": "明确出现负责人和截止时间。",
                "entity": "客户报价确认",
                "owner": "张三",
                "deadline": "下周五",
                "status": "open",
                "topic": "客户报价",
                "tags": ["action_item"],
                "promotion_candidate": False,
            },
            {
                "candidate_type": "decision",
                "text": "MVP 版本先不做自动推送，只保留手动查看入口。",
                "confidence": 0.82,
                "write_policy": "eligible",
                "privacy_level": "low",
                "privacy_risk": 0.08,
                "source_refs": ["transcript:transcript_smoke_mem_003"],
                "reason": "明确出现会议结论。",
                "entity": "MVP 自动推送",
                "owner": "",
                "deadline": "",
                "status": "decided",
                "topic": "MVP 范围",
                "tags": ["decision"],
                "promotion_candidate": False,
            },
        ],
        "extraction_notes": "smoke extraction fixture",
        "safety_flags": [],
    }


def valid_memory_compression_response() -> dict[str, object]:
    return {
        "chunk_summary": "本段确认 Project Atlas 的客户报价负责人、截止时间和 MVP 手动入口决策。",
        "key_points": ["张三负责客户报价确认，下周五截止。"],
        "open_questions": ["法务审批是否会影响报价方案仍需确认。"],
        "candidate_memories": [
            {
                "candidate_type": "action_item",
                "text": "张三负责客户报价确认，下周五截止。",
                "confidence": 0.86,
                "write_policy": "eligible",
                "privacy_level": "medium",
                "privacy_risk": 0.35,
                "source_refs": ["transcript:transcript_smoke_mem_002"],
                "reason": "明确出现负责人和截止时间。",
                "entity": "客户报价确认",
                "owner": "张三",
                "deadline": "下周五",
                "status": "open",
                "topic": "客户报价",
                "tags": ["action_item"],
                "promotion_candidate": False,
            }
        ],
        "source_refs": ["transcript:transcript_smoke_mem_002"],
        "compression_quality": 0.82,
        "coverage_score": 0.78,
        "loss_risk_score": 0.18,
        "compression_notes": "smoke compression fixture",
        "safety_flags": [],
    }


def test_openai_prompt_smoke_runner_uses_prompt_contract_with_injected_client() -> None:
    client = FakeModelClient(valid_prompt_response(), latency_ms=4)
    settings = ModelGatewaySettings(openai_api_key=None, default_model="gpt-test", max_output_tokens=256)

    result = run_openai_prompt_smoke_test(settings=settings, model_client=client, model="gpt-live-test")

    assert result.ok is True
    assert result.provider == "fake"
    assert result.model == "gpt-live-test"
    assert result.should_prompt is True
    assert result.prompt_category == "summary_gap_check"
    assert result.content_granularity == 2
    assert result.source_refs == ["transcript:transcript_smoke_001"]
    assert client.requests[0].model == "gpt-live-test"
    assert client.requests[0].response_schema_name == "PromptGenerationResult"
    assert client.requests[0].max_output_tokens == 256
    assert "这个问题谁负责" in client.requests[0].input_text
    assert "summary_gap_check" in client.requests[0].input_text


def test_openai_memory_extraction_smoke_runner_reports_quality_checks() -> None:
    client = FakeModelClient(valid_memory_extraction_response(), latency_ms=5)
    settings = ModelGatewaySettings(openai_api_key=None, default_model="gpt-test", max_output_tokens=512)

    result = run_openai_memory_extraction_smoke_test(settings=settings, model_client=client, model="gpt-live-test")

    assert result.ok is True
    assert result.provider == "fake"
    assert result.model == "gpt-live-test"
    assert result.candidate_count == 2
    assert result.quality_gate_passed is True
    assert result.quality_checks == {
        "has_candidates": True,
        "has_action_item": True,
        "has_owner_zhangsan": True,
        "has_deadline_next_friday": True,
        "has_decision": True,
        "has_source_refs": True,
        "no_blocked_candidates": True,
    }
    assert result.candidates[0].owner == "张三"
    assert result.candidates[0].deadline == "下周五"
    assert client.requests[0].model == "gpt-live-test"
    assert client.requests[0].response_schema_name == "MemoryExtractionResult"
    assert client.requests[0].metadata["contract"] == "memory_extraction_v1"
    assert "Project Atlas" in client.requests[0].input_text


def test_openai_memory_compression_smoke_runner_reports_quality_checks() -> None:
    client = FakeModelClient(valid_memory_compression_response(), latency_ms=6)
    settings = ModelGatewaySettings(openai_api_key=None, default_model="gpt-test", max_output_tokens=768)

    result = run_openai_memory_compression_smoke_test(settings=settings, model_client=client, model="gpt-live-test")

    assert result.ok is True
    assert result.provider == "fake"
    assert result.model == "gpt-live-test"
    assert result.candidate_count == 1
    assert result.compression_quality == 0.82
    assert result.quality_gate_passed is True
    assert result.quality_checks == {
        "has_summary": True,
        "has_source_refs": True,
        "candidate_refs_are_covered": True,
        "quality_not_low": True,
        "loss_risk_not_high": True,
        "has_memory_candidates": True,
    }
    assert client.requests[0].model == "gpt-live-test"
    assert client.requests[0].response_schema_name == "MemoryCompressionResult"
    assert client.requests[0].metadata["contract"] == "memory_compression_v1"
    assert client.requests[0].metadata["chunk_id"] == "chunk_smoke_001"


def test_openai_prompt_smoke_runner_requires_api_key_without_injected_client() -> None:
    settings = ModelGatewaySettings(openai_api_key=None, default_model="gpt-test")

    with pytest.raises(ModelGatewayError, match="OPENAI_API_KEY"):
        run_openai_prompt_smoke_test(settings=settings)
