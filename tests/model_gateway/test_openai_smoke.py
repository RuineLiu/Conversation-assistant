import pytest

from proactive_assistant.model_gateway import (
    FakeModelClient,
    ModelGatewayError,
    ModelGatewaySettings,
)
from proactive_assistant.model_gateway.smoke import run_openai_prompt_smoke_test


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


def test_openai_prompt_smoke_runner_requires_api_key_without_injected_client() -> None:
    settings = ModelGatewaySettings(openai_api_key=None, default_model="gpt-test")

    with pytest.raises(ModelGatewayError, match="OPENAI_API_KEY"):
        run_openai_prompt_smoke_test(settings=settings)
