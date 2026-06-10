import json

import pytest

from proactive_assistant.model_gateway import FakeModelClient, ModelOutputValidationError
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import (
    GlassesLengthLimits,
    PromptGenerationRequest,
    PromptGenerationService,
    TranscriptWindowItem,
)


def make_prompt_request() -> PromptGenerationRequest:
    return PromptGenerationRequest(
        session_id="meeting_001",
        transcript_window=[
            TranscriptWindowItem(
                transcript_id="transcript_001",
                speaker="Bao",
                text="Who owns the launch risk follow-up?",
            )
        ],
        session_context={"meeting_goal": "align launch risks"},
    )


def test_prompt_generation_service_validates_fake_model_output() -> None:
    client = FakeModelClient(
        {
            "should_prompt": True,
            "prompt_category": "summary_gap_check",
            "content_granularity": 2,
            "glasses_title": "Owner missing",
            "glasses_text": "Launch risk owner is still unconfirmed.",
            "app_detail_text": "Bao asked who owns the launch risk follow-up; no owner is present in the context.",
            "source_refs": ["transcript:transcript_001"],
            "confidence": 0.84,
            "privacy_level": "low",
            "privacy_risk": 0.08,
            "rationale": "The transcript contains an unresolved ownership gap.",
            "safety_flags": [],
        },
        latency_ms=3,
    )
    service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test", max_output_tokens=256),
    )

    result = service.generate_prompt(make_prompt_request())

    assert result.should_prompt is True
    assert result.glasses_text == "Launch risk owner is still unconfirmed."
    assert result.model_usage is not None
    assert result.model_usage.model == "gpt-test"
    assert result.model_usage.cached is True
    assert client.requests[0].response_schema_name == "PromptGenerationResult"
    assert client.requests[0].max_output_tokens == 256
    assert "Who owns the launch risk" in client.requests[0].input_text
    assert "HARD LIMITS" in client.requests[0].instructions
    assert "when possible" not in client.requests[0].instructions
    assert "40 Chinese characters" not in client.requests[0].instructions
    assert "prd_glasses_card" in client.requests[0].input_text
    assert "explain unfamiliar terms" in client.requests[0].input_text
    assert "when possible" not in client.requests[0].input_text
    assert "40 Chinese characters" not in client.requests[0].input_text

    payload = json.loads(client.requests[0].input_text)
    limits = GlassesLengthLimits()
    hard_limits = payload["output_rules"]["prd_glasses_card"]["hard_limits"]
    assert hard_limits["hard_limit"] is True
    assert hard_limits["zh"]["glasses_title_max"] == limits.title_cjk_chars
    assert hard_limits["zh"]["glasses_text_max"] == limits.text_cjk_chars
    assert hard_limits["en"]["glasses_title_max"] == limits.title_en_words
    assert hard_limits["en"]["glasses_text_max"] == limits.text_en_words
    assert payload["output_rules"]["prd_glasses_card"]["detail_destination"] == "app_detail_text"
    assert "content_granularity=1" in payload["output_rules"]["prd_glasses_card"]["overflow_policy"]


def test_prompt_generation_service_normalizes_loose_openai_compatible_output() -> None:
    client = FakeModelClient(
        {
            "should_prompt": True,
            "prompt_category": "summary_gap_check",
            "content_granularity": 2,
            "glasses_title": "风险待确认",
            "glasses_text": "需要确认负责人和截止时间。",
            "app_detail_text": "模型可能返回额外字段，解析层需要宽进严出。",
            "source_refs": [{"type": "transcript", "id": "transcript_001", "timestamp_ms": 0}],
            "privacy_level": "low",
            "memory_update": None,
        },
        latency_ms=3,
    )
    service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )

    result = service.generate_prompt(make_prompt_request())

    assert result.should_prompt is True
    assert result.confidence == 0.7
    assert result.privacy_risk == 0.08
    assert result.source_refs == ["transcript:transcript_001"]
    assert not hasattr(result, "memory_update")


def test_prompt_generation_service_rejects_invalid_model_output() -> None:
    client = FakeModelClient(
        {
            "should_prompt": True,
            "content_granularity": 2,
            "confidence": 0.7,
            "privacy_risk": 0.1,
        }
    )
    service = PromptGenerationService(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )

    with pytest.raises(ModelOutputValidationError):
        service.generate_prompt(make_prompt_request())
