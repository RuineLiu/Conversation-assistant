import pytest
from pydantic import ValidationError

from proactive_assistant.prompting import (
    ContentGranularity,
    PromptCategory,
    PromptGenerationModelOutput,
    PromptGenerationRequest,
    PromptGenerationResult,
    TranscriptWindowItem,
    openai_strict_json_schema,
)


def test_prompt_generation_request_accepts_prd_context() -> None:
    request = PromptGenerationRequest(
        session_id="session_001",
        transcript_window=[
            TranscriptWindowItem(
                transcript_id="transcript_001",
                speaker="Alex",
                text="Can we confirm the budget owner?",
            )
        ],
        prompt_category_candidate=PromptCategory.SUGGESTION,
        target_content_granularity=ContentGranularity.ONE_LINE_ANSWER,
        privacy_constraints=["avoid customer names"],
    )

    assert request.scenario_id == "meeting_business_v1"
    assert request.prd_surface == "glasses_popup"
    assert request.display_mode == "auto"


def test_prompt_generation_result_requires_no_action_granularity_when_not_prompting() -> None:
    result = PromptGenerationResult(
        should_prompt=False,
        content_granularity=ContentGranularity.NO_ACTION,
        confidence=0.2,
        privacy_risk=0.1,
    )

    assert result.should_prompt is False


def test_prompt_generation_result_requires_content_for_prompt() -> None:
    with pytest.raises(ValidationError):
        PromptGenerationResult(
            should_prompt=True,
            prompt_category=PromptCategory.QUESTION_ANSWER,
            content_granularity=ContentGranularity.ONE_LINE_ANSWER,
            confidence=0.8,
            privacy_risk=0.1,
            source_refs=["transcript:1"],
        )


def test_prompt_generation_result_accepts_valid_prompt() -> None:
    result = PromptGenerationResult(
        should_prompt=True,
        prompt_category=PromptCategory.SUMMARY_GAP_CHECK,
        content_granularity=ContentGranularity.CONCISE_BULLETS,
        glasses_title="Action item gap",
        glasses_text="Budget owner is still unconfirmed.",
        app_detail_text="The latest transcript asks about budget, but no owner has accepted it yet.",
        source_refs=["transcript:transcript_001"],
        confidence=0.87,
        privacy_risk=0.1,
    )

    assert result.content_granularity == 3


def test_openai_strict_schema_excludes_gateway_usage_metadata() -> None:
    schema = openai_strict_json_schema(PromptGenerationModelOutput)

    assert schema["additionalProperties"] is False
    assert "model_usage" not in schema["properties"]
    assert set(schema["required"]) == set(schema["properties"])
