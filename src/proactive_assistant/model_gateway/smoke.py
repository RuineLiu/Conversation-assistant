from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.model_gateway.clients import (
    ModelClient,
    ModelGatewayError,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import (
    PromptGenerationRequest,
    PromptGenerationResult,
    PromptGenerationService,
    TranscriptWindowItem,
)


class PromptSmokeTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    ok: bool
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    should_prompt: bool
    prompt_category: str | None = None
    content_granularity: int = Field(ge=0, le=4)
    glasses_title: str = ""
    glasses_text: str = ""
    app_detail_text: str = ""
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    privacy_level: str
    privacy_risk: float = Field(ge=0.0, le=1.0)
    raw_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


def run_openai_prompt_smoke_test(
    *,
    settings: ModelGatewaySettings | None = None,
    model: str | None = None,
    model_client: ModelClient | None = None,
) -> PromptSmokeTestResult:
    resolved_settings = settings or ModelGatewaySettings()
    client = model_client or _openai_client_from_settings(resolved_settings)
    prompt_service = PromptGenerationService(model_client=client, settings=resolved_settings)
    result = prompt_service.generate_prompt(build_smoke_prompt_request(), model=model)
    return prompt_smoke_result_from_generation(result)


def build_smoke_prompt_request() -> PromptGenerationRequest:
    return PromptGenerationRequest(
        session_id="smoke_openai_prompt_001",
        scenario_id="meeting_business",
        locale="zh-CN",
        transcript_window=[
            TranscriptWindowItem(
                transcript_id="transcript_smoke_001",
                speaker="Bao",
                text="这个问题谁负责，下周五 deadline 前能不能定？",
                timestamp_ms=0,
                topic="launch risk ownership",
            )
        ],
        session_context={
            "status": "running",
            "pre_context": "这是一场项目风险会议。只验证结构化输出是否可用，不写入任何长期数据。",
            "transcript_stats": {
                "segment_count": 1,
                "total_chars": 25,
                "duration_ms": 1200,
                "speaker_count": 1,
            },
        },
        privacy_constraints=["do not expose sensitive customer data"],
        prompt_category_candidate="summary_gap_check",
        target_content_granularity=2,
        prd_surface="glasses_popup",
        display_mode="auto",
        duration_policy="5s",
    )


def prompt_smoke_result_from_generation(result: PromptGenerationResult) -> PromptSmokeTestResult:
    usage = result.model_usage
    if usage is None:
        raise ModelGatewayError("smoke test result did not include model usage metadata")
    return PromptSmokeTestResult(
        ok=True,
        provider=usage.provider,
        model=usage.model,
        latency_ms=usage.latency_ms,
        should_prompt=result.should_prompt,
        prompt_category=str(result.prompt_category) if result.prompt_category is not None else None,
        content_granularity=int(result.content_granularity),
        glasses_title=result.glasses_title,
        glasses_text=result.glasses_text,
        app_detail_text=result.app_detail_text,
        source_refs=list(result.source_refs),
        confidence=result.confidence,
        privacy_level=str(result.privacy_level),
        privacy_risk=result.privacy_risk,
        raw_response_id=usage.raw_response_id,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def _openai_client_from_settings(settings: ModelGatewaySettings) -> ModelClient:
    if settings.openai_api_key is None:
        raise ModelGatewayError("OPENAI_API_KEY or PROACTIVE_OPENAI_API_KEY is required for live smoke tests")
    if settings.model_api_style == "chat_completions" or settings.openai_base_url is not None:
        return OpenAIChatCompletionsClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            response_format=settings.chat_response_format,
            max_tokens_param=settings.chat_max_tokens_param,
        )
    return OpenAIResponsesClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
