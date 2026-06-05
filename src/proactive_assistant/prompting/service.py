from pydantic import ValidationError

from proactive_assistant.model_gateway import (
    ModelClient,
    ModelOutputValidationError,
    ModelRequest,
    ModelResponse,
)
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting.builder import SYSTEM_INSTRUCTIONS, build_prompt_generation_input
from proactive_assistant.prompting.contracts import (
    ModelUsageMetadata,
    PromptGenerationModelOutput,
    PromptGenerationRequest,
    PromptGenerationResult,
    openai_strict_json_schema,
)


class PromptGenerationService:
    """Business-facing service for PRD-fit prompt generation."""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        settings: ModelGatewaySettings | None = None,
    ) -> None:
        self._model_client = model_client
        self._settings = settings or ModelGatewaySettings()

    def generate_prompt(
        self,
        request: PromptGenerationRequest,
        *,
        model: str | None = None,
    ) -> PromptGenerationResult:
        model_request = ModelRequest(
            model=model or self._settings.default_model,
            instructions=SYSTEM_INSTRUCTIONS,
            input_text=build_prompt_generation_input(request),
            response_schema_name="PromptGenerationResult",
            response_schema=openai_strict_json_schema(PromptGenerationModelOutput),
            max_output_tokens=self._settings.max_output_tokens,
            store=self._settings.store_model_responses,
            metadata={
                "session_id": request.session_id[:64],
                "scenario_id": request.scenario_id[:64],
                "contract": "prompt_generation_v1",
            },
        )
        response = self._model_client.generate_structured(model_request)
        return _parse_prompt_result(response)


def _parse_prompt_result(response: ModelResponse) -> PromptGenerationResult:
    try:
        output = PromptGenerationModelOutput.model_validate(response.parsed)
    except ValidationError as exc:
        raise ModelOutputValidationError("model output failed PromptGenerationResult validation") from exc
    usage = ModelUsageMetadata(
        provider=response.provider,
        model=response.model,
        latency_ms=response.latency_ms,
        raw_response_id=response.raw_response_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached=response.cached,
    )
    result = PromptGenerationResult.model_validate(output.model_dump(mode="json"))
    return result.with_usage(usage)
