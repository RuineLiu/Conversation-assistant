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
        return _parse_prompt_result(response, request)


def _parse_prompt_result(
    response: ModelResponse,
    request: PromptGenerationRequest | None = None,
) -> PromptGenerationResult:
    try:
        output = PromptGenerationModelOutput.model_validate(
            _normalize_prompt_payload(response.parsed, request=request)
        )
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


def _normalize_prompt_payload(
    payload: object,
    *,
    request: PromptGenerationRequest | None = None,
) -> object:
    if not isinstance(payload, dict):
        return payload

    allowed_fields = set(PromptGenerationModelOutput.model_fields)
    payload = _apply_prompt_aliases(payload)
    normalized = {key: value for key, value in payload.items() if key in allowed_fields}

    should_prompt = bool(normalized.get("should_prompt", _infer_should_prompt(normalized)))
    normalized["should_prompt"] = should_prompt
    if should_prompt and normalized.get("prompt_category") is None and request is not None:
        normalized["prompt_category"] = request.prompt_category_candidate
    normalized.setdefault("content_granularity", 2 if should_prompt else 0)
    normalized.setdefault("confidence", 0.7 if should_prompt else 0.0)
    normalized.setdefault("privacy_level", "low")
    normalized.setdefault("privacy_risk", _default_privacy_risk(normalized["privacy_level"]))
    normalized["source_refs"] = _normalize_source_refs(normalized.get("source_refs", []))
    if should_prompt and not normalized["source_refs"] and request is not None:
        normalized["source_refs"] = _fallback_source_refs(request)
    normalized.setdefault("rationale", "")
    normalized.setdefault("safety_flags", [])
    return normalized


def _apply_prompt_aliases(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    aliases = {
        "should_show": "should_prompt",
        "show_prompt": "should_prompt",
        "category": "prompt_category",
        "prompt_type": "prompt_category",
        "title": "glasses_title",
        "answer": "glasses_text",
        "text": "glasses_text",
        "detail": "app_detail_text",
        "details": "app_detail_text",
    }
    for source, target in aliases.items():
        if target not in normalized and source in normalized:
            normalized[target] = normalized[source]
    return normalized


def _default_privacy_risk(privacy_level: object) -> float:
    if privacy_level == "high":
        return 0.85
    if privacy_level == "medium":
        return 0.35
    return 0.08


def _infer_should_prompt(payload: dict[str, object]) -> bool:
    if payload.get("prompt_category") is not None:
        return True
    for key in ("glasses_title", "glasses_text", "app_detail_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return True
    granularity = payload.get("content_granularity")
    if isinstance(granularity, int) and granularity > 0:
        return True
    return False


def _normalize_source_refs(source_refs: object) -> list[str]:
    if isinstance(source_refs, str):
        source_refs = [source_refs]
    if not isinstance(source_refs, list):
        return []

    refs: list[str] = []
    for ref in source_refs:
        if isinstance(ref, str):
            clean_ref = ref.strip()
            if clean_ref:
                refs.append(clean_ref if ":" in clean_ref else f"transcript:{clean_ref}")
            continue
        if isinstance(ref, dict):
            ref_type = ref.get("type") or ref.get("source_type") or "source"
            ref_id = ref.get("id") or ref.get("transcript_id") or ref.get("source_id")
            if ref_id:
                refs.append(f"{ref_type}:{ref_id}")
    return refs


def _fallback_source_refs(request: PromptGenerationRequest) -> list[str]:
    refs: list[str] = []
    for item in request.transcript_window[-3:]:
        transcript_id = item.transcript_id.strip()
        if transcript_id:
            refs.append(f"transcript:{transcript_id}")
    return refs or ["transcript:unknown"]
