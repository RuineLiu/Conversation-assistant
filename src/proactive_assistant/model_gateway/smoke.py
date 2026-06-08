from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.model_gateway.clients import (
    ModelClient,
    ModelGatewayError,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from proactive_assistant.model_gateway.embeddings import EmbeddingClient, OpenAIEmbeddingClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.memory import (
    MemoryCompressionRequest,
    MemoryCompressionResult,
    MemoryCompressionService,
    MemoryExtractionRequest,
    MemoryExtractionResult,
    MemoryExtractionService,
)
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


class MemoryExtractionSmokeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_candidate_id: str
    candidate_type: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    write_policy: str
    privacy_level: str
    owner: str = ""
    deadline: str = ""
    status: str = ""
    entity: str = ""
    source_refs: list[str] = Field(default_factory=list)


class MemoryExtractionSmokeTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    quality_gate_passed: bool
    quality_checks: dict[str, bool]
    candidates: list[MemoryExtractionSmokeCandidate] = Field(default_factory=list)
    extraction_notes: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    raw_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class MemoryCompressionSmokeTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    chunk_summary: str
    key_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    candidate_count: int = Field(ge=0)
    compression_quality: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    loss_risk_score: float = Field(ge=0.0, le=1.0)
    source_refs: list[str] = Field(default_factory=list)
    quality_gate_passed: bool
    quality_checks: dict[str, bool]
    raw_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class EmbeddingSmokeTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    input_count: int = Field(ge=1)
    dimensions: int = Field(ge=1)
    vector_preview: list[float] = Field(default_factory=list)
    raw_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)


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


def run_openai_memory_extraction_smoke_test(
    *,
    settings: ModelGatewaySettings | None = None,
    model: str | None = None,
    model_client: ModelClient | None = None,
) -> MemoryExtractionSmokeTestResult:
    resolved_settings = settings or ModelGatewaySettings()
    client = model_client or _openai_client_from_settings(resolved_settings)
    extraction_service = MemoryExtractionService(model_client=client, settings=resolved_settings)
    result = extraction_service.extract_candidates(build_smoke_memory_extraction_request(), model=model)
    return memory_extraction_smoke_result_from_extraction(result)


def run_openai_memory_compression_smoke_test(
    *,
    settings: ModelGatewaySettings | None = None,
    model: str | None = None,
    model_client: ModelClient | None = None,
) -> MemoryCompressionSmokeTestResult:
    resolved_settings = settings or ModelGatewaySettings()
    client = model_client or _openai_client_from_settings(resolved_settings)
    compression_service = MemoryCompressionService(model_client=client, settings=resolved_settings)
    result = compression_service.compress_chunk(build_smoke_memory_compression_request(), model=model)
    return memory_compression_smoke_result_from_compression(result)


def run_openai_embedding_smoke_test(
    *,
    settings: ModelGatewaySettings | None = None,
    model: str | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> EmbeddingSmokeTestResult:
    resolved_settings = settings or ModelGatewaySettings()
    client = embedding_client or _openai_embedding_client_from_settings(resolved_settings)
    resolved_model = model or resolved_settings.embedding_model
    texts = [
        "李四是法务接口，负责客户数据合规风险确认。",
        "合同隐私条款归属",
    ]
    response = client.embed_texts(texts, model=resolved_model)
    first_vector = response.embeddings[0]
    return EmbeddingSmokeTestResult(
        ok=True,
        provider=response.provider,
        model=response.model,
        latency_ms=response.latency_ms,
        input_count=len(texts),
        dimensions=len(first_vector),
        vector_preview=[round(value, 6) for value in first_vector[:6]],
        raw_response_id=response.raw_response_id,
        input_tokens=response.input_tokens,
    )


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


def build_smoke_memory_extraction_request() -> MemoryExtractionRequest:
    return MemoryExtractionRequest(
        session_id="smoke_memory_extraction_001",
        scenario_id="meeting_business",
        locale="zh-CN",
        transcript_window=[
            TranscriptWindowItem(
                transcript_id="transcript_smoke_mem_001",
                speaker="Bao",
                text="今天 Project Atlas 的主要目标是把客户报价口径和上线风险对齐。",
                timestamp_ms=0,
                topic="Project Atlas",
            ),
            TranscriptWindowItem(
                transcript_id="transcript_smoke_mem_002",
                speaker="Mia",
                text="我确认一下，张三负责客户报价确认，下周五截止。",
                timestamp_ms=1200,
                topic="action item",
            ),
            TranscriptWindowItem(
                transcript_id="transcript_smoke_mem_003",
                speaker="Bao",
                text="刚才定了，MVP 版本先不做自动推送，只保留手动查看入口。",
                timestamp_ms=2500,
                topic="decision",
            ),
            TranscriptWindowItem(
                transcript_id="transcript_smoke_mem_004",
                speaker="Mia",
                text="李四是法务接口，她会确认合同隐私条款是否影响报价方案。",
                timestamp_ms=3800,
                topic="person fact",
            ),
            TranscriptWindowItem(
                transcript_id="transcript_smoke_mem_005",
                speaker="Bao",
                text="会后需要重点查漏：报价口径、法务审批、以及下周五前是否能完成确认。",
                timestamp_ms=5200,
                topic="summary gap",
            ),
        ],
        session_context={
            "status": "running",
            "pre_context": "这是一场项目风险会议。只验证结构化 memory extraction，不写入真实长期数据。",
            "org_id": "org_smoke",
            "subject_user_id": "user_smoke",
            "metadata": {
                "participants": ["Bao", "Mia", "张三", "李四"],
                "project": "Project Atlas",
            },
            "transcript_stats": {
                "segment_count": 5,
                "total_chars": 129,
                "duration_ms": 6400,
                "speaker_count": 2,
            },
        },
        meeting_state={
            "known_project": "Project Atlas",
            "expected_memory_types": ["action_item", "decision", "person_or_fact", "project_context", "summary"],
        },
        privacy_constraints=["不要在眼镜端直接暴露客户敏感报价细节"],
        max_candidates=8,
    )


def build_smoke_memory_compression_request() -> MemoryCompressionRequest:
    extraction_request = build_smoke_memory_extraction_request()
    return MemoryCompressionRequest(
        session_id=extraction_request.session_id,
        chunk_id="chunk_smoke_001",
        scenario_id=extraction_request.scenario_id,
        locale=extraction_request.locale,
        transcript_window=extraction_request.transcript_window,
        session_context=extraction_request.session_context,
        meeting_state=extraction_request.meeting_state,
        memory_context=extraction_request.memory_context,
        privacy_constraints=extraction_request.privacy_constraints,
        max_candidate_memories=6,
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


def memory_extraction_smoke_result_from_extraction(
    result: MemoryExtractionResult,
) -> MemoryExtractionSmokeTestResult:
    usage = result.model_usage
    if usage is None:
        raise ModelGatewayError("smoke test result did not include model usage metadata")
    candidates = [_smoke_candidate_from_memory_candidate(candidate) for candidate in result.candidates]
    checks = _memory_extraction_quality_checks(candidates)
    return MemoryExtractionSmokeTestResult(
        ok=True,
        provider=usage.provider,
        model=usage.model,
        latency_ms=usage.latency_ms,
        candidate_count=len(candidates),
        quality_gate_passed=all(checks.values()),
        quality_checks=checks,
        candidates=candidates,
        extraction_notes=result.extraction_notes,
        safety_flags=list(result.safety_flags),
        raw_response_id=usage.raw_response_id,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def memory_compression_smoke_result_from_compression(
    result: MemoryCompressionResult,
) -> MemoryCompressionSmokeTestResult:
    usage = result.model_usage
    if usage is None:
        raise ModelGatewayError("smoke test result did not include model usage metadata")
    checks = _memory_compression_quality_checks(result)
    return MemoryCompressionSmokeTestResult(
        ok=True,
        provider=usage.provider,
        model=usage.model,
        latency_ms=usage.latency_ms,
        chunk_summary=result.chunk_summary,
        key_points=list(result.key_points),
        open_questions=list(result.open_questions),
        candidate_count=len(result.candidate_memories),
        compression_quality=result.compression_quality,
        coverage_score=result.coverage_score,
        loss_risk_score=result.loss_risk_score,
        source_refs=list(result.source_refs),
        quality_gate_passed=all(
            checks[key]
            for key in [
                "has_summary",
                "has_source_refs",
                "candidate_refs_are_covered",
                "quality_not_low",
                "loss_risk_not_high",
            ]
        ),
        quality_checks=checks,
        raw_response_id=usage.raw_response_id,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def _smoke_candidate_from_memory_candidate(candidate: object) -> MemoryExtractionSmokeCandidate:
    metadata = getattr(candidate, "metadata")
    return MemoryExtractionSmokeCandidate(
        memory_candidate_id=getattr(candidate, "memory_candidate_id"),
        candidate_type=str(getattr(candidate, "candidate_type")),
        text=str(getattr(candidate, "text")),
        confidence=float(getattr(candidate, "confidence")),
        write_policy=str(getattr(candidate, "write_policy")),
        privacy_level=str(getattr(candidate, "privacy_level")),
        owner=str(metadata.get("owner", "")),
        deadline=str(metadata.get("deadline", "")),
        status=str(metadata.get("status", "")),
        entity=str(metadata.get("entity", metadata.get("canonical_entity", ""))),
        source_refs=[str(item) for item in metadata.get("source_refs", [])],
    )


def _memory_extraction_quality_checks(candidates: list[MemoryExtractionSmokeCandidate]) -> dict[str, bool]:
    action_items = [candidate for candidate in candidates if candidate.candidate_type == "action_item"]
    return {
        "has_candidates": bool(candidates),
        "has_action_item": bool(action_items),
        "has_owner_zhangsan": any(candidate.owner == "张三" or "张三" in candidate.text for candidate in action_items),
        "has_deadline_next_friday": any("下周五" in candidate.deadline or "下周五" in candidate.text for candidate in action_items),
        "has_decision": any(candidate.candidate_type == "decision" for candidate in candidates),
        "has_source_refs": all(candidate.source_refs for candidate in candidates),
        "no_blocked_candidates": all(candidate.write_policy != "blocked" for candidate in candidates),
    }


def _memory_compression_quality_checks(result: MemoryCompressionResult) -> dict[str, bool]:
    refs = set(result.source_refs)
    candidate_refs = {
        ref
        for candidate in result.candidate_memories
        for ref in candidate.metadata.get("source_refs", [])
        if isinstance(ref, str)
    }
    return {
        "has_summary": len(result.chunk_summary) >= 12,
        "has_source_refs": bool(result.source_refs),
        "candidate_refs_are_covered": candidate_refs.issubset(refs),
        "quality_not_low": result.compression_quality >= 0.55,
        "loss_risk_not_high": result.loss_risk_score <= 0.55,
        "has_memory_candidates": len(result.candidate_memories) >= 1,
    }


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


def _openai_embedding_client_from_settings(settings: ModelGatewaySettings) -> EmbeddingClient:
    if settings.openai_api_key is None:
        raise ModelGatewayError("OPENAI_API_KEY or PROACTIVE_OPENAI_API_KEY is required for live smoke tests")
    return OpenAIEmbeddingClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
