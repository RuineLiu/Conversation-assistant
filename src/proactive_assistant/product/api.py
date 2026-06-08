import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.meeting_state import MeetingGap, MeetingState
from proactive_assistant.memory import (
    MemoryContext,
    MemoryExtractionService,
    MemoryForgetResult,
    MemoryNotFoundError,
    MemoryPendingUpdate,
    MemoryPendingUpdateNotFoundError,
    MemoryPendingUpdateStatus,
    MemoryPromotionResult,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemoryType,
    MemoryUpsertResult,
)
from proactive_assistant.model_gateway import (
    ModelGatewayError,
    ModelGatewaySettings,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product.contracts import ProductMemoryExtractionResult, ProductPromptPayload
from proactive_assistant.product.service import ProductAssistantService
from proactive_assistant.prompting import PRDSurface, PromptCategory, PromptGenerationService
from proactive_assistant.persistence import SQLiteMemoryStore, SQLiteRuntimeStore, SQLiteSessionStore
from proactive_assistant.runtime import (
    FeedbackPolarity,
    FeedbackSignalSource,
    FeedbackSignalType,
    MemoryCandidate,
    PromptRuntimeService,
    RewardObservation,
    RuntimeFeedbackEvent,
)
from proactive_assistant.runtime.store import (
    DecisionRecordNotFoundError,
    RuntimeFeedbackEventAlreadyExistsError,
)
from proactive_assistant.sessions import (
    AssistantSession,
    InMemorySessionStore,
    SessionConfig,
    SessionService,
    TranscriptSegmentInput,
    TranscriptSegmentRecord,
)
from proactive_assistant.sessions.store import SessionAlreadyExistsError, SessionNotFoundError


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str


class CreateProductSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    config: SessionConfig = Field(default_factory=SessionConfig)
    start: bool = True


class SessionCreatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    meeting_state: MeetingState


class SessionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessions: list[AssistantSession]


class AppendTranscriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment: TranscriptSegmentInput
    segment_id: str | None = None
    max_segments: int = Field(default=12, ge=1, le=100)
    memory_context: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    use_memory: bool = True
    memory_query_text: str | None = None
    memory_limit: int = Field(default=8, ge=1, le=50)
    include_pending_memory: bool = False
    policy_version: str = "product_flow_v0"


class AppendTranscriptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AssistantSession
    transcript_segment: TranscriptSegmentRecord
    meeting_state: MeetingState
    meeting_gaps: list[MeetingGap] = Field(default_factory=list)
    retrieved_memory_context: MemoryContext | None = None
    prompts: list[ProductPromptPayload] = Field(default_factory=list)
    opportunity_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)


class MeetingStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    meeting_state: MeetingState


class PromptListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    prompts: list[ProductPromptPayload]


class RecordFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_type: FeedbackSignalType
    event_id: str | None = None
    source: FeedbackSignalSource | None = None
    polarity: FeedbackPolarity | None = None
    intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    text: str = ""
    dwell_ms: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    helpfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    timing_fit: float | None = Field(default=None, ge=-1.0, le=1.0)
    content_fit: float | None = Field(default=None, ge=-1.0, le=1.0)
    task_progress_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    flow_break_score: float | None = Field(default=None, ge=0.0, le=1.0)
    redundancy_score: float | None = Field(default=None, ge=0.0, le=1.0)
    privacy_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    compute_reward: bool = True
    propose_memory: bool = True
    commit_memory: bool = True


class FeedbackRecordedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    feedback_event: RuntimeFeedbackEvent
    reward_observation: RewardObservation | None = None
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memories: list[MemoryRecord] = Field(default_factory=list)


class MemoryCandidateListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)


class MemoryContextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    memory_context: MemoryContext


class MemorySnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: bool = True
    include_gaps: bool = True


class MemorySnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memories: list[MemoryRecord] = Field(default_factory=list)
    memory_upserts: list[MemoryUpsertResult] = Field(default_factory=list)
    committed: bool = True


class MemoryExtractionRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: bool = True
    max_segments: int = Field(default=24, ge=1, le=200)
    max_candidates: int = Field(default=8, ge=1, le=20)
    include_meeting_state: bool = True
    memory_context: list[str] = Field(default_factory=list)
    model: str | None = None


class MemoryMutationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = ""


class MemoryMutationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: MemoryRecord


class MemoryForgetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_forget: MemoryForgetResult


class MemoryPendingUpdateListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    memory_id: str
    status: MemoryPendingUpdateStatus | None = None
    pending_updates: list[MemoryPendingUpdate] = Field(default_factory=list)


class MemoryPendingUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pending_update: MemoryPendingUpdate


class MemoryPendingUpdateApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_upsert: MemoryUpsertResult


class MemoryPromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    target_scope: MemoryScope | None = None
    target_memory_type: MemoryType | None = None
    reason: str = ""
    approved: bool = False


class MemoryPromotionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_promotion: MemoryPromotionResult


def create_app(product_service: ProductAssistantService | None = None) -> FastAPI:
    service = product_service or create_default_product_service()
    app = FastAPI(
        title="Proactive Assistant Product API",
        version="0.1.0",
    )
    app.state.product_service = service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="product-api")

    @app.post("/sessions", response_model=SessionCreatedResponse, status_code=201)
    def create_session(request: CreateProductSessionRequest | None = None) -> SessionCreatedResponse:
        resolved_request = request or CreateProductSessionRequest()
        try:
            session = service.create_session(
                resolved_request.config,
                session_id=resolved_request.session_id,
                start=resolved_request.start,
            )
        except SessionAlreadyExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SessionCreatedResponse(session=session, meeting_state=service.get_meeting_state(session.session_id))

    @app.get("/sessions", response_model=SessionListResponse)
    def list_sessions() -> SessionListResponse:
        return SessionListResponse(sessions=service.sessions.list_sessions())

    @app.get("/sessions/{session_id}", response_model=SessionCreatedResponse)
    def get_session(session_id: str) -> SessionCreatedResponse:
        try:
            session = service.sessions.get_session(session_id)
            meeting_state = service.get_meeting_state(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SessionCreatedResponse(session=session, meeting_state=meeting_state)

    @app.post("/sessions/{session_id}/transcript", response_model=AppendTranscriptResponse)
    def append_transcript(session_id: str, request: AppendTranscriptRequest) -> AppendTranscriptResponse:
        try:
            result = service.append_transcript_and_generate_prompts(
                session_id,
                request.segment,
                segment_id=request.segment_id,
                max_segments=request.max_segments,
                memory_context=request.memory_context,
                memory_refs=request.memory_refs,
                use_memory=request.use_memory,
                memory_query_text=request.memory_query_text,
                memory_limit=request.memory_limit,
                include_pending_memory=request.include_pending_memory,
                policy_version=request.policy_version,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if result.meeting_state is None:
            raise HTTPException(status_code=500, detail="meeting state was not returned")
        return AppendTranscriptResponse(
            session=result.session,
            transcript_segment=result.transcript_segment,
            meeting_state=result.meeting_state,
            meeting_gaps=result.meeting_gaps,
            retrieved_memory_context=result.retrieved_memory_context,
            prompts=result.prompts,
            opportunity_count=result.opportunity_count,
            candidate_count=result.candidate_count,
        )

    @app.get("/sessions/{session_id}/meeting-state", response_model=MeetingStateResponse)
    def get_meeting_state(session_id: str) -> MeetingStateResponse:
        try:
            meeting_state = service.get_meeting_state(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return MeetingStateResponse(session_id=session_id, meeting_state=meeting_state)

    @app.get("/prompts", response_model=PromptListResponse)
    def list_prompts(session_id: str | None = Query(default=None)) -> PromptListResponse:
        return PromptListResponse(session_id=session_id, prompts=service.list_prompt_payloads(session_id=session_id))

    @app.get("/sessions/{session_id}/prompts", response_model=PromptListResponse)
    def list_session_prompts(session_id: str) -> PromptListResponse:
        try:
            service.sessions.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return PromptListResponse(session_id=session_id, prompts=service.list_prompt_payloads(session_id=session_id))

    @app.post("/prompt-decisions/{decision_id}/feedback", response_model=FeedbackRecordedResponse)
    def record_feedback(decision_id: str, request: RecordFeedbackRequest) -> FeedbackRecordedResponse:
        try:
            result = service.record_feedback(
                decision_id,
                request.signal_type,
                event_id=request.event_id,
                source=request.source,
                polarity=request.polarity,
                intensity=request.intensity,
                text=request.text,
                dwell_ms=request.dwell_ms,
                latency_ms=request.latency_ms,
                helpfulness=request.helpfulness,
                timing_fit=request.timing_fit,
                content_fit=request.content_fit,
                task_progress_delta=request.task_progress_delta,
                flow_break_score=request.flow_break_score,
                redundancy_score=request.redundancy_score,
                privacy_risk_score=request.privacy_risk_score,
                metadata=request.metadata,
                compute_reward=request.compute_reward,
                propose_memory=request.propose_memory,
                commit_memory=request.commit_memory,
            )
        except DecisionRecordNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeFeedbackEventAlreadyExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return FeedbackRecordedResponse(
            decision_id=decision_id,
            feedback_event=result.feedback_event,
            reward_observation=result.reward_observation,
            memory_candidates=result.memory_candidates,
            memories=result.memories,
        )

    @app.get("/sessions/{session_id}/memory-candidates", response_model=MemoryCandidateListResponse)
    def list_session_memory_candidates(session_id: str) -> MemoryCandidateListResponse:
        try:
            service.sessions.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return MemoryCandidateListResponse(
            session_id=session_id,
            memory_candidates=service.list_memory_candidates(session_id=session_id),
        )

    @app.get("/sessions/{session_id}/memory-context", response_model=MemoryContextResponse)
    def get_session_memory_context(
        session_id: str,
        query_text: str = Query(default=""),
        limit: int = Query(default=8, ge=1, le=50),
        include_pending: bool = Query(default=False),
        include_archived: bool = Query(default=False),
        prompt_category: PromptCategory | None = Query(default=None),
        activity_phase: str = Query(default="discussion"),
        prd_surface: PRDSurface | None = Query(default=None),
    ) -> MemoryContextResponse:
        try:
            memory_context = service.search_memory_context_for_session(
                session_id,
                query_text=query_text,
                limit=limit,
                include_pending=include_pending,
                include_archived=include_archived,
                prompt_category=prompt_category,
                activity_phase=activity_phase,
                prd_surface=prd_surface,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return MemoryContextResponse(session_id=session_id, memory_context=memory_context)

    @app.post("/sessions/{session_id}/memory-snapshot", response_model=MemorySnapshotResponse)
    def write_session_memory_snapshot(
        session_id: str,
        request: MemorySnapshotRequest | None = None,
    ) -> MemorySnapshotResponse:
        resolved_request = request or MemorySnapshotRequest()
        try:
            result = service.write_meeting_state_memory_snapshot(
                session_id,
                commit=resolved_request.commit,
                include_gaps=resolved_request.include_gaps,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return MemorySnapshotResponse(
            session_id=result.session.session_id,
            memory_candidates=result.memory_candidates,
            memories=result.memories,
            memory_upserts=result.memory_upserts,
            committed=result.committed,
        )

    @app.post("/sessions/{session_id}/memory-extraction", response_model=ProductMemoryExtractionResult)
    def extract_session_memories(
        session_id: str,
        request: MemoryExtractionRequestBody | None = None,
    ) -> ProductMemoryExtractionResult:
        resolved_request = request or MemoryExtractionRequestBody()
        try:
            return service.extract_session_memories(
                session_id,
                commit=resolved_request.commit,
                max_segments=resolved_request.max_segments,
                max_candidates=resolved_request.max_candidates,
                include_meeting_state=resolved_request.include_meeting_state,
                memory_context=resolved_request.memory_context,
                model=resolved_request.model,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelGatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 503 if "not configured" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @app.post("/memories/{memory_id}/confirm", response_model=MemoryMutationResponse)
    def confirm_memory(memory_id: str) -> MemoryMutationResponse:
        try:
            return MemoryMutationResponse(memory=service.confirm_memory(memory_id))
        except MemoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/memories/{memory_id}/reject", response_model=MemoryMutationResponse)
    def reject_memory(memory_id: str, request: MemoryMutationRequest | None = None) -> MemoryMutationResponse:
        try:
            return MemoryMutationResponse(memory=service.reject_memory(memory_id, reason=(request or MemoryMutationRequest()).reason))
        except MemoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/memories/{memory_id}/archive", response_model=MemoryMutationResponse)
    def archive_memory(memory_id: str, request: MemoryMutationRequest | None = None) -> MemoryMutationResponse:
        try:
            return MemoryMutationResponse(memory=service.archive_memory(memory_id, reason=(request or MemoryMutationRequest()).reason))
        except MemoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/memories/{memory_id}", response_model=MemoryForgetResponse)
    def forget_memory(memory_id: str, reason: str = Query(default="")) -> MemoryForgetResponse:
        try:
            return MemoryForgetResponse(memory_forget=service.forget_memory(memory_id, reason=reason))
        except MemoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/memories/{memory_id}/promote", response_model=MemoryPromotionResponse)
    def promote_memory(memory_id: str, request: MemoryPromotionRequest | None = None) -> MemoryPromotionResponse:
        resolved_request = request or MemoryPromotionRequest()
        try:
            result = service.promote_memory(
                memory_id,
                target_scope=resolved_request.target_scope,
                target_memory_type=resolved_request.target_memory_type,
                reason=resolved_request.reason,
                approved=resolved_request.approved,
            )
        except MemoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return MemoryPromotionResponse(memory_promotion=result)

    @app.get("/memories/{memory_id}/pending-updates", response_model=MemoryPendingUpdateListResponse)
    def list_memory_pending_updates(
        memory_id: str,
        status: MemoryPendingUpdateStatus | None = Query(default=MemoryPendingUpdateStatus.PENDING),
    ) -> MemoryPendingUpdateListResponse:
        try:
            service.memory.store.get_memory(memory_id)
        except MemoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return MemoryPendingUpdateListResponse(
            memory_id=memory_id,
            status=status,
            pending_updates=service.list_pending_memory_updates(memory_id=memory_id, status=status),
        )

    @app.post("/memory-updates/{update_id}/apply", response_model=MemoryPendingUpdateApplyResponse)
    def apply_memory_pending_update(
        update_id: str,
        request: MemoryMutationRequest | None = None,
    ) -> MemoryPendingUpdateApplyResponse:
        try:
            result = service.apply_pending_memory_update(update_id, reason=(request or MemoryMutationRequest()).reason)
        except MemoryPendingUpdateNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return MemoryPendingUpdateApplyResponse(memory_upsert=result)

    @app.post("/memory-updates/{update_id}/reject", response_model=MemoryPendingUpdateResponse)
    def reject_memory_pending_update(
        update_id: str,
        request: MemoryMutationRequest | None = None,
    ) -> MemoryPendingUpdateResponse:
        try:
            pending_update = service.reject_pending_memory_update(update_id, reason=(request or MemoryMutationRequest()).reason)
        except MemoryPendingUpdateNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return MemoryPendingUpdateResponse(pending_update=pending_update)

    return app


def create_default_product_service(settings: ModelGatewaySettings | None = None) -> ProductAssistantService:
    resolved_settings = settings or ModelGatewaySettings()
    model_client = _build_model_client(resolved_settings)
    prompt_service = PromptGenerationService(
        model_client=model_client,
        settings=resolved_settings,
    )
    memory_extraction_service = MemoryExtractionService(
        model_client=model_client,
        settings=resolved_settings,
    )
    session_store, runtime_service, memory_service = _build_storage_services()
    return ProductAssistantService(
        session_service=SessionService(session_store),
        prompt_orchestrator=PromptOrchestrator(prompt_service=prompt_service),
        runtime_service=runtime_service,
        memory_service=memory_service,
        memory_extraction_service=memory_extraction_service,
    )


def _build_storage_services() -> tuple[InMemorySessionStore | SQLiteSessionStore, PromptRuntimeService, MemoryService | None]:
    backend = os.getenv("PROACTIVE_STORAGE_BACKEND", os.getenv("STORAGE_BACKEND", "memory")).strip().lower()
    if backend in {"", "memory", "in_memory", "in-memory"}:
        return InMemorySessionStore(), PromptRuntimeService(), None
    if backend == "sqlite":
        db_path = Path(os.getenv("PROACTIVE_SQLITE_PATH", "data/local/proactive.db"))
        return (
            SQLiteSessionStore(db_path),
            PromptRuntimeService(SQLiteRuntimeStore(db_path)),
            MemoryService(SQLiteMemoryStore(db_path)),
        )
    raise ValueError(f"unsupported storage backend: {backend}")


def _build_model_client(settings: ModelGatewaySettings) -> OpenAIResponsesClient | OpenAIChatCompletionsClient:
    if settings.model_api_style == "responses":
        return OpenAIResponsesClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.model_api_style == "chat_completions":
        return OpenAIChatCompletionsClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            response_format=settings.chat_response_format,
            max_tokens_param=settings.chat_max_tokens_param,
        )
    raise ValueError(f"unsupported model API style: {settings.model_api_style}")


app = create_app()
