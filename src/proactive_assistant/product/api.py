import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.asr import (
    AzureSpeechRestRecognizer,
    AzureSpeechSDKStreamingRecognizer,
    AzureSpeechSettings,
    SpeechRecognitionError,
    SpeechRecognitionService,
    SpeechTranscriptionResult,
    StreamingSpeechEvent,
    StreamingSpeechEventType,
    StreamingSpeechRecognitionService,
)
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
from proactive_assistant.memory.query_understanding import QueryUnderstandingService
from proactive_assistant.model_gateway import (
    ModelGatewayError,
    ModelGatewaySettings,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from proactive_assistant.detection import PromptOpportunityDetector, UnknownTermDetector
from proactive_assistant.detection.vocabulary import PersonalVocabularyService
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product.contracts import (
    ProductAudioTranscriptStepResult,
    ProductInlineMemoryCaptureResult,
    ProductMemoryExtractionResult,
    ProductPolicyBaselineResult,
    ProductPolicyEpisodeResult,
    ProductPolicyEvaluationResult,
    ProductPolicyTrainingExportResult,
    ProductPrivacyMetrics,
    ProductPromptPayload,
    ProductSessionLifecycleResult,
    ProductSessionStateResult,
    ProductSessionSummaryResult,
)
from proactive_assistant.product.service import ProductAssistantService
from proactive_assistant.prompting import (
    PRDSurface,
    PrivacyLevel,
    PromptCategory,
    PromptGenerationService,
    RuleBasedPromptGenerationService,
)
from proactive_assistant.persistence import SQLiteMemoryStore, SQLiteRuntimeStore, SQLiteSessionStore
from proactive_assistant.runtime import (
    FeedbackInputChannel,
    FeedbackPolarity,
    FeedbackSignalSource,
    FeedbackSignalType,
    MemoryCandidate,
    MemoryCandidateType,
    ProactiveDisplayStrategy,
    PromptRuntimeService,
    RewardObservation,
    RuntimeFeedbackEvent,
    FeedbackTarget,
)
from proactive_assistant.runtime.store import (
    DecisionRecordNotFoundError,
    RuntimeFeedbackEventAlreadyExistsError,
)
from proactive_assistant.sessions import (
    AssistantSession,
    InMemorySessionStore,
    SessionConfig,
    SessionSource,
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


class TranscriptListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    transcript: list[TranscriptSegmentRecord] = Field(default_factory=list)


class MeetingStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    meeting_state: MeetingState


class PromptListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    prompts: list[ProductPromptPayload]


class SessionStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_memory_context: bool = False
    memory_query_text: str = ""
    memory_limit: int = Field(default=8, ge=1, le=50)
    include_pending_memory: bool = False


class EndSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generate_summary: bool = False
    use_memory: bool = True
    memory_limit: int = Field(default=8, ge=1, le=50)
    include_pending_memory: bool = True


class GenerateSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_segments: int = Field(default=48, ge=1, le=200)
    use_memory: bool = True
    memory_limit: int = Field(default=8, ge=1, le=50)
    include_pending_memory: bool = True
    policy_version: str = "product_summary_v0"


class RecordFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_type: FeedbackSignalType
    event_id: str | None = None
    source: FeedbackSignalSource | None = None
    input_channel: FeedbackInputChannel | None = None
    target: FeedbackTarget | None = None
    polarity: FeedbackPolarity | None = None
    intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    display_strategy: ProactiveDisplayStrategy | None = None
    text: str = ""
    dwell_ms: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    helpfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    timing_fit: float | None = Field(default=None, ge=-1.0, le=1.0)
    content_fit: float | None = Field(default=None, ge=-1.0, le=1.0)
    granularity_fit: float | None = Field(default=None, ge=-1.0, le=1.0)
    display_fit: float | None = Field(default=None, ge=-1.0, le=1.0)
    task_progress_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    flow_break_score: float | None = Field(default=None, ge=0.0, le=1.0)
    redundancy_score: float | None = Field(default=None, ge=0.0, le=1.0)
    privacy_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    missed_opportunity_score: float | None = Field(default=None, ge=0.0, le=1.0)
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


class InlineMemoryCaptureRequest(BaseModel):
    """P2-2 request payload for explicit "记一下 / remember this" captures."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    text: str = Field(min_length=1)
    memory_type: MemoryCandidateType = MemoryCandidateType.USER_PREFERENCE
    source_segment_id: str = ""
    target_speaker_id: str = ""
    topic: str = ""
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class InlineMemoryCaptureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture: ProductInlineMemoryCaptureResult


class PrivacyMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: ProductPrivacyMetrics


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

    @app.post("/asr/transcribe", response_model=SpeechTranscriptionResult)
    async def transcribe_audio(
        request: Request,
        language: str | None = Query(default=None),
        content_type: str | None = Header(default=None),
    ) -> SpeechTranscriptionResult:
        try:
            return service.transcribe_audio(
                await request.body(),
                language=language,
                content_type=content_type,
            )
        except SpeechRecognitionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 503 if "not configured" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

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

    @app.get("/sessions/{session_id}/state", response_model=ProductSessionStateResult)
    def get_session_state(
        session_id: str,
        include_memory_context: bool = Query(default=False),
        memory_query_text: str = Query(default=""),
        memory_limit: int = Query(default=8, ge=1, le=50),
        include_pending_memory: bool = Query(default=False),
    ) -> ProductSessionStateResult:
        try:
            return service.get_session_state(
                session_id,
                include_memory_context=include_memory_context,
                memory_query_text=memory_query_text,
                memory_limit=memory_limit,
                include_pending_memory=include_pending_memory,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/pause", response_model=ProductSessionLifecycleResult)
    def pause_session(session_id: str) -> ProductSessionLifecycleResult:
        try:
            return service.pause_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/resume", response_model=ProductSessionLifecycleResult)
    def resume_session(session_id: str) -> ProductSessionLifecycleResult:
        try:
            return service.resume_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/end", response_model=ProductSessionLifecycleResult)
    def end_session(session_id: str, request: EndSessionRequest | None = None) -> ProductSessionLifecycleResult:
        resolved_request = request or EndSessionRequest()
        try:
            return service.end_session(
                session_id,
                generate_summary=resolved_request.generate_summary,
                use_memory=resolved_request.use_memory,
                memory_limit=resolved_request.memory_limit,
                include_pending_memory=resolved_request.include_pending_memory,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    @app.post("/sessions/{session_id}/audio-transcript", response_model=ProductAudioTranscriptStepResult)
    async def append_audio_transcript(
        session_id: str,
        request: Request,
        speaker: str = Query(default="unknown"),
        start_ms: int = Query(default=0, ge=0),
        end_ms: int = Query(gt=0),
        segment_id: str | None = Query(default=None),
        language: str | None = Query(default=None),
        content_type: str | None = Header(default=None),
        is_final: bool = Query(default=True),
        max_segments: int = Query(default=12, ge=1, le=100),
        use_memory: bool = Query(default=True),
        memory_limit: int = Query(default=8, ge=1, le=50),
        include_pending_memory: bool = Query(default=False),
        policy_version: str = Query(default="product_audio_flow_v0"),
    ) -> ProductAudioTranscriptStepResult:
        try:
            return service.append_audio_transcript_and_generate_prompts(
                session_id,
                await request.body(),
                speaker=speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                segment_id=segment_id,
                language=language,
                content_type=content_type,
                is_final=is_final,
                max_segments=max_segments,
                use_memory=use_memory,
                memory_limit=memory_limit,
                include_pending_memory=include_pending_memory,
                policy_version=policy_version,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SpeechRecognitionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 503 if "not configured" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @app.websocket("/sessions/{session_id}/asr/stream")
    async def stream_session_asr(
        websocket: WebSocket,
        session_id: str,
        speaker: str = Query(default="unknown"),
        language: str | None = Query(default=None),
        audio_format: str = Query(default="pcm16k"),
        max_segments: int = Query(default=12, ge=1, le=100),
        use_memory: bool = Query(default=True),
        memory_limit: int = Query(default=8, ge=1, le=50),
        include_pending_memory: bool = Query(default=False),
        policy_version: str = Query(default="product_streaming_asr_v0"),
    ) -> None:
        await websocket.accept()
        try:
            service.sessions.get_session(session_id)
        except SessionNotFoundError:
            await websocket.send_json({"type": "error", "detail": f"session not found: {session_id}"})
            await websocket.close(code=4404)
            return
        try:
            stream = service.start_streaming_transcription(language=language, audio_format=audio_format)
        except (SpeechRecognitionError, ValueError) as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await websocket.close(code=1011)
            return

        stop_event = asyncio.Event()
        stream_started_at = time.monotonic()
        state = {
            "last_final_end_ms": 0,
            "final_count": 0,
            "client_disconnected": False,
        }
        await websocket.send_json(
            {
                "type": "stream_opened",
                "session_id": session_id,
                "speaker": speaker,
                "language": language,
                "audio_format": audio_format,
            }
        )

        async def receive_audio() -> None:
            try:
                while not stop_event.is_set():
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        state["client_disconnected"] = True
                        break
                    audio = message.get("bytes")
                    if audio:
                        await asyncio.to_thread(stream.write_audio, audio)
                        continue
                    text = message.get("text")
                    if text and _stream_control_type(text) in {"stop", "end_audio", "close"}:
                        break
            except WebSocketDisconnect:
                state["client_disconnected"] = True
            finally:
                await asyncio.to_thread(stream.end_audio)

        async def send_events() -> None:
            while not stop_event.is_set():
                event = await asyncio.to_thread(stream.read_event, 0.1)
                if event is None:
                    continue
                payload = _streaming_event_payload(event)
                event_type = StreamingSpeechEventType(event.event_type)
                if event_type == StreamingSpeechEventType.FINAL_TRANSCRIPT and event.text.strip():
                    state["final_count"] = int(state["final_count"]) + 1
                    start_ms, end_ms = _stream_segment_timing(event, stream_started_at, int(state["last_final_end_ms"]))
                    state["last_final_end_ms"] = end_ms
                    segment_id = f"{session_id}_stream_{int(state['final_count']):04d}"
                    try:
                        transcript_step = service.append_transcript_and_generate_prompts(
                            session_id,
                            TranscriptSegmentInput(
                                speaker=speaker,
                                start_ms=start_ms,
                                end_ms=end_ms,
                                text=event.text,
                                asr_confidence=event.confidence if event.confidence > 0 else 1.0,
                                language=event.language or language,
                                is_final=True,
                                source=SessionSource.LIVE_ASR_FUTURE,
                                metadata={
                                    "asr_provider": str(event.metadata.get("provider", "streaming")),
                                    "asr_streaming": True,
                                },
                            ),
                            segment_id=segment_id,
                            max_segments=max_segments,
                            use_memory=use_memory,
                            memory_limit=memory_limit,
                            include_pending_memory=include_pending_memory,
                            policy_version=policy_version,
                        )
                    except Exception as exc:
                        payload["transcript_error"] = str(exc)
                    else:
                        payload["transcript_step"] = transcript_step.model_dump(mode="json")
                if not state["client_disconnected"]:
                    await websocket.send_json(payload)
                if event_type in {
                    StreamingSpeechEventType.SESSION_STOPPED,
                    StreamingSpeechEventType.CANCELED,
                    StreamingSpeechEventType.ERROR,
                }:
                    stop_event.set()
                    break

        receiver = asyncio.create_task(receive_audio())
        sender = asyncio.create_task(send_events())
        try:
            await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
            if receiver.done() and not sender.done():
                try:
                    await asyncio.wait_for(sender, timeout=5.0)
                except TimeoutError:
                    stop_event.set()
            else:
                stop_event.set()
        finally:
            stop_event.set()
            await asyncio.to_thread(stream.stop)
            for task in (receiver, sender):
                if not task.done():
                    task.cancel()
            if not state["client_disconnected"]:
                try:
                    await websocket.close()
                except RuntimeError:
                    pass

    @app.get("/sessions/{session_id}/transcript", response_model=TranscriptListResponse)
    def list_session_transcript(session_id: str) -> TranscriptListResponse:
        try:
            transcript = service.list_transcript(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return TranscriptListResponse(session_id=session_id, transcript=transcript)

    @app.post("/sessions/{session_id}/summary", response_model=ProductSessionSummaryResult)
    def generate_session_summary(
        session_id: str,
        request: GenerateSummaryRequest | None = None,
    ) -> ProductSessionSummaryResult:
        resolved_request = request or GenerateSummaryRequest()
        try:
            return service.generate_session_summary(
                session_id,
                max_segments=resolved_request.max_segments,
                use_memory=resolved_request.use_memory,
                memory_limit=resolved_request.memory_limit,
                include_pending_memory=resolved_request.include_pending_memory,
                policy_version=resolved_request.policy_version,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelGatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    @app.get("/sessions/{session_id}/policy-episode", response_model=ProductPolicyEpisodeResult)
    def get_policy_episode(session_id: str) -> ProductPolicyEpisodeResult:
        try:
            return service.get_policy_episode(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/policy-evaluation", response_model=ProductPolicyEvaluationResult)
    def get_policy_evaluation(session_id: str) -> ProductPolicyEvaluationResult:
        try:
            return service.evaluate_policy_episode(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/policy-baselines", response_model=ProductPolicyBaselineResult)
    def get_policy_baselines(session_id: str) -> ProductPolicyBaselineResult:
        try:
            return service.evaluate_policy_baselines(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/policy-export", response_model=ProductPolicyTrainingExportResult)
    def get_policy_export(session_id: str) -> ProductPolicyTrainingExportResult:
        try:
            return service.export_policy_training_examples(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/prompt-decisions/{decision_id}/feedback", response_model=FeedbackRecordedResponse)
    def record_feedback(decision_id: str, request: RecordFeedbackRequest) -> FeedbackRecordedResponse:
        try:
            result = service.record_feedback(
                decision_id,
                request.signal_type,
                event_id=request.event_id,
                source=request.source,
                input_channel=request.input_channel,
                target=request.target,
                polarity=request.polarity,
                intensity=request.intensity,
                display_strategy=request.display_strategy,
                text=request.text,
                dwell_ms=request.dwell_ms,
                latency_ms=request.latency_ms,
                helpfulness=request.helpfulness,
                timing_fit=request.timing_fit,
                content_fit=request.content_fit,
                granularity_fit=request.granularity_fit,
                display_fit=request.display_fit,
                task_progress_delta=request.task_progress_delta,
                flow_break_score=request.flow_break_score,
                redundancy_score=request.redundancy_score,
                privacy_risk_score=request.privacy_risk_score,
                missed_opportunity_score=request.missed_opportunity_score,
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

    @app.get("/sessions/{session_id}/warmup-memory", response_model=MemoryContextResponse)
    def get_session_warmup_memory(session_id: str) -> MemoryContextResponse:
        try:
            warmup_context = service.get_warmup_memory_context(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return MemoryContextResponse(session_id=session_id, memory_context=warmup_context)

    @app.post(
        "/sessions/{session_id}/memories/capture",
        response_model=InlineMemoryCaptureResponse,
    )
    def capture_inline_memory(
        session_id: str, request: InlineMemoryCaptureRequest
    ) -> InlineMemoryCaptureResponse:
        try:
            result = service.capture_inline_memory(
                session_id,
                request.text,
                memory_type=request.memory_type,
                source_segment_id=request.source_segment_id,
                target_speaker_id=request.target_speaker_id,
                topic=request.topic,
                privacy_level=request.privacy_level,
                confidence=request.confidence,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return InlineMemoryCaptureResponse(capture=result)

    @app.get(
        "/sessions/{session_id}/privacy-metrics",
        response_model=PrivacyMetricsResponse,
    )
    def get_session_privacy_metrics(session_id: str) -> PrivacyMetricsResponse:
        try:
            metrics = service.privacy_metrics(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return PrivacyMetricsResponse(metrics=metrics)

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
    query_understanding_service = _build_query_understanding_service(
        model_client=model_client,
        settings=resolved_settings,
    )
    session_store, runtime_service, memory_service = _build_storage_services(
        query_understanding=query_understanding_service,
    )
    speech_service = _build_speech_service()
    streaming_speech_service = _build_streaming_speech_service()
    return ProductAssistantService(
        session_service=SessionService(session_store),
        prompt_orchestrator=_build_prompt_orchestrator(
            prompt_service=prompt_service,
            model_client=model_client,
            settings=resolved_settings,
            memory_service=memory_service,
        ),
        runtime_service=runtime_service,
        memory_service=memory_service,
        memory_extraction_service=memory_extraction_service,
        speech_recognition_service=speech_service,
        streaming_speech_recognition_service=streaming_speech_service,
    )


def _build_prompt_orchestrator(
    *,
    prompt_service: PromptGenerationService,
    model_client: Any = None,
    settings: ModelGatewaySettings | None = None,
    memory_service: MemoryService | None = None,
) -> PromptOrchestrator:
    mode = os.getenv("PROACTIVE_PROMPT_MODE", "llm_with_rule_fallback").strip().lower()
    rule_service = RuleBasedPromptGenerationService()
    detector = _build_opportunity_detector(
        model_client=model_client,
        settings=settings,
        memory_service=memory_service,
    )
    glasses_timeout = (
        settings.glasses_prompt_timeout_seconds if settings is not None else None
    )
    if mode in {"rules", "rule", "rule_based", "rule-based"}:
        return PromptOrchestrator(
            prompt_service=rule_service,
            detector=detector,
            glasses_prompt_timeout_seconds=glasses_timeout,
        )
    if mode in {"llm", "model"}:
        return PromptOrchestrator(
            prompt_service=prompt_service,
            fallback_on_generation_failure=False,
            detector=detector,
            glasses_prompt_timeout_seconds=glasses_timeout,
        )
    if mode in {"llm_with_rule_fallback", "llm_with_rules", "fallback"}:
        return PromptOrchestrator(
            prompt_service=prompt_service,
            fallback_prompt_service=rule_service,
            fallback_on_generation_failure=True,
            detector=detector,
            glasses_prompt_timeout_seconds=glasses_timeout,
        )
    raise ValueError(f"unsupported prompt mode: {mode}")


def _build_opportunity_detector(
    *,
    model_client: Any,
    settings: ModelGatewaySettings | None,
    memory_service: MemoryService | None,
) -> PromptOpportunityDetector:
    """Assemble the realtime opportunity detector with optional LLM unknown-term arm.

    Controlled by ``PROACTIVE_UNKNOWN_TERM_DETECTOR`` env var:

    - ``on`` (default): LLM unknown-term detection is enabled. A
      ``PersonalVocabularyService`` is wired in when a memory service is
      available; otherwise the detector still runs without personalization.
    - ``off``: rule-based detection only; the LLM arm and vocabulary
      service are not constructed.

    The detector falls back to ``off`` semantics when no model client is
    available (e.g. tests calling this helper directly without settings).
    """

    raw = os.getenv("PROACTIVE_UNKNOWN_TERM_DETECTOR", "on").strip().lower()
    enabled = raw in {"on", "true", "1", "enable", "enabled", "auto"}
    if not enabled or model_client is None:
        return PromptOpportunityDetector()
    unknown_term_detector = UnknownTermDetector(
        model_client=model_client,
        settings=_settings_with_default_fast_model(settings),
    )
    vocabulary_service = (
        PersonalVocabularyService(memory_service)
        if memory_service is not None
        else None
    )
    return PromptOpportunityDetector(
        unknown_term_detector=unknown_term_detector,
        vocabulary_service=vocabulary_service,
    )


def _build_query_understanding_service(
    *,
    model_client: Any,
    settings: ModelGatewaySettings | None,
) -> QueryUnderstandingService | None:
    raw = os.getenv("PROACTIVE_QUERY_UNDERSTANDING", "on").strip().lower()
    enabled = raw in {"", "on", "true", "1", "enable", "enabled", "auto"}
    if not enabled or model_client is None:
        return None
    return QueryUnderstandingService(
        model_client=model_client,
        settings=_settings_with_default_fast_model(settings),
    )


def _settings_with_default_fast_model(settings: ModelGatewaySettings | None) -> ModelGatewaySettings | None:
    if settings is None or _has_explicit_fast_model_env():
        return settings
    return settings.model_copy(update={"fast_model": settings.default_model})


def _has_explicit_fast_model_env() -> bool:
    return bool(os.getenv("OPENAI_FAST_MODEL") or os.getenv("PROACTIVE_OPENAI_FAST_MODEL"))


def _build_storage_services(
    *,
    query_understanding: QueryUnderstandingService | None = None,
) -> tuple[InMemorySessionStore | SQLiteSessionStore, PromptRuntimeService, MemoryService]:
    backend = os.getenv("PROACTIVE_STORAGE_BACKEND", os.getenv("STORAGE_BACKEND", "memory")).strip().lower()
    if backend in {"", "memory", "in_memory", "in-memory"}:
        # The unknown-term LLM detector relies on a real memory service
        # so PersonalVocabularyService can pull session-explained terms.
        # Construct one even in in-memory mode so demo wiring works
        # without further env tweaks.
        from proactive_assistant.memory import InMemoryMemoryStore

        return (
            InMemorySessionStore(),
            PromptRuntimeService(),
            MemoryService(
                InMemoryMemoryStore(),
                query_understanding=query_understanding,
            ),
        )
    if backend == "sqlite":
        db_path = Path(os.getenv("PROACTIVE_SQLITE_PATH", "data/local/proactive.db"))
        return (
            SQLiteSessionStore(db_path),
            PromptRuntimeService(SQLiteRuntimeStore(db_path)),
            MemoryService(
                SQLiteMemoryStore(db_path),
                query_understanding=query_understanding,
            ),
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


def _build_speech_service(settings: AzureSpeechSettings | None = None) -> SpeechRecognitionService | None:
    resolved_settings = settings or AzureSpeechSettings()
    if not resolved_settings.is_configured:
        return None
    return SpeechRecognitionService(AzureSpeechRestRecognizer(resolved_settings))


def _build_streaming_speech_service(settings: AzureSpeechSettings | None = None) -> StreamingSpeechRecognitionService | None:
    resolved_settings = settings or AzureSpeechSettings()
    if not resolved_settings.is_configured:
        return None
    return StreamingSpeechRecognitionService(AzureSpeechSDKStreamingRecognizer(resolved_settings))


def _stream_control_type(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.strip().lower()
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("type", "")).strip().lower()


def _streaming_event_payload(event: StreamingSpeechEvent) -> dict[str, Any]:
    return {
        "type": str(event.event_type),
        "transcription": {
            "text": event.text,
            "language": event.language,
            "confidence": event.confidence,
            "offset_ms": event.offset_ms,
            "duration_ms": event.duration_ms,
            "reason": event.reason,
            "metadata": dict(event.metadata),
        },
        "raw_response": dict(event.raw_response),
    }


def _stream_segment_timing(event: StreamingSpeechEvent, stream_started_at: float, last_final_end_ms: int) -> tuple[int, int]:
    if event.offset_ms is not None:
        start_ms = event.offset_ms
    else:
        start_ms = last_final_end_ms
    if event.duration_ms is not None:
        end_ms = start_ms + max(1, event.duration_ms)
    else:
        elapsed_ms = int((time.monotonic() - stream_started_at) * 1000)
        end_ms = max(start_ms + 1, elapsed_ms)
    return start_ms, end_ms


app = create_app()
