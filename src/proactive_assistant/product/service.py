from hashlib import sha1
from typing import Any

from proactive_assistant.asr import (
    SpeechRecognitionService,
    SpeechTranscriptionResult,
    StreamingSpeechRecognitionService,
    StreamingSpeechSession,
)
from proactive_assistant.detection import (
    CandidateTimingAction,
    DetectionRuleMatch,
    PromptOpportunity,
    PromptPriority,
    opportunities_from_meeting_gaps,
)
from proactive_assistant.meeting_state import MeetingState, MeetingStateTracker
from proactive_assistant.memory import (
    ExplanationMemoryWriter,
    InMemoryMemoryStore,
    MemoryContext,
    MemoryExtractionRequest,
    MemoryExtractionService,
    MemoryForgetResult,
    MemoryPendingUpdate,
    MemoryPendingUpdateStatus,
    MemoryPromotionResult,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemoryType,
    MemoryUpsertResult,
    memory_candidates_from_meeting_state,
)
from proactive_assistant.memory.snapshot import (
    _candidate_from_action_item,
    _candidate_from_decision,
    _candidate_from_risk,
)
from proactive_assistant.orchestration import PromptCandidate, PromptCandidateStatus, PromptOrchestrationResult, PromptOrchestrator
from proactive_assistant.prompting import ContentGranularity, PrivacyLevel, PromptCategory, TranscriptWindowItem
from proactive_assistant.runtime import (
    FeedbackInputChannel,
    FeedbackPolarity,
    FeedbackSignalSource,
    FeedbackSignalType,
    FeedbackTarget,
    MemoryCandidate,
    MemoryCandidateType,
    MemoryWritePolicy,
    PromptDecisionDisplayStatus,
    PromptDecisionRecord,
    ProactiveDisplayStrategy,
    PromptRuntimeService,
    RewardObservation,
)
from proactive_assistant.sessions import (
    AssistantSession,
    SessionConfig,
    SessionContextSnapshot,
    SessionSource,
    SessionService,
    TranscriptSegmentInput,
    TranscriptSegmentRecord,
    TranscriptWindow,
)

from proactive_assistant.product.contracts import (
    ProductAudioTranscriptStepResult,
    ProductFeedbackResult,
    ProductInlineMemoryCaptureResult,
    ProductMemoryExtractionResult,
    ProductMemorySnapshotResult,
    ProductPolicyBaselineResult,
    ProductPolicyEvaluationResult,
    ProductPolicyEpisodeResult,
    ProductPolicyTrainingExportResult,
    ProductPrivacyMetrics,
    ProductPromptPayload,
    ProductSessionLifecycleResult,
    ProductSessionStateResult,
    ProductSessionSummaryResult,
    ProductTranscriptStepResult,
)
from proactive_assistant.schemas.scenario import ActivityPhase


class ProductAssistantService:
    """Product-facing orchestration over session, prompt generation, and runtime feedback."""

    def __init__(
        self,
        *,
        session_service: SessionService,
        prompt_orchestrator: PromptOrchestrator,
        runtime_service: PromptRuntimeService,
        meeting_state_tracker: MeetingStateTracker | None = None,
        memory_service: MemoryService | None = None,
        memory_extraction_service: MemoryExtractionService | None = None,
        speech_recognition_service: SpeechRecognitionService | None = None,
        streaming_speech_recognition_service: StreamingSpeechRecognitionService | None = None,
        explanation_writer: ExplanationMemoryWriter | None = None,
        auto_memory_snapshot: bool = True,
    ) -> None:
        self.sessions = session_service
        self.prompt_orchestrator = prompt_orchestrator
        self.runtime = runtime_service
        self.meeting_state_tracker = meeting_state_tracker or MeetingStateTracker()
        self.memory = memory_service or MemoryService(InMemoryMemoryStore())
        self.memory_extraction = memory_extraction_service
        self.speech = speech_recognition_service
        self.streaming_speech = streaming_speech_recognition_service
        # Default the explanation writer to one backed by the same memory
        # service so the demo flow closes the unknown-term loop without
        # extra wiring. Pass None or an alternative writer to opt out.
        self.explanation_writer = (
            explanation_writer
            if explanation_writer is not None
            else ExplanationMemoryWriter(self.memory)
        )
        self._meeting_states: dict[str, MeetingState] = {}
        # PR1 (auto-snapshot): when set, every transcript step that adds a
        # new commitment-worthy MeetingState item (action_item with owner /
        # deadline / next_step, decided Decision, new Risk) auto-runs the
        # memory snapshot path so the item lands in long-term memory the
        # moment it is uttered. Avoids "session ended without snapshot ->
        # cross-session recall returns empty".
        self._auto_memory_snapshot = auto_memory_snapshot
        # Track the per-session set of action_item/decision/risk ids that
        # have already been auto-snapshotted, so subsequent transcript
        # steps in the same session only snapshot deltas, not the whole
        # state every time.
        self._auto_snapshotted_ids: dict[str, set[str]] = {}
        # B1: session-warmup cross-session memory cache, populated on
        # create_session() and re-used by every transcript step so the
        # demo path consistently shows "I remember last week..." context.
        self._warmup_contexts: dict[str, MemoryContext] = {}

    def create_session(
        self,
        config: SessionConfig | None = None,
        *,
        session_id: str | None = None,
        start: bool = True,
        warmup_memory: bool = True,
        warmup_limit: int = 5,
    ) -> AssistantSession:
        session = self.sessions.create_session(config, session_id=session_id, start=start)
        self._meeting_states[session.session_id] = self._create_meeting_state_for_session(session)
        if warmup_memory:
            # Memory hiccups must not block session creation. A failed
            # warmup leaves an empty cached context and the realtime flow
            # falls through to normal per-segment retrieval.
            try:
                self._warmup_contexts[session.session_id] = self.warmup_session_memory(
                    session.session_id,
                    limit=warmup_limit,
                )
            except Exception:
                self._warmup_contexts[session.session_id] = MemoryContext()
        return session

    def warmup_session_memory(
        self,
        session_id: str,
        *,
        limit: int = 5,
        include_pending: bool = False,
    ) -> MemoryContext:
        """Retrieve cross-session memory context for the freshly created session.

        The warmup query is built from ``session.title`` plus the
        ``project / topic / participants / tags`` fields in
        ``session.metadata``. Only USER / ORG / GLOBAL scope memories are
        returned; SESSION-scope memories belong to other sessions and are
        not useful as warmup context.

        Returns an empty ``MemoryContext`` when the session metadata does
        not yield a useful query (e.g. brand-new user with no project tags).
        """

        session = self.sessions.get_session(session_id)
        org_id, user_id = _memory_identity(session)
        # P2-5: detect cold start = the user/org/global memory store has
        # no records visible to this user at all. We do this with a cheap
        # zero-text list (no keyword filter), capped at 1 record.
        any_user_records = self.memory.store.list_memories(
            MemoryQuery(
                org_id=org_id,
                user_id=user_id,
                session_id=session_id,
                scopes=[MemoryScope.USER, MemoryScope.ORG, MemoryScope.GLOBAL],
                limit=1,
            )
        )
        cold_start = not any_user_records
        query_text = _warmup_query_text_from_session(session)
        if not query_text:
            return MemoryContext(is_empty_cold_start=cold_start)
        context = self.memory.search_context(
            MemoryQuery(
                query_text=query_text,
                org_id=org_id,
                user_id=user_id,
                session_id=session_id,
                scopes=[MemoryScope.USER, MemoryScope.ORG, MemoryScope.GLOBAL],
                include_pending=include_pending,
                limit=limit,
                prompt_category=PromptCategory.PERSON_OR_FACT,
                activity_phase=ActivityPhase.PRE_ACTIVITY.value,
                privacy_constraints=list(session.privacy_constraints),
                prd_surface="app_prompt_tab",
            )
        )
        return context.model_copy(update={"is_empty_cold_start": cold_start})

    def get_warmup_memory_context(self, session_id: str) -> MemoryContext:
        """Return the cached warmup context for a session.

        Empty ``MemoryContext`` if the session was created with
        ``warmup_memory=False`` or warmup yielded no matches.
        """

        # touch existence so callers get a clear error on bad ids
        self.sessions.get_session(session_id)
        return self._warmup_contexts.get(session_id, MemoryContext())

    def pause_session(self, session_id: str) -> ProductSessionLifecycleResult:
        session = self.sessions.pause_session(session_id)
        return ProductSessionLifecycleResult(
            session=session,
            meeting_state=self._get_or_create_meeting_state(session_id),
        )

    def resume_session(self, session_id: str) -> ProductSessionLifecycleResult:
        session = self.sessions.resume_session(session_id)
        return ProductSessionLifecycleResult(
            session=session,
            meeting_state=self._get_or_create_meeting_state(session_id),
        )

    def end_session(
        self,
        session_id: str,
        *,
        generate_summary: bool = False,
        use_memory: bool = True,
        memory_limit: int = 8,
        include_pending_memory: bool = True,
    ) -> ProductSessionLifecycleResult:
        session = self.sessions.end_session(session_id)
        meeting_state = self._get_or_create_meeting_state(session_id)
        summary = (
            self.generate_session_summary(
                session_id,
                use_memory=use_memory,
                memory_limit=memory_limit,
                include_pending_memory=include_pending_memory,
            )
            if generate_summary
            else None
        )
        return ProductSessionLifecycleResult(
            session=session,
            meeting_state=meeting_state,
            summary=summary,
        )

    def append_transcript_and_generate_prompts(
        self,
        session_id: str,
        segment: TranscriptSegmentInput,
        *,
        segment_id: str | None = None,
        max_segments: int = 12,
        memory_context: list[str] | None = None,
        memory_refs: list[str] | None = None,
        use_memory: bool = True,
        memory_query_text: str | None = None,
        memory_limit: int = 8,
        include_pending_memory: bool = False,
        policy_version: str = "product_flow_v0",
        max_prompts: int = 1,
    ) -> ProductTranscriptStepResult:
        transcript_segment = self.sessions.append_transcript(session_id, segment, segment_id=segment_id)
        meeting_update = self.meeting_state_tracker.update_from_segment(
            self._get_or_create_meeting_state(session_id),
            transcript_segment,
        )
        self._meeting_states[session_id] = meeting_update.state
        retrieved_memory_context = (
            self.search_memory_context_for_session(
                session_id,
                query_text=memory_query_text if memory_query_text is not None else segment.text,
                limit=memory_limit,
                include_pending=include_pending_memory,
            )
            if use_memory
            else None
        )
        warmup_context = self._warmup_contexts.get(session_id)
        warmup_lines = list(warmup_context.memory_context) if warmup_context else []
        warmup_refs = list(warmup_context.memory_refs) if warmup_context else []
        base_memory_context = _merge_unique(
            memory_context or [],
            warmup_lines,
        )
        base_memory_context = _merge_unique(
            base_memory_context,
            retrieved_memory_context.memory_context if retrieved_memory_context is not None else [],
        )
        base_memory_refs = _merge_unique(
            memory_refs or [],
            warmup_refs,
        )
        base_memory_refs = _merge_unique(
            base_memory_refs,
            retrieved_memory_context.memory_refs if retrieved_memory_context is not None else [],
        )
        snapshot = self.sessions.get_context_snapshot(
            session_id,
            max_segments=max_segments,
            memory_context=base_memory_context,
            memory_refs=base_memory_refs,
        )
        state_opportunities = opportunities_from_meeting_gaps(
            session_id,
            _gaps_for_current_segment(meeting_update.gaps, transcript_segment.segment_id),
            fallback_segment_id=transcript_segment.segment_id,
            privacy_constraints=snapshot.privacy_constraints,
            speaker_by_segment_id={
                seg.segment_id: seg.speaker or ""
                for seg in snapshot.recent_transcript.segments
            },
        )
        opportunities = self.prompt_orchestrator.select_opportunities(
            snapshot,
            extra_opportunities=state_opportunities,
        )
        # A4: glasses shows a single prompt. Generate only the top-ranked
        # opportunity (already sorted by priority/category/confidence) so the
        # realtime path makes one generation call instead of up to N.
        if max_prompts > 0:
            opportunities = opportunities[:max_prompts]
        orchestration_result = self._run_memory_aware_orchestration(
            session_id,
            base_snapshot=snapshot,
            base_memory_context=base_memory_context,
            base_memory_refs=base_memory_refs,
            opportunities=opportunities,
            use_memory=use_memory,
            memory_limit=memory_limit,
            include_pending_memory=include_pending_memory,
        )
        decisions = [
            self.runtime.log_candidate(
                candidate,
                policy_version=policy_version,
                metadata=_decision_memory_metadata(candidate),
            )
            for candidate in orchestration_result.candidates
        ]
        # B2: close the unknown-term loop. Any candidate flagged by the
        # LLM detector with a pre-generated explanation gets written to
        # session-scope memory so the next detector pass sees it.
        self._commit_explanation_memories_for(
            session_id,
            orchestration_result.candidates,
            decisions,
        )
        # PR1: auto-snapshot new commitment-worthy MeetingState items into
        # long-term memory the moment they appear. Without this, action
        # items only land in memory when someone calls /memory-snapshot
        # manually -- meaning a session that ended without that call
        # produces no cross-session recall data.
        if self._auto_memory_snapshot:
            self._auto_snapshot_new_commitments(session_id, meeting_update.state)
        return ProductTranscriptStepResult(
            session=self.sessions.get_session(session_id),
            transcript_segment=transcript_segment,
            snapshot=snapshot,
            meeting_state=meeting_update.state,
            meeting_gaps=meeting_update.gaps,
            retrieved_memory_context=retrieved_memory_context,
            prompts=[prompt_payload_from_decision(decision) for decision in decisions],
            decisions=decisions,
            opportunity_count=len(orchestration_result.opportunities),
            candidate_count=len(orchestration_result.candidates),
        )

    def preview_transcript_prompts(
        self,
        session_id: str,
        segment: TranscriptSegmentInput,
        *,
        segment_id: str | None = None,
        max_segments: int = 12,
        memory_context: list[str] | None = None,
        memory_refs: list[str] | None = None,
        use_memory: bool = False,
        max_prompts: int = 1,
    ) -> ProductTranscriptStepResult:
        """Generate provisional prompts from an ASR partial without persistence."""

        session = self.sessions.get_session(session_id)
        transcript_segment = TranscriptSegmentRecord(
            **segment.model_dump(mode="python"),
            session_id=session_id,
            segment_id=segment_id or f"preview_{sha1(segment.text.encode('utf-8')).hexdigest()[:12]}",
        )
        stored_transcript = self.sessions.get_transcript(session_id)
        recent_segments = [*stored_transcript[-max(max_segments - 1, 0):], transcript_segment]
        warmup_context = self._warmup_contexts.get(session_id)
        base_memory_context = _merge_unique(
            memory_context or [],
            list(warmup_context.memory_context) if warmup_context and use_memory else [],
        )
        base_memory_refs = _merge_unique(
            memory_refs or [],
            list(warmup_context.memory_refs) if warmup_context and use_memory else [],
        )
        snapshot = SessionContextSnapshot(
            session_id=session.session_id,
            scene=session.scene,
            status=session.status,
            locale=session.locale,
            pre_context=session.pre_context,
            recent_transcript=TranscriptWindow.from_segments(session_id, recent_segments[-max_segments:]),
            transcript_stats={
                "segment_count": len(stored_transcript) + 1,
                "total_chars": sum(len(item.text) for item in stored_transcript) + len(transcript_segment.text),
                "duration_ms": _transcript_duration_ms(recent_segments),
                "speaker_count": len({item.speaker for item in recent_segments}),
                "provisional": True,
            },
            privacy_constraints=session.privacy_constraints,
            memory_context=base_memory_context,
            memory_refs=base_memory_refs,
            metadata={**session.metadata, "provisional": True},
        )
        opportunities = self.prompt_orchestrator.select_opportunities(snapshot)
        if max_prompts > 0:
            opportunities = opportunities[:max_prompts]
        candidates = [
            self.prompt_orchestrator.generate_candidate(snapshot, opportunity)
            for opportunity in opportunities
        ]
        return ProductTranscriptStepResult(
            session=session,
            transcript_segment=transcript_segment,
            snapshot=snapshot,
            meeting_state=self._get_or_create_meeting_state(session_id),
            meeting_gaps=[],
            retrieved_memory_context=None,
            prompts=[
                prompt_payload_from_candidate(
                    candidate,
                    decision_id=f"preview_{candidate.candidate_id}",
                )
                for candidate in candidates
            ],
            decisions=[],
            opportunity_count=len(opportunities),
            candidate_count=len(candidates),
        )

    def _auto_snapshot_new_commitments(self, session_id: str, state: MeetingState) -> None:
        """PR1: snapshot MeetingState items that appeared in this step.

        Strategy: only write items that
          (a) have not been snapshotted before in this session, AND
          (b) carry enough commitment to be worth persisting:
              - action_item with owner OR deadline OR next_step
              - decision with a recorded conclusion
              - any risk (risks are worth tracking even when open)

        Phase 2 additions:
          - Before writing, normalize the item entity. If a memory record
            with the same normalized_entity already exists in this session,
            reuse its memory_candidate_id so MemoryService.upsert_candidate
            updates the existing record rather than creating a sibling.
            This is how "same task mentioned in later conversation" merges
            details and time-window updates into one memory record.

        Snapshot failures are swallowed so the realtime transcript path
        never breaks.
        """

        seen = self._auto_snapshotted_ids.setdefault(session_id, set())
        new_action_items = [
            item for item in state.action_items
            if item.action_item_id not in seen
            and bool(item.owner or item.deadline or getattr(item, "next_step", ""))
        ]
        new_decisions = [
            item for item in state.decisions
            if item.decision_id not in seen and item.conclusion
        ]
        new_risks = [item for item in state.risks if item.risk_id not in seen]
        if not (new_action_items or new_decisions or new_risks):
            return

        org_id = state.org_id
        user_id = state.subject_user_id
        for action_item in new_action_items:
            try:
                candidate = _candidate_from_action_item(state, action_item)
                candidate = self._rebind_to_existing_entity(candidate, session_id=session_id)
                self.memory.upsert_candidate(candidate, org_id=org_id, user_id=user_id)
                seen.add(action_item.action_item_id)
            except Exception:
                continue
        for decision in new_decisions:
            try:
                candidate = _candidate_from_decision(state, decision)
                candidate = self._rebind_to_existing_entity(candidate, session_id=session_id)
                self.memory.upsert_candidate(candidate, org_id=org_id, user_id=user_id)
                seen.add(decision.decision_id)
            except Exception:
                continue
        for risk in new_risks:
            try:
                candidate = _candidate_from_risk(state, risk)
                candidate = self._rebind_to_existing_entity(candidate, session_id=session_id)
                self.memory.upsert_candidate(candidate, org_id=org_id, user_id=user_id)
                seen.add(risk.risk_id)
            except Exception:
                continue

    def _rebind_to_existing_entity(
        self,
        candidate: MemoryCandidate,
        *,
        session_id: str,
    ) -> MemoryCandidate:
        """Phase 2 entity-based dedup.

        Look up existing memories in this session that share the same
        ``normalized_entity`` as the incoming candidate. If found, return
        a candidate whose ``memory_candidate_id`` is derived from the same
        decision_id+entity tuple that produced the existing memory, so
        ``upsert_candidate`` runs the merge path (updates the same record)
        rather than the create path.

        If no existing entity matches, return the candidate unchanged.
        """

        candidate_meta = candidate.metadata or {}
        target_entity = (
            candidate_meta.get("normalized_entity")
            or candidate_meta.get("canonical_entity")
            or candidate_meta.get("entity")
        )
        if not target_entity:
            return candidate
        normalized_target = str(target_entity).strip().lower()
        if not normalized_target:
            return candidate
        existing_records = self.memory.store.list_memories(
            MemoryQuery(session_id=session_id, limit=50)
        )
        for record in existing_records:
            record_entity = (
                record.metadata.get("normalized_entity")
                or record.metadata.get("canonical_entity")
                or record.metadata.get("entity")
            )
            if not record_entity:
                continue
            if str(record_entity).strip().lower() != normalized_target:
                continue
            # Found a matching entity. Reuse the stored candidate_id so
            # upsert_candidate triggers the merge path on the same record.
            stored_candidate_id = record.metadata.get("candidate_id")
            stored_decision_id = record.metadata.get("decision_id")
            if not stored_candidate_id or not stored_decision_id:
                continue
            return candidate.model_copy(
                update={
                    "memory_candidate_id": stored_candidate_id,
                    "decision_id": stored_decision_id,
                }
            )
        return candidate

    def _commit_explanation_memories_for(
        self,
        session_id: str,
        candidates: list[Any],
        decisions: list[PromptDecisionRecord],
    ) -> None:
        """Persist LLM-detected unknown-term explanations into memory.

        Memory write failures are silenced: the realtime prompt path must
        not be broken by a downstream memory hiccup. The writer itself is
        idempotent on ``(session_id, normalized_term)``, so the same term
        appearing in multiple turns within a session yields exactly one
        memory record.
        """

        if self.explanation_writer is None or not candidates:
            return
        try:
            session = self.sessions.get_session(session_id)
        except Exception:
            return
        org_id, user_id = _memory_identity(session)
        for candidate, decision in zip(candidates, decisions):
            opportunity = candidate.opportunity
            opportunity_meta = (opportunity.metadata or {}) if opportunity is not None else {}
            # Detector marks the opportunity, not the candidate.
            if opportunity_meta.get("detection_source") != "llm_unknown_term_detector":
                continue
            status_value = (
                candidate.status if isinstance(candidate.status, str) else candidate.status.value
            )
            if status_value != "generated":
                continue
            result = candidate.prompt_result
            if result is None or not result.should_prompt:
                continue
            term = str(opportunity_meta.get("unknown_term") or "").strip()
            explanation = str(opportunity_meta.get("pre_generated_explanation") or "").strip()
            if not term or not explanation:
                continue
            source_segment_id = (
                opportunity.trigger_segment_ids[0]
                if opportunity is not None and opportunity.trigger_segment_ids
                else ""
            )
            try:
                self.explanation_writer.commit_explanation(
                    session_id=session_id,
                    org_id=org_id,
                    user_id=user_id,
                    term=term,
                    explanation=explanation,
                    term_type=str(opportunity_meta.get("unknown_term_type") or "acronym"),
                    source_segment_id=source_segment_id,
                    detector_candidate_id=str(opportunity_meta.get("unknown_term_candidate_id") or ""),
                    decision_id=decision.decision_id,
                    confidence=float(result.confidence),
                    privacy_level=result.privacy_level,
                    target_speaker_id=getattr(opportunity, "target_speaker_id", "") or "",
                )
            except Exception:
                continue

    def transcribe_audio(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        content_type: str | None = None,
    ) -> SpeechTranscriptionResult:
        if self.speech is None:
            raise ValueError("speech recognition service is not configured")
        return self.speech.transcribe(audio, language=language, content_type=content_type)

    def append_audio_transcript_and_generate_prompts(
        self,
        session_id: str,
        audio: bytes,
        *,
        speaker: str,
        start_ms: int,
        end_ms: int,
        segment_id: str | None = None,
        language: str | None = None,
        content_type: str | None = None,
        is_final: bool = True,
        max_segments: int = 12,
        use_memory: bool = True,
        memory_limit: int = 8,
        include_pending_memory: bool = False,
        policy_version: str = "product_audio_flow_v0",
    ) -> ProductAudioTranscriptStepResult:
        transcription = self.transcribe_audio(audio, language=language, content_type=content_type)
        transcript_step = self.append_transcript_and_generate_prompts(
            session_id,
            TranscriptSegmentInput(
                speaker=speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                text=transcription.text,
                asr_confidence=transcription.confidence if transcription.confidence > 0 else 1.0,
                language=transcription.language,
                is_final=is_final,
                source=SessionSource.UPLOADED_AUDIO_TRANSCRIPT,
                metadata={
                    "asr_provider": transcription.provider,
                    "asr_duration_ms": transcription.duration_ms,
                },
            ),
            segment_id=segment_id,
            max_segments=max_segments,
            use_memory=use_memory,
            memory_limit=memory_limit,
            include_pending_memory=include_pending_memory,
            policy_version=policy_version,
        )
        return ProductAudioTranscriptStepResult(transcription=transcription, transcript_step=transcript_step)

    def start_streaming_transcription(
        self,
        *,
        language: str | None = None,
        audio_format: str = "pcm16k",
    ) -> StreamingSpeechSession:
        if self.streaming_speech is None:
            raise ValueError("streaming speech recognition service is not configured")
        return self.streaming_speech.start_stream(language=language, audio_format=audio_format)

    def run_prompt_flow_for_snapshot(
        self,
        session_id: str,
        *,
        max_segments: int = 12,
        memory_context: list[str] | None = None,
        memory_refs: list[str] | None = None,
        policy_version: str = "product_flow_v0",
    ) -> PromptOrchestrationResult:
        snapshot = self.sessions.get_context_snapshot(
            session_id,
            max_segments=max_segments,
            memory_context=memory_context,
            memory_refs=memory_refs,
        )
        result = self.prompt_orchestrator.run(snapshot)
        self.runtime.log_orchestration_result(result, policy_version=policy_version)
        return result

    def _run_memory_aware_orchestration(
        self,
        session_id: str,
        *,
        base_snapshot: Any,
        base_memory_context: list[str],
        base_memory_refs: list[str],
        opportunities: list[Any],
        use_memory: bool,
        memory_limit: int,
        include_pending_memory: bool,
    ) -> PromptOrchestrationResult:
        candidates = []
        snapshot_for_result = base_snapshot
        max_segments = max(len(base_snapshot.recent_transcript.segments), 1)
        for opportunity in opportunities:
            should_search_memory = use_memory and _opportunity_needs_realtime_memory(opportunity)
            opportunity_memory = (
                self.search_memory_context_for_session(
                    session_id,
                    query_text=opportunity.captured_text,
                    limit=memory_limit,
                    include_pending=include_pending_memory,
                    prompt_category=str(_enum_value(opportunity.prompt_category)),
                    activity_phase=str(_enum_value(opportunity.activity_phase)),
                    prd_surface=_prd_surface_for_opportunity(opportunity),
                )
                if should_search_memory
                else None
            )
            memory_context = _merge_unique(
                base_memory_context,
                opportunity_memory.memory_context if opportunity_memory is not None else [],
            )
            memory_refs = _merge_unique(
                base_memory_refs,
                opportunity_memory.memory_refs if opportunity_memory is not None else [],
            )
            snapshot = self.sessions.get_context_snapshot(
                session_id,
                max_segments=max_segments,
                memory_context=memory_context,
                memory_refs=memory_refs,
            )
            snapshot_for_result = snapshot
            candidate = self.prompt_orchestrator.generate_candidate(snapshot, opportunity)
            candidate = candidate.model_copy(
                update={
                    "metadata": {
                        **candidate.metadata,
                        "memory_context": list(memory_context),
                        "memory_refs": list(memory_refs),
                        "retrieved_memory_refs": list(opportunity_memory.memory_refs)
                        if opportunity_memory is not None
                        else [],
                        "retrieved_memory_result_count": len(opportunity_memory.results)
                        if opportunity_memory is not None
                        else 0,
                        "memory_query_text": opportunity.captured_text,
                        "memory_query_prompt_category": str(_enum_value(opportunity.prompt_category)),
                        "memory_query_activity_phase": str(_enum_value(opportunity.activity_phase)),
                        "memory_lookup_skipped": use_memory and not should_search_memory,
                    }
                }
            )
            candidates.append(candidate)
        return PromptOrchestrationResult(
            session_id=session_id,
            snapshot=snapshot_for_result,
            opportunities=opportunities,
            candidates=candidates,
        )

    def get_meeting_state(self, session_id: str) -> MeetingState:
        return self._get_or_create_meeting_state(session_id)

    def get_session_state(
        self,
        session_id: str,
        *,
        include_memory_context: bool = False,
        memory_query_text: str = "",
        memory_limit: int = 8,
        include_pending_memory: bool = False,
    ) -> ProductSessionStateResult:
        session = self.sessions.get_session(session_id)
        memory_context = (
            self.search_memory_context_for_session(
                session_id,
                query_text=memory_query_text,
                limit=memory_limit,
                include_pending=include_pending_memory,
            )
            if include_memory_context
            else None
        )
        return ProductSessionStateResult(
            session=session,
            meeting_state=self._get_or_create_meeting_state(session_id),
            transcript=self.sessions.get_transcript(session_id),
            prompts=self.list_prompt_payloads(session_id=session_id),
            memory_context=memory_context,
        )

    def list_transcript(self, session_id: str) -> list[TranscriptSegmentRecord]:
        self.sessions.get_session(session_id)
        return self.sessions.get_transcript(session_id)

    def list_prompt_decisions(self, *, session_id: str | None = None) -> list[PromptDecisionRecord]:
        return self.runtime.store.list_decisions(session_id=session_id)

    def list_prompt_payloads(self, *, session_id: str | None = None) -> list[ProductPromptPayload]:
        return [prompt_payload_from_decision(decision) for decision in self.list_prompt_decisions(session_id=session_id)]

    def generate_session_summary(
        self,
        session_id: str,
        *,
        max_segments: int = 48,
        use_memory: bool = True,
        memory_limit: int = 8,
        include_pending_memory: bool = True,
        policy_version: str = "product_summary_v0",
    ) -> ProductSessionSummaryResult:
        session = self.sessions.get_session(session_id)
        transcript = self.sessions.get_transcript(session_id)
        if not transcript:
            raise ValueError("cannot generate session summary without transcript segments")
        state = self._get_or_create_meeting_state(session_id)
        current_ms = max(segment.end_ms for segment in transcript)
        gaps = self.meeting_state_tracker.scan_state_gaps(state, current_ms=current_ms, force_end_summary=True)
        retrieved_memory_context = (
            self.search_memory_context_for_session(
                session_id,
                query_text=_summary_memory_query(state),
                limit=memory_limit,
                include_pending=include_pending_memory,
                prompt_category=PromptCategory.SUMMARY_GAP_CHECK,
                activity_phase=ActivityPhase.POST_ACTIVITY,
                prd_surface="app_summary_tab",
            )
            if use_memory
            else None
        )
        base_memory_context = retrieved_memory_context.memory_context if retrieved_memory_context is not None else []
        base_memory_refs = retrieved_memory_context.memory_refs if retrieved_memory_context is not None else []
        snapshot = self.sessions.get_context_snapshot(
            session_id,
            max_segments=max_segments,
            memory_context=base_memory_context,
            memory_refs=base_memory_refs,
        )
        opportunity = _summary_opportunity(
            session_id,
            state=state,
            gaps=gaps,
            transcript_segment_ids=[segment.segment_id for segment in snapshot.recent_transcript.segments],
            sequence=len(self.list_prompt_decisions(session_id=session_id)),
        )
        orchestration_result = self._run_memory_aware_orchestration(
            session_id,
            base_snapshot=snapshot,
            base_memory_context=base_memory_context,
            base_memory_refs=base_memory_refs,
            opportunities=[opportunity],
            use_memory=use_memory,
            memory_limit=memory_limit,
            include_pending_memory=include_pending_memory,
        )
        decisions = [
            self.runtime.log_candidate(
                candidate,
                policy_version=policy_version,
                metadata={
                    **_decision_memory_metadata(candidate),
                    "summary_source": "session_summary_v0",
                    "summary_gap_count": len(gaps),
                    "summary_transcript_segment_count": len(snapshot.recent_transcript.segments),
                },
            )
            for candidate in orchestration_result.candidates
        ]
        return ProductSessionSummaryResult(
            session=session,
            meeting_state=state,
            meeting_gaps=gaps,
            retrieved_memory_context=retrieved_memory_context,
            prompts=[prompt_payload_from_decision(decision) for decision in decisions],
            decisions=decisions,
        )

    def record_feedback(
        self,
        decision_id: str,
        signal_type: FeedbackSignalType | str,
        *,
        event_id: str | None = None,
        source: FeedbackSignalSource | str | None = None,
        input_channel: FeedbackInputChannel | str | None = None,
        target: FeedbackTarget | str | None = None,
        polarity: FeedbackPolarity | str | None = None,
        intensity: float | None = None,
        display_strategy: ProactiveDisplayStrategy | str | None = None,
        text: str = "",
        dwell_ms: int | None = None,
        latency_ms: int | None = None,
        helpfulness: float | None = None,
        timing_fit: float | None = None,
        content_fit: float | None = None,
        granularity_fit: float | None = None,
        display_fit: float | None = None,
        task_progress_delta: float | None = None,
        flow_break_score: float | None = None,
        redundancy_score: float | None = None,
        privacy_risk_score: float | None = None,
        missed_opportunity_score: float | None = None,
        metadata: dict[str, Any] | None = None,
        compute_reward: bool = True,
        propose_memory: bool = True,
        commit_memory: bool = True,
    ) -> ProductFeedbackResult:
        event = self.runtime.record_feedback(
            decision_id,
            signal_type,
            event_id=event_id,
            source=source,
            input_channel=input_channel,
            target=target,
            polarity=polarity,
            intensity=intensity,
            display_strategy=display_strategy,
            text=text,
            dwell_ms=dwell_ms,
            latency_ms=latency_ms,
            helpfulness=helpfulness,
            timing_fit=timing_fit,
            content_fit=content_fit,
            granularity_fit=granularity_fit,
            display_fit=display_fit,
            task_progress_delta=task_progress_delta,
            flow_break_score=flow_break_score,
            redundancy_score=redundancy_score,
            privacy_risk_score=privacy_risk_score,
            missed_opportunity_score=missed_opportunity_score,
            metadata=metadata,
        )
        reward = self.runtime.compute_reward_observation(decision_id) if compute_reward else None
        memory_candidates = self._persist_memory_candidates_for_event(decision_id, event.event_id) if propose_memory else []
        memories = self._commit_memory_candidates(memory_candidates) if commit_memory else []
        return ProductFeedbackResult(
            decision=self.runtime.store.get_decision(decision_id),
            feedback_event=event,
            reward_observation=reward,
            memory_candidates=memory_candidates,
            memories=memories,
        )

    def compute_reward(self, decision_id: str) -> RewardObservation:
        return self.runtime.compute_reward_observation(decision_id)

    def get_policy_episode(self, session_id: str) -> ProductPolicyEpisodeResult:
        session = self.sessions.get_session(session_id)
        return ProductPolicyEpisodeResult(session=session, episode=self.runtime.build_policy_episode(session_id))

    def evaluate_policy_episode(self, session_id: str) -> ProductPolicyEvaluationResult:
        session = self.sessions.get_session(session_id)
        return ProductPolicyEvaluationResult(session=session, report=self.runtime.evaluate_policy_episode(session_id))

    def evaluate_policy_baselines(self, session_id: str) -> ProductPolicyBaselineResult:
        session = self.sessions.get_session(session_id)
        return ProductPolicyBaselineResult(session=session, report=self.runtime.evaluate_policy_baselines(session_id))

    def export_policy_training_examples(self, session_id: str) -> ProductPolicyTrainingExportResult:
        session = self.sessions.get_session(session_id)
        return ProductPolicyTrainingExportResult(
            session=session,
            export=self.runtime.export_policy_training_examples(session_id),
        )

    def list_memory_candidates(
        self,
        *,
        decision_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryCandidate]:
        return self.runtime.store.list_memory_candidates(decision_id=decision_id, session_id=session_id)

    def search_memory_context_for_session(
        self,
        session_id: str,
        *,
        query_text: str = "",
        limit: int = 8,
        include_pending: bool = False,
        include_archived: bool = False,
        prompt_category: str | None = None,
        activity_phase: str = "discussion",
        prd_surface: str | None = None,
        active_entities: list[dict[str, Any]] | None = None,
    ) -> MemoryContext:
        session = self.sessions.get_session(session_id)
        org_id, user_id = _memory_identity(session)
        snapshot = self.sessions.get_context_snapshot(session_id, max_segments=8)
        state = self._get_or_create_meeting_state(session_id)
        return self.memory.search_context(
            MemoryQuery(
                query_text=query_text,
                org_id=org_id,
                user_id=user_id,
                session_id=session_id,
                include_pending=include_pending,
                include_archived=include_archived,
                limit=limit,
                prompt_category=prompt_category,
                activity_phase=activity_phase,
                recent_transcript_text="\n".join(segment.text for segment in snapshot.recent_transcript.segments),
                active_entities=active_entities or _active_entities_from_meeting_state(state),
                privacy_constraints=session.privacy_constraints,
                prd_surface=prd_surface,
            )
        )

    def confirm_memory(self, memory_id: str) -> MemoryRecord:
        return self.memory.confirm_memory(memory_id)

    def reject_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        return self.memory.reject_memory(memory_id, reason=reason)

    def capture_inline_memory(
        self,
        session_id: str,
        text: str,
        *,
        memory_type: MemoryCandidateType | str = MemoryCandidateType.USER_PREFERENCE,
        source_segment_id: str = "",
        target_speaker_id: str = "",
        topic: str = "",
        privacy_level: PrivacyLevel | str = PrivacyLevel.LOW,
        confidence: float = 0.85,
        metadata: dict[str, Any] | None = None,
    ) -> ProductInlineMemoryCaptureResult:
        """P2-2: explicit user-driven "记一下 / remember this" capture.

        Bypasses the feedback path and writes a MemoryCandidate straight to
        long-term memory via the upsert pipeline. Idempotent on the same
        (session, normalized_text, memory_type) tuple so re-clicking the
        UI save button does not create duplicates.
        """

        if not text.strip():
            raise ValueError("capture text must not be empty")
        session = self.sessions.get_session(session_id)
        org_id, user_id = _memory_identity(session)
        resolved_type = MemoryCandidateType(memory_type)
        resolved_privacy = PrivacyLevel(privacy_level) if not isinstance(privacy_level, PrivacyLevel) else privacy_level
        candidate_id = _inline_capture_candidate_id(session_id, text, resolved_type)
        decision_id = f"inline_capture:{session_id}"
        candidate_metadata: dict[str, Any] = {
            "org_id": org_id,
            "subject_user_id": user_id,
            "capture_source": "user_explicit_inline",
            "target_speaker_id": target_speaker_id,
            "source_capture_ref": f"transcript:{source_segment_id}" if source_segment_id else "",
            "topic": topic,
            "tags": ["inline_capture", "user_capture"],
        }
        if metadata:
            candidate_metadata["user_metadata"] = dict(metadata)
        candidate = MemoryCandidate(
            memory_candidate_id=candidate_id,
            decision_id=decision_id,
            session_id=session_id,
            source_event_ids=[],
            candidate_type=resolved_type,
            text=text.strip(),
            confidence=max(0.0, min(1.0, confidence)),
            write_policy=MemoryWritePolicy.ELIGIBLE,
            privacy_level=resolved_privacy,
            reason="User-explicit inline capture from product surface.",
            metadata=candidate_metadata,
        )
        upsert = self.memory.upsert_candidate(candidate, org_id=org_id, user_id=user_id)
        return ProductInlineMemoryCaptureResult(
            session=session,
            memory=upsert.memory,
            upsert=upsert,
            committed=True,
        )

    def privacy_metrics(self, session_id: str) -> ProductPrivacyMetrics:
        """P2-4: per-session privacy enforcement counters.

        Walks the runtime decision ledger for this session and counts
        glasses-side privacy outcomes: high-privacy detections, rate
        limiter defers, safety-flag suppressions, enforcer interventions.
        Provides what the demo needs to show "the system is actively
        protecting sensitive content".
        """

        self.sessions.get_session(session_id)
        decisions = self.runtime.store.list_decisions(session_id=session_id)
        metrics = ProductPrivacyMetrics(session_id=session_id)
        for decision in decisions:
            metrics.total_decisions += 1
            candidate = decision.candidate
            opportunity = candidate.opportunity
            opp_privacy_level = (
                opportunity.privacy_level
                if isinstance(opportunity.privacy_level, str)
                else opportunity.privacy_level.value
            )
            if opp_privacy_level == PrivacyLevel.HIGH.value:
                metrics.high_privacy_opportunities_detected += 1
            prd_surface = (
                candidate.prompt_request.prd_surface
                if isinstance(candidate.prompt_request.prd_surface, str)
                else candidate.prompt_request.prd_surface.value
            )
            if prd_surface in {"glasses_popup", "glasses_starting", "glasses_persistent"}:
                metrics.glasses_decisions += 1
            cand_meta = candidate.metadata or {}
            rate_limit_action = cand_meta.get("rate_limit_action")
            if rate_limit_action == "defer_to_app":
                metrics.deferred_to_app_by_rate_limiter += 1
                for reason in cand_meta.get("rate_limit_reasons", []):
                    metrics.by_rate_limit_reason[reason] = (
                        metrics.by_rate_limit_reason.get(reason, 0) + 1
                    )
                    if reason in {"enforcer_fallback_to_app"} or opp_privacy_level == PrivacyLevel.HIGH.value:
                        metrics.deferred_due_to_privacy += 1
            if cand_meta.get("enforcement_actions") and "text_truncated" in cand_meta["enforcement_actions"]:
                metrics.enforcer_text_truncated += 1
            result = candidate.prompt_result
            if result is not None:
                for flag in result.safety_flags or []:
                    metrics.by_safety_flag[flag] = metrics.by_safety_flag.get(flag, 0) + 1
                    if flag in {"llm_timeout", "sensitive_business_context", "business_context"}:
                        metrics.suppressed_by_safety_flag += 1
        # Memory writes blocked by privacy: count BLOCKED MemoryCandidates
        # we proposed but never committed in this session.
        for candidate in self.runtime.store.list_memory_candidates(session_id=session_id):
            policy_value = (
                candidate.write_policy
                if isinstance(candidate.write_policy, str)
                else candidate.write_policy.value
            )
            level_value = (
                candidate.privacy_level
                if isinstance(candidate.privacy_level, str)
                else candidate.privacy_level.value
            )
            if policy_value == MemoryWritePolicy.BLOCKED.value or level_value == PrivacyLevel.HIGH.value and policy_value != MemoryWritePolicy.ELIGIBLE.value:
                metrics.memory_writes_blocked_by_privacy += 1
        return metrics

    def archive_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        return self.memory.archive_memory(memory_id, reason=reason)

    def forget_memory(self, memory_id: str, *, reason: str = "") -> MemoryForgetResult:
        return self.memory.forget_memory(memory_id, reason=reason)

    def list_pending_memory_updates(
        self,
        *,
        memory_id: str | None = None,
        status: MemoryPendingUpdateStatus | str | None = MemoryPendingUpdateStatus.PENDING,
    ) -> list[MemoryPendingUpdate]:
        return self.memory.list_pending_updates(memory_id=memory_id, status=status)

    def apply_pending_memory_update(self, update_id: str, *, reason: str = "") -> MemoryUpsertResult:
        return self.memory.apply_pending_update(update_id, reason=reason)

    def reject_pending_memory_update(self, update_id: str, *, reason: str = "") -> MemoryPendingUpdate:
        return self.memory.reject_pending_update(update_id, reason=reason)

    def promote_memory(
        self,
        memory_id: str,
        *,
        target_scope: MemoryScope | str | None = None,
        target_memory_type: MemoryType | str | None = None,
        reason: str = "",
        approved: bool = False,
    ) -> MemoryPromotionResult:
        return self.memory.promote_memory(
            memory_id,
            target_scope=target_scope,
            target_memory_type=target_memory_type,
            reason=reason,
            approved=approved,
        )

    def write_meeting_state_memory_snapshot(
        self,
        session_id: str,
        *,
        commit: bool = True,
        include_gaps: bool = True,
    ) -> ProductMemorySnapshotResult:
        session = self.sessions.get_session(session_id)
        state = self._get_or_create_meeting_state(session_id)
        current_ms = max((utterance.end_ms for utterance in state.utterances), default=None)
        gaps = self.meeting_state_tracker.scan_state_gaps(state, current_ms=current_ms) if include_gaps else []
        candidates = memory_candidates_from_meeting_state(state, gaps=gaps)
        upserts = (
            [
                self.memory.upsert_candidate(candidate, org_id=state.org_id, user_id=state.subject_user_id)
                for candidate in candidates
            ]
            if commit
            else []
        )
        memories = [result.memory for result in upserts]
        return ProductMemorySnapshotResult(
            session=session,
            meeting_state=state,
            memory_candidates=candidates,
            memories=memories,
            memory_upserts=upserts,
            committed=commit,
        )

    def extract_session_memories(
        self,
        session_id: str,
        *,
        commit: bool = True,
        max_segments: int = 24,
        max_candidates: int = 8,
        include_meeting_state: bool = True,
        memory_context: list[str] | None = None,
        model: str | None = None,
    ) -> ProductMemoryExtractionResult:
        if self.memory_extraction is None:
            raise ValueError("memory extraction service is not configured")
        session = self.sessions.get_session(session_id)
        snapshot = self.sessions.get_context_snapshot(
            session_id,
            max_segments=max_segments,
            memory_context=memory_context,
        )
        if not snapshot.recent_transcript.segments:
            raise ValueError("cannot extract memory without transcript segments")

        org_id, user_id = _memory_identity(session)
        state = self._get_or_create_meeting_state(session_id)
        extraction = self.memory_extraction.extract_candidates(
            MemoryExtractionRequest(
                session_id=session_id,
                scenario_id=str(session.scene),
                locale=session.locale,
                transcript_window=[
                    TranscriptWindowItem(
                        transcript_id=segment.segment_id,
                        speaker=segment.speaker,
                        text=segment.text,
                        timestamp_ms=segment.start_ms,
                        topic=segment.topic,
                    )
                    for segment in snapshot.recent_transcript.segments
                ],
                session_context={
                    "status": session.status,
                    "pre_context": session.pre_context,
                    "metadata": session.metadata,
                    "org_id": org_id,
                    "subject_user_id": user_id,
                    "transcript_stats": snapshot.transcript_stats,
                },
                meeting_state=state.model_dump(mode="json") if include_meeting_state else {},
                memory_context=snapshot.memory_context,
                privacy_constraints=session.privacy_constraints,
                max_candidates=max_candidates,
            ),
            model=model,
        )
        upserts = (
            [self.memory.upsert_candidate(candidate, org_id=org_id, user_id=user_id) for candidate in extraction.candidates]
            if commit
            else []
        )
        return ProductMemoryExtractionResult(
            session=session,
            memory_candidates=extraction.candidates,
            memories=[result.memory for result in upserts],
            memory_upserts=upserts,
            committed=commit,
            extraction_notes=extraction.extraction_notes,
            safety_flags=extraction.safety_flags,
            model_usage=extraction.model_usage,
        )

    def _persist_memory_candidates_for_event(self, decision_id: str, event_id: str) -> list[MemoryCandidate]:
        candidates = [
            candidate
            for candidate in self.runtime.propose_memory_candidates(decision_id, persist=False)
            if event_id in candidate.source_event_ids
        ]
        return [self.runtime.store.add_memory_candidate(candidate) for candidate in candidates]

    def _commit_memory_candidates(self, candidates: list[MemoryCandidate]) -> list[MemoryRecord]:
        memories: list[MemoryRecord] = []
        for candidate in candidates:
            session = self.sessions.get_session(candidate.session_id)
            org_id, user_id = _memory_identity(session)
            memories.append(self.memory.commit_candidate(candidate, org_id=org_id, user_id=user_id))
        return memories

    def _get_or_create_meeting_state(self, session_id: str) -> MeetingState:
        state = self._meeting_states.get(session_id)
        if state is not None:
            return state
        session = self.sessions.get_session(session_id)
        state = self._create_meeting_state_for_session(session)
        self._meeting_states[session_id] = state
        return state

    def _create_meeting_state_for_session(self, session: AssistantSession) -> MeetingState:
        metadata = session.metadata
        return self.meeting_state_tracker.create_state(
            session.session_id,
            org_id=str(metadata.get("org_id", "default_org")),
            subject_user_id=str(metadata.get("subject_user_id", "default_user")),
            participants=list(metadata.get("participants", [])) if isinstance(metadata.get("participants"), list) else [],
            scheduled_end_ms=_optional_int(metadata.get("scheduled_end_ms")),
        )


def prompt_payload_from_decision(decision: PromptDecisionRecord) -> ProductPromptPayload:
    result = decision.candidate.prompt_result
    return ProductPromptPayload(
        decision_id=decision.decision_id,
        session_id=decision.session_id,
        candidate_id=decision.candidate_id,
        opportunity_id=decision.opportunity_id,
        should_display=decision.display_status == PromptDecisionDisplayStatus.SHOWN,
        display_status=decision.display_status,
        prompt_category=decision.prompt_category,
        content_granularity=decision.content_granularity,
        prd_surface=decision.prd_surface,
        display_mode=decision.display_mode,
        duration_policy=decision.duration_policy,
        glasses_title=result.glasses_title if result is not None else "",
        glasses_text=result.glasses_text if result is not None else "",
        app_detail_text=result.app_detail_text if result is not None else "",
        source_refs=list(result.source_refs) if result is not None else [],
        confidence=result.confidence if result is not None else decision.candidate.opportunity.confidence,
        privacy_level=decision.privacy_level,
        privacy_risk=decision.privacy_risk,
        reason=decision.candidate.reason,
        safety_flags=_safety_flags(decision),
    )


def prompt_payload_from_candidate(candidate: PromptCandidate, *, decision_id: str) -> ProductPromptPayload:
    result = candidate.prompt_result
    status = _enum_value(candidate.status)
    should_display = (
        status == PromptCandidateStatus.GENERATED.value
        and result is not None
        and result.should_prompt
    )
    if should_display:
        display_status = PromptDecisionDisplayStatus.SHOWN
    elif status == PromptCandidateStatus.GENERATION_FAILED.value:
        display_status = PromptDecisionDisplayStatus.FAILED
    else:
        display_status = PromptDecisionDisplayStatus.SUPPRESSED
    return ProductPromptPayload(
        decision_id=decision_id,
        session_id=candidate.session_id,
        candidate_id=candidate.candidate_id,
        opportunity_id=candidate.opportunity.opportunity_id,
        should_display=should_display,
        display_status=display_status,
        prompt_category=(
            str(_enum_value(result.prompt_category))
            if result is not None and result.prompt_category is not None
            else str(_enum_value(candidate.opportunity.prompt_category))
        ),
        content_granularity=(
            result.content_granularity
            if result is not None
            else candidate.opportunity.suggested_content_granularity
        ),
        prd_surface=candidate.prompt_request.prd_surface,
        display_mode=candidate.prompt_request.display_mode,
        duration_policy=candidate.prompt_request.duration_policy,
        glasses_title=result.glasses_title if result is not None else "",
        glasses_text=result.glasses_text if result is not None else "",
        app_detail_text=result.app_detail_text if result is not None else "",
        source_refs=list(result.source_refs) if result is not None else [],
        confidence=result.confidence if result is not None else candidate.opportunity.confidence,
        privacy_level=result.privacy_level if result is not None else candidate.opportunity.privacy_level,
        privacy_risk=result.privacy_risk if result is not None else candidate.opportunity.privacy_risk,
        reason=candidate.reason,
        safety_flags=sorted(
            set(
                [
                    *candidate.opportunity.safety_flags,
                    *(result.safety_flags if result is not None else []),
                    "provisional_asr_partial",
                ]
            )
        ),
    )


def _safety_flags(decision: PromptDecisionRecord) -> list[str]:
    flags = list(decision.candidate.opportunity.safety_flags)
    if decision.candidate.prompt_result is not None:
        flags.extend(decision.candidate.prompt_result.safety_flags)
    return sorted(set(flags))


def _decision_memory_metadata(candidate: Any) -> dict[str, Any]:
    return {
        "memory_context": list(candidate.metadata.get("memory_context", candidate.prompt_request.memory_context)),
        "memory_refs": list(candidate.metadata.get("memory_refs", candidate.prompt_request.session_context.get("memory_refs", []))),
        "retrieved_memory_refs": list(candidate.metadata.get("retrieved_memory_refs", [])),
        "retrieved_memory_result_count": int(candidate.metadata.get("retrieved_memory_result_count", 0)),
        "memory_query_text": str(candidate.metadata.get("memory_query_text", "")),
        "memory_query_prompt_category": str(candidate.metadata.get("memory_query_prompt_category", "")),
        "memory_query_activity_phase": str(candidate.metadata.get("memory_query_activity_phase", "")),
    }


def _prd_surface_for_opportunity(opportunity: Any) -> str:
    if str(_enum_value(opportunity.candidate_timing_action)) == "after_activity":
        return "app_summary_tab"
    if str(_enum_value(opportunity.privacy_level)) == "high" or opportunity.privacy_risk >= 0.7:
        return "app_prompt_tab"
    if str(_enum_value(opportunity.priority)) == "P2":
        return "app_prompt_tab"
    return "glasses_popup"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _transcript_duration_ms(segments: list[TranscriptSegmentRecord]) -> int:
    if not segments:
        return 0
    return max(segment.end_ms for segment in segments) - min(segment.start_ms for segment in segments)


def _inline_capture_candidate_id(
    session_id: str, text: str, memory_type: MemoryCandidateType
) -> str:
    """Stable id for inline-capture candidates: idempotent within session."""

    normalized = " ".join(text.strip().lower().split())
    type_value = memory_type.value if hasattr(memory_type, "value") else str(memory_type)
    digest = sha1(":".join([session_id, type_value, normalized]).encode("utf-8")).hexdigest()[:12]
    return f"memcand_inline_{digest}"


def _warmup_query_text_from_session(session: AssistantSession) -> str:
    """Compose a warmup query string from session metadata.

    Pulls title + project / topic / participants / tags so the memory
    retriever has something to anchor cross-session matches. Returns ""
    when none of those fields carry useful text.
    """

    metadata = session.metadata or {}
    parts: list[str] = []
    title = getattr(session, "title", "") or ""
    if title.strip():
        parts.append(title.strip())
    for key in ("project", "project_name", "topic"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    participants = metadata.get("participants")
    if isinstance(participants, list):
        parts.extend(str(p).strip() for p in participants if str(p).strip())
    for key in ("tags", "keywords"):
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(t).strip() for t in value if str(t).strip())
    return " ".join(parts).strip()


def _memory_identity(session: AssistantSession) -> tuple[str, str]:
    metadata = session.metadata
    org_id = str(metadata.get("org_id", "default_org"))
    user_id = str(metadata.get("subject_user_id", metadata.get("user_id", "default_user")))
    return org_id, user_id


def _merge_unique(first: list[str], second: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*first, *second]:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _gaps_for_current_segment(gaps: list[Any], segment_id: str) -> list[Any]:
    return [gap for gap in gaps if not gap.source_utterance_ids or segment_id in gap.source_utterance_ids]


def _summary_opportunity(
    session_id: str,
    *,
    state: MeetingState,
    gaps: list[Any],
    transcript_segment_ids: list[str],
    sequence: int,
) -> PromptOpportunity:
    gap_text = "\n".join(f"- {gap.text}" for gap in gaps[:8]) or "- 暂无明显未闭合 GAP。"
    captured_text = (
        "请生成本场会议的会后总结、关键结论、待办事项、未确认 GAP 和下一步 Suggest。\n"
        f"会议发言数：{len(state.utterances)}\n"
        f"行动项数：{len(state.action_items)}\n"
        f"决策数：{len(state.decisions)}\n"
        f"风险数：{len(state.risks)}\n"
        f"当前 GAP：\n{gap_text}"
    )
    return PromptOpportunity(
        opportunity_id=f"opp_summary_{session_id}_{sequence:04d}",
        session_id=session_id,
        trigger_segment_ids=transcript_segment_ids[-8:] or ["summary"],
        captured_text=captured_text,
        prompt_category=PromptCategory.SUMMARY_GAP_CHECK,
        activity_phase=ActivityPhase.POST_ACTIVITY,
        candidate_timing_action=CandidateTimingAction.AFTER_ACTIVITY,
        suggested_content_granularity=ContentGranularity.CONCISE_BULLETS,
        priority=PromptPriority.P0,
        confidence=0.9,
        privacy_level=PrivacyLevel.LOW,
        privacy_risk=0.1,
        reason="Generate post-session summary, gap check, and suggest for app summary tab.",
        rule_matches=[
            DetectionRuleMatch(
                rule_name="product_session_summary",
                matched_terms=["summary", "gap_check", "suggest"],
                confidence_delta=0.0,
                reason="Session summary requested by product flow.",
            )
        ],
        metadata={
            "source": "product_session_summary",
            "utterance_count": len(state.utterances),
            "action_item_count": len(state.action_items),
            "decision_count": len(state.decisions),
            "risk_count": len(state.risks),
            "gap_count": len(gaps),
        },
    )


def _summary_memory_query(state: MeetingState) -> str:
    parts = [
        "会后总结",
        "行动项",
        "决策",
        "风险",
        "未确认事项",
        *[item.desc for item in state.action_items[-5:]],
        *[item.topic for item in state.decisions[-5:]],
        *[item.desc for item in state.risks[-5:]],
    ]
    return " ".join(part for part in parts if part)


def _active_entities_from_meeting_state(state: MeetingState) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for item in state.action_items:
        entities.append(
            {
                "id": item.action_item_id,
                "canonical_name": item.desc,
                "type": "action",
                "aliases": [item.owner, item.deadline, item.next_step],
                "last_ts": item.source_ts_ms,
            }
        )
    for item in state.decisions:
        entities.append(
            {
                "id": item.decision_id,
                "canonical_name": item.topic,
                "type": "decision",
                "aliases": [item.conclusion],
                "last_ts": item.ts_ms,
            }
        )
    for item in state.risks:
        entities.append(
            {
                "id": item.risk_id,
                "canonical_name": item.desc,
                "type": "risk",
                "aliases": [],
                "last_ts": item.ts_ms,
            }
        )
    for item in state.mentioned_refs:
        entities.append(
            {
                "id": item.mentioned_ref_id,
                "canonical_name": item.text,
                "type": item.ref_type,
                "aliases": [],
                "last_ts": item.ts_ms,
            }
        )
    return [_clean_active_entity(entity) for entity in entities]


def _clean_active_entity(entity: dict[str, Any]) -> dict[str, Any]:
    aliases = entity.get("aliases", [])
    return {
        **entity,
        "aliases": [str(alias) for alias in aliases if alias],
    }


PUBLIC_KNOWLEDGE_MEMORY_SKIP_TERMS = (
    "导演",
    "编剧",
    "作者",
    "主演",
    "演员",
    "电影",
    "影片",
    "小说",
    "歌曲",
    "歌手",
    "director",
    "author",
    "writer",
    "actor",
    "movie",
    "film",
    "novel",
)

PRIVATE_MEMORY_TERMS = (
    "会议",
    "项目",
    "客户",
    "负责人",
    "谁负责",
    "跟进",
    "截止",
    "deadline",
    "ddl",
    "合同",
    "报价",
    "风险",
    "待办",
    "下一步",
    "meeting",
    "project",
    "customer",
    "owner",
    "todo",
    "action item",
)


def _opportunity_needs_realtime_memory(opportunity: Any) -> bool:
    category = str(_enum_value(getattr(opportunity, "prompt_category", "")))
    text = str(getattr(opportunity, "captured_text", "") or "").lower()
    if any(term.lower() in text for term in PRIVATE_MEMORY_TERMS):
        return True
    if category not in {"question_answer", "concept_explanation", "person_or_fact"}:
        return False
    if any(term.lower() in text for term in PUBLIC_KNOWLEDGE_MEMORY_SKIP_TERMS):
        return False
    return category == "person_or_fact"


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value
