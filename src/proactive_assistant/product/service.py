from typing import Any

from proactive_assistant.detection import opportunities_from_meeting_gaps
from proactive_assistant.meeting_state import MeetingState, MeetingStateTracker
from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryContext,
    MemoryQuery,
    MemoryRecord,
    MemoryService,
    memory_candidates_from_meeting_state,
)
from proactive_assistant.orchestration import PromptOrchestrationResult, PromptOrchestrator
from proactive_assistant.runtime import (
    FeedbackPolarity,
    FeedbackSignalSource,
    FeedbackSignalType,
    MemoryCandidate,
    PromptDecisionDisplayStatus,
    PromptDecisionRecord,
    PromptRuntimeService,
    RewardObservation,
)
from proactive_assistant.sessions import AssistantSession, SessionConfig, SessionService, TranscriptSegmentInput

from proactive_assistant.product.contracts import (
    ProductFeedbackResult,
    ProductMemorySnapshotResult,
    ProductPromptPayload,
    ProductTranscriptStepResult,
)


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
    ) -> None:
        self.sessions = session_service
        self.prompt_orchestrator = prompt_orchestrator
        self.runtime = runtime_service
        self.meeting_state_tracker = meeting_state_tracker or MeetingStateTracker()
        self.memory = memory_service or MemoryService(InMemoryMemoryStore())
        self._meeting_states: dict[str, MeetingState] = {}

    def create_session(
        self,
        config: SessionConfig | None = None,
        *,
        session_id: str | None = None,
        start: bool = True,
    ) -> AssistantSession:
        session = self.sessions.create_session(config, session_id=session_id, start=start)
        self._meeting_states[session.session_id] = self._create_meeting_state_for_session(session)
        return session

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
        resolved_memory_context = _merge_unique(
            memory_context or [],
            retrieved_memory_context.memory_context if retrieved_memory_context is not None else [],
        )
        resolved_memory_refs = _merge_unique(
            memory_refs or [],
            retrieved_memory_context.memory_refs if retrieved_memory_context is not None else [],
        )
        snapshot = self.sessions.get_context_snapshot(
            session_id,
            max_segments=max_segments,
            memory_context=resolved_memory_context,
            memory_refs=resolved_memory_refs,
        )
        state_opportunities = opportunities_from_meeting_gaps(
            session_id,
            _gaps_for_current_segment(meeting_update.gaps, transcript_segment.segment_id),
            fallback_segment_id=transcript_segment.segment_id,
            privacy_constraints=snapshot.privacy_constraints,
        )
        orchestration_result = self.prompt_orchestrator.run(snapshot, extra_opportunities=state_opportunities)
        decisions = self.runtime.log_orchestration_result(orchestration_result, policy_version=policy_version)
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

    def get_meeting_state(self, session_id: str) -> MeetingState:
        return self._get_or_create_meeting_state(session_id)

    def list_prompt_decisions(self, *, session_id: str | None = None) -> list[PromptDecisionRecord]:
        return self.runtime.store.list_decisions(session_id=session_id)

    def list_prompt_payloads(self, *, session_id: str | None = None) -> list[ProductPromptPayload]:
        return [prompt_payload_from_decision(decision) for decision in self.list_prompt_decisions(session_id=session_id)]

    def record_feedback(
        self,
        decision_id: str,
        signal_type: FeedbackSignalType | str,
        *,
        event_id: str | None = None,
        source: FeedbackSignalSource | str | None = None,
        polarity: FeedbackPolarity | str | None = None,
        intensity: float | None = None,
        text: str = "",
        dwell_ms: int | None = None,
        latency_ms: int | None = None,
        helpfulness: float | None = None,
        timing_fit: float | None = None,
        content_fit: float | None = None,
        task_progress_delta: float | None = None,
        flow_break_score: float | None = None,
        redundancy_score: float | None = None,
        privacy_risk_score: float | None = None,
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
            polarity=polarity,
            intensity=intensity,
            text=text,
            dwell_ms=dwell_ms,
            latency_ms=latency_ms,
            helpfulness=helpfulness,
            timing_fit=timing_fit,
            content_fit=content_fit,
            task_progress_delta=task_progress_delta,
            flow_break_score=flow_break_score,
            redundancy_score=redundancy_score,
            privacy_risk_score=privacy_risk_score,
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

    def archive_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        return self.memory.archive_memory(memory_id, reason=reason)

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
        memories = (
            [
                self.memory.commit_candidate_once(candidate, org_id=state.org_id, user_id=state.subject_user_id)
                for candidate in candidates
            ]
            if commit
            else []
        )
        return ProductMemorySnapshotResult(
            session=session,
            meeting_state=state,
            memory_candidates=candidates,
            memories=memories,
            committed=commit,
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


def _safety_flags(decision: PromptDecisionRecord) -> list[str]:
    flags = list(decision.candidate.opportunity.safety_flags)
    if decision.candidate.prompt_result is not None:
        flags.extend(decision.candidate.prompt_result.safety_flags)
    return sorted(set(flags))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


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
