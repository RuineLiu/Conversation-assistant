import concurrent.futures
from hashlib import sha1
from typing import Any

from proactive_assistant.detection import CandidateTimingAction, PromptOpportunity, PromptOpportunityDetector, PromptPriority
from proactive_assistant.model_gateway import ModelGatewayError, ModelGatewayTimeoutError
from proactive_assistant.prompting import (
    ContentGranularity,
    DisplayMode,
    DurationPolicy,
    GLASSES_SURFACES,
    ModelUsageMetadata,
    PRDSurface,
    PrivacyLevel,
    PromptCategory,
    PromptGenerationRequest,
    PromptGenerationResult,
    PromptGenerationService,
    PromptResultEnforcer,
    TranscriptWindowItem,
    enforcement_metadata,
)
from proactive_assistant.sessions import SessionContextSnapshot, TranscriptSegmentRecord

from proactive_assistant.orchestration.contracts import (
    PromptCandidate,
    PromptCandidateStatus,
    PromptOrchestrationResult,
)
from proactive_assistant.orchestration.rate_limiter import (
    RateLimitAction,
    RateLimiter,
    rate_limit_metadata,
)


class PromptOrchestrator:
    """Connect transcript snapshots, opportunity detection, and prompt generation."""

    def __init__(
        self,
        *,
        prompt_service: PromptGenerationService,
        fallback_prompt_service: Any | None = None,
        fallback_on_generation_failure: bool = True,
        detector: PromptOpportunityDetector | None = None,
        max_candidates: int = 3,
        enforcer: PromptResultEnforcer | None = None,
        rate_limiter: RateLimiter | None = None,
        glasses_prompt_timeout_seconds: float | None = None,
    ) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self._prompt_service = prompt_service
        self._fallback_prompt_service = fallback_prompt_service
        self._fallback_on_generation_failure = fallback_on_generation_failure
        self._detector = detector or PromptOpportunityDetector(max_opportunities=max_candidates)
        self._max_candidates = max_candidates
        self._enforcer = enforcer if enforcer is not None else PromptResultEnforcer()
        self._rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        # When set, glasses-bound prompt generations get a hard deadline so
        # the wearable never displays a popup that arrived too late. None
        # disables this and uses the model gateway's own request timeout.
        self._glasses_prompt_timeout_seconds = glasses_prompt_timeout_seconds

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    def run(
        self,
        snapshot: SessionContextSnapshot,
        *,
        extra_opportunities: list[PromptOpportunity] | None = None,
    ) -> PromptOrchestrationResult:
        opportunities = self.select_opportunities(snapshot, extra_opportunities=extra_opportunities)
        candidates = [self.generate_candidate(snapshot, opportunity) for opportunity in opportunities]
        return PromptOrchestrationResult(
            session_id=snapshot.session_id,
            snapshot=snapshot,
            opportunities=opportunities,
            candidates=candidates,
        )

    def select_opportunities(
        self,
        snapshot: SessionContextSnapshot,
        *,
        extra_opportunities: list[PromptOpportunity] | None = None,
    ) -> list[PromptOpportunity]:
        detection_result = self._detector.detect(snapshot)
        return _select_opportunities(
            [*detection_result.opportunities, *(extra_opportunities or [])],
            max_opportunities=self._max_candidates,
        )

    def generate_candidate(
        self,
        snapshot: SessionContextSnapshot,
        opportunity: PromptOpportunity,
    ) -> PromptCandidate:
        return self._generate_candidate(snapshot, opportunity)

    def _generate_candidate(
        self,
        snapshot: SessionContextSnapshot,
        opportunity: PromptOpportunity,
    ) -> PromptCandidate:
        prompt_request = build_prompt_generation_request(snapshot, opportunity)

        # Fast path: the detector pre-generated an explanation. No need to
        # call the prompt generation model again.
        pre_generated = opportunity.metadata.get("pre_generated_explanation")
        if isinstance(pre_generated, str) and pre_generated.strip():
            prompt_result = _result_from_pre_generated_explanation(
                opportunity=opportunity,
                explanation=pre_generated.strip(),
            )
            return self._finalize_candidate(
                snapshot=snapshot,
                opportunity=opportunity,
                prompt_request=prompt_request,
                prompt_result=prompt_result,
                stage="prompt_generation_fast_path",
                extra_metadata={"fast_path_source": "llm_unknown_term_detector"},
                candidate_status_prefix=None,
            )

        try:
            prompt_result = self._generate_prompt_with_optional_deadline(prompt_request)
        except ModelGatewayTimeoutError as timeout_exc:
            # Hard deadline missed on a glasses-bound generation. Build a
            # SUPPRESSED candidate rather than failing or routing through
            # fallback; the wearable never shows a too-late popup.
            return self._build_timeout_candidate(
                snapshot=snapshot,
                opportunity=opportunity,
                prompt_request=prompt_request,
                error=timeout_exc,
            )
        except ModelGatewayError as exc:
            if self._fallback_prompt_service is not None and self._fallback_on_generation_failure:
                return self._generate_fallback_candidate(
                    snapshot=snapshot,
                    opportunity=opportunity,
                    prompt_request=prompt_request,
                    original_error=exc,
                )
            return PromptCandidate(
                candidate_id=_candidate_id(opportunity, "generation_failed"),
                session_id=snapshot.session_id,
                opportunity=opportunity,
                prompt_request=prompt_request,
                status=PromptCandidateStatus.GENERATION_FAILED,
                reason=f"{type(exc).__name__}: {exc}",
                metadata={"stage": "prompt_generation"},
            )

        return self._finalize_candidate(
            snapshot=snapshot,
            opportunity=opportunity,
            prompt_request=prompt_request,
            prompt_result=prompt_result,
            stage="prompt_generation",
            extra_metadata={},
            candidate_status_prefix=None,
        )

    def _generate_prompt_with_optional_deadline(
        self,
        prompt_request: PromptGenerationRequest,
    ) -> PromptGenerationResult:
        deadline = self._glasses_prompt_timeout_seconds
        if deadline is None or deadline <= 0 or not _is_glasses_surface(prompt_request.prd_surface):
            return self._prompt_service.generate_prompt(prompt_request)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._prompt_service.generate_prompt, prompt_request)
            try:
                return future.result(timeout=deadline)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise ModelGatewayTimeoutError(
                    f"glasses prompt generation exceeded {deadline:.2f}s deadline"
                ) from exc

    def _build_timeout_candidate(
        self,
        *,
        snapshot: SessionContextSnapshot,
        opportunity: PromptOpportunity,
        prompt_request: PromptGenerationRequest,
        error: ModelGatewayTimeoutError,
    ) -> PromptCandidate:
        suppressed_result = PromptGenerationResult(
            should_prompt=False,
            prompt_category=None,
            content_granularity=ContentGranularity.NO_ACTION,
            glasses_title="",
            glasses_text="",
            app_detail_text="",
            source_refs=[],
            confidence=0.0,
            privacy_level=PrivacyLevel.LOW,
            privacy_risk=0.0,
            rationale=f"Glasses prompt suppressed by hard deadline: {error}",
            safety_flags=["llm_timeout"],
            model_usage=ModelUsageMetadata(
                provider="orchestrator",
                model="timeout",
                latency_ms=int((self._glasses_prompt_timeout_seconds or 0) * 1000),
                cached=False,
            ),
        )
        return PromptCandidate(
            candidate_id=_candidate_id(opportunity, "timeout_suppressed"),
            session_id=snapshot.session_id,
            opportunity=opportunity,
            prompt_request=prompt_request,
            status=PromptCandidateStatus.SUPPRESSED,
            prompt_result=suppressed_result,
            reason=str(error),
            metadata={
                "stage": "prompt_generation",
                "timeout_seconds": self._glasses_prompt_timeout_seconds,
                "safety_flag_reason": "llm_timeout",
            },
        )

    def _generate_fallback_candidate(
        self,
        *,
        snapshot: SessionContextSnapshot,
        opportunity: PromptOpportunity,
        prompt_request: PromptGenerationRequest,
        original_error: ModelGatewayError,
    ) -> PromptCandidate:
        try:
            prompt_result: PromptGenerationResult = self._fallback_prompt_service.generate_prompt(prompt_request)
        except Exception as fallback_error:
            return PromptCandidate(
                candidate_id=_candidate_id(opportunity, "generation_failed"),
                session_id=snapshot.session_id,
                opportunity=opportunity,
                prompt_request=prompt_request,
                status=PromptCandidateStatus.GENERATION_FAILED,
                reason=(
                    f"{type(original_error).__name__}: {original_error}; "
                    f"fallback {type(fallback_error).__name__}: {fallback_error}"
                ),
                metadata={"stage": "prompt_generation", "fallback_failed": True},
            )

        return self._finalize_candidate(
            snapshot=snapshot,
            opportunity=opportunity,
            prompt_request=prompt_request,
            prompt_result=prompt_result,
            stage="prompt_generation_fallback",
            extra_metadata={
                "fallback_reason": f"{type(original_error).__name__}: {original_error}",
            },
            candidate_status_prefix="fallback_",
        )

    def _finalize_candidate(
        self,
        *,
        snapshot: SessionContextSnapshot,
        opportunity: PromptOpportunity,
        prompt_request: PromptGenerationRequest,
        prompt_result: PromptGenerationResult,
        stage: str,
        extra_metadata: dict[str, Any],
        candidate_status_prefix: str | None,
    ) -> PromptCandidate:
        outcome = self._enforcer.enforce(
            prompt_result,
            prd_surface=prompt_request.prd_surface,
            locale=prompt_request.locale,
        )
        enforced_result = outcome.result

        limit_decision = self._rate_limiter.decide(
            session_id=snapshot.session_id,
            opportunity=opportunity,
            surface=prompt_request.prd_surface,
            phase=opportunity.activity_phase,
            enforcer_fallback_to_app=outcome.fallback_to_app_surface,
        )

        effective_request = prompt_request
        effective_result = enforced_result
        if enforced_result.should_prompt:
            action_value = (
                limit_decision.action
                if isinstance(limit_decision.action, str)
                else limit_decision.action.value
            )
            if action_value == RateLimitAction.DEFER_TO_APP.value and limit_decision.new_surface is not None:
                effective_request = _request_with_surface(prompt_request, limit_decision.new_surface)
            elif action_value == RateLimitAction.DROP.value:
                effective_result = _result_as_suppressed(enforced_result)

        status = (
            PromptCandidateStatus.GENERATED
            if effective_result.should_prompt
            else PromptCandidateStatus.SUPPRESSED
        )
        status_label = (
            f"{candidate_status_prefix}{status.value}"
            if candidate_status_prefix
            else status.value
        )
        metadata: dict[str, Any] = {"stage": stage}
        metadata.update(extra_metadata)
        metadata.update(enforcement_metadata(outcome))
        metadata.update(rate_limit_metadata(limit_decision))

        candidate = PromptCandidate(
            candidate_id=_candidate_id(opportunity, status_label),
            session_id=snapshot.session_id,
            opportunity=opportunity,
            prompt_request=effective_request,
            status=status,
            prompt_result=effective_result,
            reason=effective_result.rationale,
            metadata=metadata,
        )

        # Only register glasses-surface shows. App-surface shows are
        # not rate-limited so do not need to be tracked.
        if status == PromptCandidateStatus.GENERATED and effective_result.should_prompt:
            self._rate_limiter.record_show(
                session_id=snapshot.session_id,
                opportunity=opportunity,
                surface=effective_request.prd_surface,
            )

        return candidate


def build_prompt_generation_request(
    snapshot: SessionContextSnapshot,
    opportunity: PromptOpportunity,
) -> PromptGenerationRequest:
    return PromptGenerationRequest(
        session_id=snapshot.session_id,
        scenario_id=str(_enum_value(snapshot.scene)),
        locale=snapshot.locale,
        transcript_window=[_to_transcript_window_item(segment) for segment in snapshot.recent_transcript.segments],
        session_context={
            "status": _enum_value(snapshot.status),
            "scene": _enum_value(snapshot.scene),
            "pre_context": snapshot.pre_context,
            "transcript_stats": snapshot.transcript_stats,
            "memory_refs": snapshot.memory_refs,
            "metadata": snapshot.metadata,
            "opportunity": _opportunity_context(opportunity),
        },
        memory_context=snapshot.memory_context,
        privacy_constraints=snapshot.privacy_constraints,
        prompt_category_candidate=opportunity.prompt_category,
        target_content_granularity=_target_granularity(opportunity),
        prd_surface=_prd_surface(opportunity),
        display_mode=_display_mode(opportunity),
        duration_policy=_duration_policy(opportunity),
    )


def _to_transcript_window_item(segment: TranscriptSegmentRecord) -> TranscriptWindowItem:
    return TranscriptWindowItem(
        transcript_id=segment.segment_id,
        speaker=segment.speaker,
        text=segment.text,
        timestamp_ms=segment.start_ms,
        topic=segment.topic,
    )


def _opportunity_context(opportunity: PromptOpportunity) -> dict[str, Any]:
    return {
        "opportunity_id": opportunity.opportunity_id,
        "trigger_segment_ids": list(opportunity.trigger_segment_ids),
        "captured_text": opportunity.captured_text,
        "prompt_category": _enum_value(opportunity.prompt_category),
        "activity_phase": _enum_value(opportunity.activity_phase),
        "candidate_timing_action": _enum_value(opportunity.candidate_timing_action),
        "suggested_content_granularity": int(opportunity.suggested_content_granularity),
        "priority": _enum_value(opportunity.priority),
        "detector_confidence": opportunity.confidence,
        "privacy_level": _enum_value(opportunity.privacy_level),
        "privacy_risk": opportunity.privacy_risk,
        "reason": opportunity.reason,
        "safety_flags": list(opportunity.safety_flags),
        "rule_matches": [match.model_dump(mode="json") for match in opportunity.rule_matches],
        "metadata": opportunity.metadata,
    }


def _prd_surface(opportunity: PromptOpportunity) -> PRDSurface:
    if _enum_value(opportunity.candidate_timing_action) == CandidateTimingAction.AFTER_ACTIVITY.value:
        return PRDSurface.APP_SUMMARY_TAB
    if _is_high_privacy(opportunity) or _enum_value(opportunity.priority) == PromptPriority.P2.value:
        return PRDSurface.APP_PROMPT_TAB
    return PRDSurface.GLASSES_POPUP


def _display_mode(opportunity: PromptOpportunity) -> DisplayMode:
    if _is_high_privacy(opportunity) or _enum_value(opportunity.priority) == PromptPriority.P2.value:
        return DisplayMode.WRIST_TURN
    return DisplayMode.AUTO


def _duration_policy(opportunity: PromptOpportunity) -> DurationPolicy:
    if _enum_value(opportunity.candidate_timing_action) == CandidateTimingAction.AFTER_ACTIVITY.value:
        return DurationPolicy.AUTO
    if _enum_value(opportunity.priority) == PromptPriority.P0.value and not _is_high_privacy(opportunity):
        return DurationPolicy.FIVE_SECONDS
    return DurationPolicy.AUTO


def _target_granularity(opportunity: PromptOpportunity) -> ContentGranularity:
    suggested = int(opportunity.suggested_content_granularity)
    if _is_high_privacy(opportunity) or _enum_value(opportunity.priority) == PromptPriority.P2.value:
        suggested = min(suggested, ContentGranularity.ONE_LINE_ANSWER.value)
    return ContentGranularity(suggested)


def _is_high_privacy(opportunity: PromptOpportunity) -> bool:
    return _enum_value(opportunity.privacy_level) == PrivacyLevel.HIGH.value or opportunity.privacy_risk >= 0.7


def _candidate_id(opportunity: PromptOpportunity, status: str) -> str:
    digest = sha1(f"{opportunity.opportunity_id}:{status}".encode("utf-8")).hexdigest()[:12]
    return f"cand_{digest}"


def _select_opportunities(
    opportunities: list[PromptOpportunity],
    *,
    max_opportunities: int,
) -> list[PromptOpportunity]:
    deduped = _dedupe_opportunities(opportunities)
    return sorted(
        deduped,
        key=lambda item: (
            _priority_rank(item.priority),
            _category_rank(item.prompt_category),
            -item.confidence,
            item.trigger_segment_ids[0],
        ),
    )[:max_opportunities]


def _dedupe_opportunities(opportunities: list[PromptOpportunity]) -> list[PromptOpportunity]:
    best_by_key: dict[tuple[str, str, str], PromptOpportunity] = {}
    for opportunity in opportunities:
        key = (
            "|".join(opportunity.trigger_segment_ids),
            str(_enum_value(opportunity.prompt_category)),
            str(_enum_value(opportunity.candidate_timing_action)),
        )
        current = best_by_key.get(key)
        if current is None or _is_better_opportunity(opportunity, current):
            best_by_key[key] = opportunity
    return list(best_by_key.values())


def _is_better_opportunity(candidate: PromptOpportunity, current: PromptOpportunity) -> bool:
    candidate_key = (
        _priority_rank(candidate.priority),
        _category_rank(candidate.prompt_category),
        -candidate.confidence,
    )
    current_key = (
        _priority_rank(current.priority),
        _category_rank(current.prompt_category),
        -current.confidence,
    )
    return candidate_key < current_key


def _priority_rank(priority: object) -> int:
    return {"P0": 0, "P1": 1, "P2": 2}[str(_enum_value(priority))]


def _category_rank(category: object) -> int:
    return {
        "summary_gap_check": 0,
        "suggestion": 1,
        "person_or_fact": 2,
        "concept_explanation": 3,
        "question_answer": 4,
    }[str(_enum_value(category))]


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _is_glasses_surface(surface: Any) -> bool:
    value = surface.value if hasattr(surface, "value") else str(surface)
    return value in GLASSES_SURFACES


def _request_with_surface(
    request: PromptGenerationRequest, new_surface: PRDSurface
) -> PromptGenerationRequest:
    """Return a copy of ``request`` with ``prd_surface`` replaced.

    Uses ``model_validate`` so all cross-field validators re-run.
    """

    payload = request.model_dump(mode="python")
    payload["prd_surface"] = new_surface.value if hasattr(new_surface, "value") else new_surface
    return PromptGenerationRequest.model_validate(payload)


def _result_from_pre_generated_explanation(
    *,
    opportunity: PromptOpportunity,
    explanation: str,
) -> PromptGenerationResult:
    """Build a ``PromptGenerationResult`` from a detector pre-generated explanation.

    Used by the orchestrator fast path. The result skips the second LLM
    round trip so glasses-side latency for unknown-term explanations is
    bounded by the detector call alone. The enforcer still runs downstream
    to enforce hard length caps.
    """

    metadata = opportunity.metadata or {}
    term = str(metadata.get("unknown_term", opportunity.captured_text)).strip() or "提示"
    source_refs = [f"transcript:{seg_id}" for seg_id in opportunity.trigger_segment_ids[:3]] or [
        "transcript:unknown"
    ]
    return PromptGenerationResult(
        should_prompt=True,
        prompt_category=PromptCategory.CONCEPT_EXPLANATION,
        content_granularity=ContentGranularity.ONE_LINE_ANSWER,
        glasses_title=term[:40],
        glasses_text=explanation,
        app_detail_text=explanation,
        source_refs=source_refs,
        confidence=opportunity.confidence,
        privacy_level=opportunity.privacy_level,
        privacy_risk=opportunity.privacy_risk,
        rationale="LLM-detected unknown term explanation, fast-pathed without secondary generation.",
        safety_flags=[],
        model_usage=ModelUsageMetadata(
            provider="unknown_term_detector",
            model="cached",
            latency_ms=0,
            cached=True,
        ),
    )


def _result_as_suppressed(result: PromptGenerationResult) -> PromptGenerationResult:
    """Return a copy of ``result`` rewritten as a no-prompt result.

    Used when the rate limiter drops a candidate. The original glasses
    title/text and source refs are preserved on the returned result so the
    runtime ledger can still surface what would have been shown, but the
    ``should_prompt`` / ``content_granularity`` pair satisfies the contract
    validator for no-action results.
    """

    payload = result.model_dump(mode="python")
    payload["should_prompt"] = False
    payload["content_granularity"] = ContentGranularity.NO_ACTION.value
    return PromptGenerationResult.model_validate(payload)
