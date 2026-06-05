from hashlib import sha1
from typing import Any

from proactive_assistant.detection import CandidateTimingAction, PromptOpportunity, PromptOpportunityDetector, PromptPriority
from proactive_assistant.model_gateway import ModelGatewayError
from proactive_assistant.prompting import (
    ContentGranularity,
    DisplayMode,
    DurationPolicy,
    PRDSurface,
    PrivacyLevel,
    PromptGenerationRequest,
    PromptGenerationService,
    TranscriptWindowItem,
)
from proactive_assistant.sessions import SessionContextSnapshot, TranscriptSegmentRecord

from proactive_assistant.orchestration.contracts import (
    PromptCandidate,
    PromptCandidateStatus,
    PromptOrchestrationResult,
)


class PromptOrchestrator:
    """Connect transcript snapshots, opportunity detection, and prompt generation."""

    def __init__(
        self,
        *,
        prompt_service: PromptGenerationService,
        detector: PromptOpportunityDetector | None = None,
        max_candidates: int = 3,
    ) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self._prompt_service = prompt_service
        self._detector = detector or PromptOpportunityDetector(max_opportunities=max_candidates)
        self._max_candidates = max_candidates

    def run(
        self,
        snapshot: SessionContextSnapshot,
        *,
        extra_opportunities: list[PromptOpportunity] | None = None,
    ) -> PromptOrchestrationResult:
        detection_result = self._detector.detect(snapshot)
        opportunities = _select_opportunities(
            [*detection_result.opportunities, *(extra_opportunities or [])],
            max_opportunities=self._max_candidates,
        )
        candidates = [self._generate_candidate(snapshot, opportunity) for opportunity in opportunities]
        return PromptOrchestrationResult(
            session_id=snapshot.session_id,
            snapshot=snapshot,
            opportunities=opportunities,
            candidates=candidates,
        )

    def _generate_candidate(
        self,
        snapshot: SessionContextSnapshot,
        opportunity: PromptOpportunity,
    ) -> PromptCandidate:
        prompt_request = build_prompt_generation_request(snapshot, opportunity)
        try:
            prompt_result = self._prompt_service.generate_prompt(prompt_request)
        except ModelGatewayError as exc:
            return PromptCandidate(
                candidate_id=_candidate_id(opportunity, "generation_failed"),
                session_id=snapshot.session_id,
                opportunity=opportunity,
                prompt_request=prompt_request,
                status=PromptCandidateStatus.GENERATION_FAILED,
                reason=f"{type(exc).__name__}: {exc}",
                metadata={"stage": "prompt_generation"},
            )

        status = PromptCandidateStatus.GENERATED if prompt_result.should_prompt else PromptCandidateStatus.SUPPRESSED
        return PromptCandidate(
            candidate_id=_candidate_id(opportunity, status.value),
            session_id=snapshot.session_id,
            opportunity=opportunity,
            prompt_request=prompt_request,
            status=status,
            prompt_result=prompt_result,
            reason=prompt_result.rationale,
            metadata={"stage": "prompt_generation"},
        )


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
        "question_answer": 3,
        "concept_explanation": 4,
    }[str(_enum_value(category))]


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value
