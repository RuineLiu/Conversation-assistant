from __future__ import annotations

import concurrent.futures
from hashlib import sha1
from typing import TYPE_CHECKING

from proactive_assistant.detection.contracts import (
    CandidateTimingAction,
    DetectionRuleMatch,
    PromptOpportunity,
    PromptOpportunityResult,
    PromptPriority,
)
from proactive_assistant.detection.rules import (
    assess_privacy,
    category_rank,
    detect_candidates,
    downgrade_for_privacy,
    priority_rank,
    to_rule_match,
)
from proactive_assistant.detection.opportunity_detector import (
    PROMPT_CATEGORY_BY_OPPORTUNITY,
    OpportunityCandidate,
    OpportunityDetectionRequest,
    OpportunityDetector,
    OpportunityGapType,
)
from proactive_assistant.detection.unknown_term_detector import (
    UnknownTermCandidate,
    UnknownTermDetectionRequest,
    UnknownTermDetector,
)
from proactive_assistant.model_gateway import ModelGatewayError, ModelOutputValidationError
from proactive_assistant.prompting import (
    ContentGranularity,
    PromptCategory,
    TranscriptWindowItem,
)
from proactive_assistant.schemas.scenario import ActivityPhase
from proactive_assistant.sessions import SessionContextSnapshot, TranscriptSegmentRecord


if TYPE_CHECKING:
    # PersonalVocabularyService pulls the memory package which is part of a
    # pre-existing import cycle (runtime.contracts -> orchestration.service
    # -> detection). Defer the import to runtime so module load order
    # stays compatible with the rest of the codebase.
    from proactive_assistant.detection.vocabulary import PersonalVocabularyService


class PromptOpportunityDetector:
    def __init__(
        self,
        *,
        max_opportunities: int = 3,
        unknown_term_detector: UnknownTermDetector | None = None,
        vocabulary_service: PersonalVocabularyService | None = None,
        opportunity_detector: OpportunityDetector | None = None,
    ) -> None:
        if max_opportunities <= 0:
            raise ValueError("max_opportunities must be positive")
        self.max_opportunities = max_opportunities
        self._unknown_term_detector = unknown_term_detector
        self._vocabulary_service = vocabulary_service
        self._opportunity_detector = opportunity_detector

    def detect(self, snapshot: SessionContextSnapshot) -> PromptOpportunityResult:
        opportunities: list[PromptOpportunity] = []
        segments = snapshot.recent_transcript.segments
        for segment in segments:
            opportunities.extend(self._detect_segment(snapshot, segment))

        # A1: the two LLM detector arms are independent network calls, so run
        # them concurrently instead of serially. Each arm already swallows its
        # own errors and returns [] on failure, so the futures never raise.
        # - unknown-term detection -> CONCEPT_EXPLANATION opportunities
        # - opportunity detection  -> gap/question/suggestion opportunities the
        #   keyword tables miss (e.g. "ddl" not matching "deadline")
        opportunities.extend(self._detect_llm_arms_concurrently(snapshot))

        deduped = _dedupe_opportunities(opportunities)
        ordered = sorted(
            deduped,
            key=lambda item: (
                priority_rank(item.priority),
                category_rank(item.prompt_category),
                -item.confidence,
                item.trigger_segment_ids[0],
            ),
        )[: self.max_opportunities]
        return PromptOpportunityResult(
            session_id=snapshot.session_id,
            opportunities=ordered,
            inspected_segment_ids=[segment.segment_id for segment in segments],
        )

    def _detect_llm_arms_concurrently(self, snapshot: SessionContextSnapshot) -> list[PromptOpportunity]:
        """Run the unknown-term and opportunity LLM arms in parallel.

        Falls back to sequential execution when only one arm is configured
        (no thread-pool overhead for a single call).
        """

        arms = []
        if self._unknown_term_detector is not None:
            arms.append(self._detect_unknown_terms)
        if self._opportunity_detector is not None:
            arms.append(self._detect_llm_opportunities)
        if not arms:
            return []
        if len(arms) == 1:
            return arms[0](snapshot)
        results: list[PromptOpportunity] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as executor:
            futures = [executor.submit(arm, snapshot) for arm in arms]
            for future in futures:
                results.extend(future.result())
        return results

    def _detect_unknown_terms(self, snapshot: SessionContextSnapshot) -> list[PromptOpportunity]:
        if self._unknown_term_detector is None:
            return []
        segments = snapshot.recent_transcript.segments
        if not segments:
            return []
        window = [_to_window_item(segment) for segment in segments]
        org_id, user_id = _identity_from_snapshot(snapshot)
        known_vocab: list[str] = []
        explained: list[str] = []
        if self._vocabulary_service is not None:
            known_vocab = self._vocabulary_service.get_known_vocabulary(
                session_id=snapshot.session_id,
                org_id=org_id,
                user_id=user_id,
            )
            explained = self._vocabulary_service.get_explained_terms_for_session(
                session_id=snapshot.session_id,
            )
        try:
            result = self._unknown_term_detector.detect(
                UnknownTermDetectionRequest(
                    session_id=snapshot.session_id,
                    locale=snapshot.locale,
                    transcript_window=window,
                    known_vocabulary=known_vocab,
                    explained_in_session=explained,
                    privacy_constraints=list(snapshot.privacy_constraints),
                )
            )
        except (ModelGatewayError, ModelOutputValidationError):
            return []

        opportunities: list[PromptOpportunity] = []
        for candidate in result.candidates:
            opportunities.append(_opportunity_from_unknown_term(snapshot, candidate))
        return opportunities

    def _detect_llm_opportunities(self, snapshot: SessionContextSnapshot) -> list[PromptOpportunity]:
        if self._opportunity_detector is None:
            return []
        segments = snapshot.recent_transcript.segments
        if not segments:
            return []
        window = [_to_window_item(segment) for segment in segments]
        try:
            result = self._opportunity_detector.detect(
                OpportunityDetectionRequest(
                    session_id=snapshot.session_id,
                    locale=snapshot.locale,
                    transcript_window=window,
                    privacy_constraints=list(snapshot.privacy_constraints),
                )
            )
        except (ModelGatewayError, ModelOutputValidationError):
            return []

        opportunities: list[PromptOpportunity] = []
        for candidate in result.candidates:
            opportunities.append(_opportunity_from_llm_candidate(snapshot, candidate))
        return opportunities

    def _detect_segment(
        self,
        snapshot: SessionContextSnapshot,
        segment: TranscriptSegmentRecord,
    ) -> list[PromptOpportunity]:
        candidates = detect_candidates(segment)
        if not candidates:
            return []
        privacy = assess_privacy(segment.text, snapshot.privacy_constraints)
        opportunities: list[PromptOpportunity] = []
        for candidate in candidates:
            adjusted = downgrade_for_privacy(candidate, privacy)
            opportunity_id = _opportunity_id(
                snapshot.session_id,
                segment.segment_id,
                str(adjusted.prompt_category),
                str(adjusted.candidate_timing_action),
            )
            opportunities.append(
                PromptOpportunity(
                    opportunity_id=opportunity_id,
                    session_id=snapshot.session_id,
                    trigger_segment_ids=[segment.segment_id],
                    captured_text=segment.text,
                    prompt_category=adjusted.prompt_category,
                    activity_phase=adjusted.activity_phase,
                    candidate_timing_action=adjusted.candidate_timing_action,
                    suggested_content_granularity=adjusted.suggested_content_granularity,
                    priority=adjusted.priority,
                    confidence=adjusted.confidence,
                    privacy_level=privacy.privacy_level,
                    privacy_risk=privacy.privacy_risk,
                    reason=adjusted.reason,
                    safety_flags=list(privacy.safety_flags),
                    rule_matches=[to_rule_match(adjusted)],
                    target_speaker_id=segment.speaker or "",
                    metadata={
                        "speaker": segment.speaker,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "privacy_terms": list(privacy.matched_terms),
                    },
                )
            )
        return opportunities


def _opportunity_from_unknown_term(
    snapshot: SessionContextSnapshot,
    candidate: UnknownTermCandidate,
) -> PromptOpportunity:
    opportunity_id = _opportunity_id(
        snapshot.session_id,
        candidate.source_segment_id,
        PromptCategory.CONCEPT_EXPLANATION.value,
        candidate.candidate_id,
    )
    term_type_value = (
        candidate.term_type
        if isinstance(candidate.term_type, str)
        else candidate.term_type.value
    )
    speaker = _speaker_for_segment(snapshot, candidate.source_segment_id)
    return PromptOpportunity(
        opportunity_id=opportunity_id,
        session_id=snapshot.session_id,
        trigger_segment_ids=[candidate.source_segment_id],
        captured_text=candidate.term,
        prompt_category=PromptCategory.CONCEPT_EXPLANATION,
        activity_phase=ActivityPhase.IN_ACTIVITY,
        candidate_timing_action=CandidateTimingAction.DURING_ACTIVITY,
        suggested_content_granularity=ContentGranularity.ONE_LINE_ANSWER,
        priority=PromptPriority.P1,
        confidence=candidate.confidence,
        privacy_level=candidate.privacy_level,
        privacy_risk=candidate.privacy_risk,
        reason=f"LLM detected unfamiliar term: {candidate.term}",
        safety_flags=[],
        rule_matches=[
            DetectionRuleMatch(
                rule_name="llm_unknown_term_detector",
                matched_terms=[candidate.term],
                confidence_delta=0.0,
                reason=candidate.rationale or "LLM detector flagged term as unfamiliar.",
            )
        ],
        target_speaker_id=speaker or "",
        metadata={
            "speaker": speaker,
            "pre_generated_explanation": candidate.explanation,
            "unknown_term": candidate.term,
            "unknown_term_type": term_type_value,
            "unknown_term_rationale": candidate.rationale,
            "unknown_term_candidate_id": candidate.candidate_id,
            "detection_source": "llm_unknown_term_detector",
        },
    )


def _opportunity_from_llm_candidate(
    snapshot: SessionContextSnapshot,
    candidate: OpportunityCandidate,
) -> PromptOpportunity:
    category = PROMPT_CATEGORY_BY_OPPORTUNITY[str(candidate.prompt_category)]
    gap_type_value = str(candidate.gap_type)
    speaker = _speaker_for_segment(snapshot, candidate.source_segment_id)
    opportunity_id = _opportunity_id(
        snapshot.session_id,
        candidate.source_segment_id,
        category.value,
        candidate.candidate_id,
    )
    # Gap checks read better as concise bullets; everything else is a
    # one-line answer. The enforcer still caps glasses length downstream.
    granularity = (
        ContentGranularity.CONCISE_BULLETS
        if category == PromptCategory.SUMMARY_GAP_CHECK
        else ContentGranularity.ONE_LINE_ANSWER
    )
    metadata: dict[str, object] = {
        "speaker": speaker,
        "detection_source": "llm_opportunity_detector",
        "llm_opportunity_candidate_id": candidate.candidate_id,
        "llm_opportunity_rationale": candidate.rationale,
    }
    if gap_type_value and gap_type_value != OpportunityGapType.NONE.value:
        metadata["gap_type"] = gap_type_value
    if candidate.owner:
        metadata["owner"] = candidate.owner
        metadata["assignee"] = candidate.owner
    if candidate.deadline:
        metadata["deadline"] = candidate.deadline
    if candidate.entity:
        metadata["entity"] = candidate.entity
        metadata["canonical_entity"] = candidate.entity
    return PromptOpportunity(
        opportunity_id=opportunity_id,
        session_id=snapshot.session_id,
        trigger_segment_ids=[candidate.source_segment_id],
        captured_text=candidate.captured_text,
        prompt_category=category,
        activity_phase=ActivityPhase.IN_ACTIVITY,
        candidate_timing_action=CandidateTimingAction.DURING_ACTIVITY,
        suggested_content_granularity=granularity,
        priority=PromptPriority(str(candidate.priority)),
        confidence=candidate.confidence,
        privacy_level=candidate.privacy_level,
        privacy_risk=candidate.privacy_risk,
        reason=candidate.rationale or f"LLM detected {category.value} opportunity.",
        safety_flags=[],
        rule_matches=[
            DetectionRuleMatch(
                rule_name="llm_opportunity_detector",
                matched_terms=[gap_type_value] if gap_type_value != OpportunityGapType.NONE.value else [],
                confidence_delta=0.0,
                reason=candidate.rationale or "LLM opportunity detector.",
            )
        ],
        target_speaker_id=speaker or "",
        metadata=metadata,
    )


def _to_window_item(segment: TranscriptSegmentRecord) -> TranscriptWindowItem:
    return TranscriptWindowItem(
        transcript_id=segment.segment_id,
        speaker=segment.speaker,
        text=segment.text,
        timestamp_ms=segment.start_ms,
        topic=segment.topic,
    )


def _identity_from_snapshot(snapshot: SessionContextSnapshot) -> tuple[str, str]:
    metadata = snapshot.metadata or {}
    org_id = str(metadata.get("org_id", "default_org"))
    user_id = str(
        metadata.get("subject_user_id", metadata.get("user_id", "default_user"))
    )
    return org_id, user_id


def _speaker_for_segment(snapshot: SessionContextSnapshot, segment_id: str) -> str:
    for segment in snapshot.recent_transcript.segments:
        if segment.segment_id == segment_id:
            return segment.speaker
    return ""


def _opportunity_id(session_id: str, segment_id: str, category: str, timing: str) -> str:
    digest = sha1(f"{session_id}:{segment_id}:{category}:{timing}".encode("utf-8")).hexdigest()[:12]
    return f"opp_{digest}"


def _dedupe_opportunities(opportunities: list[PromptOpportunity]) -> list[PromptOpportunity]:
    best_by_segment: dict[str, PromptOpportunity] = {}
    for opportunity in opportunities:
        key = "|".join(opportunity.trigger_segment_ids)
        current = best_by_segment.get(key)
        if current is None or _is_better(opportunity, current):
            best_by_segment[key] = opportunity
    return list(best_by_segment.values())


def _is_better(candidate: PromptOpportunity, current: PromptOpportunity) -> bool:
    candidate_key = (
        priority_rank(candidate.priority),
        category_rank(candidate.prompt_category),
        -candidate.confidence,
    )
    current_key = (
        priority_rank(current.priority),
        category_rank(current.prompt_category),
        -current.confidence,
    )
    return candidate_key < current_key
