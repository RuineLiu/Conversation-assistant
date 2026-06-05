from hashlib import sha1

from proactive_assistant.detection.contracts import PromptOpportunity, PromptOpportunityResult
from proactive_assistant.detection.rules import (
    assess_privacy,
    category_rank,
    detect_candidates,
    downgrade_for_privacy,
    priority_rank,
    to_rule_match,
)
from proactive_assistant.sessions import SessionContextSnapshot, TranscriptSegmentRecord


class PromptOpportunityDetector:
    def __init__(self, *, max_opportunities: int = 3) -> None:
        if max_opportunities <= 0:
            raise ValueError("max_opportunities must be positive")
        self.max_opportunities = max_opportunities

    def detect(self, snapshot: SessionContextSnapshot) -> PromptOpportunityResult:
        opportunities: list[PromptOpportunity] = []
        segments = snapshot.recent_transcript.segments
        for segment in segments:
            opportunities.extend(self._detect_segment(snapshot, segment))

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
                    metadata={
                        "speaker": segment.speaker,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "privacy_terms": list(privacy.matched_terms),
                    },
                )
            )
        return opportunities


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
