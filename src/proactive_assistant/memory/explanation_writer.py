"""Close the loop for LLM-detected term explanations.

When the unknown-term detector produces an explanation for a term, the
explanation should be persisted to memory so the same term, encountered
again later in the same session (or in a future session once promoted),
does not trigger another LLM call.

This writer is a thin façade over ``MemoryService.commit_candidate_once``:
it builds a deterministic ``MemoryCandidate`` keyed on
``(session_id, normalized_term)`` so repeated commits are idempotent, and
tags the resulting memory with ``term_explanation`` so
``PersonalVocabularyService.get_explained_terms_for_session`` picks it up
on the next detection cycle.

Failures are surfaced via exceptions; the product flow is expected to
wrap calls in ``try/except`` so a memory write hiccup never breaks the
realtime prompt path.
"""

from __future__ import annotations

from hashlib import sha1
from typing import Iterable

from proactive_assistant.memory.contracts import MemoryRecord
from proactive_assistant.memory.service import MemoryService
from proactive_assistant.prompting import PrivacyLevel
from proactive_assistant.runtime import (
    MemoryCandidate,
    MemoryCandidateType,
    MemoryWritePolicy,
)


_EXPLANATION_TAG = "term_explanation"
_DEFAULT_TERM_TYPE = "acronym"


class ExplanationMemoryWriter:
    """Persist LLM-generated term explanations into session-scope memory."""

    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service

    @property
    def memory_service(self) -> MemoryService:
        return self._memory_service

    def commit_explanation(
        self,
        *,
        session_id: str,
        org_id: str,
        user_id: str,
        term: str,
        explanation: str,
        term_type: str = _DEFAULT_TERM_TYPE,
        source_segment_id: str = "",
        detector_candidate_id: str = "",
        decision_id: str = "",
        source_event_ids: Iterable[str] | None = None,
        confidence: float = 0.8,
        privacy_level: PrivacyLevel | str = PrivacyLevel.LOW,
        target_speaker_id: str = "",
    ) -> MemoryRecord:
        """Write a term explanation as a session-scope memory record.

        Idempotent on ``(session_id, normalized_term)``: a second call with
        the same session + term returns the existing record without
        re-inserting. The detector candidate id is preserved in metadata
        for traceability but does not affect the storage key.
        """

        if not term.strip():
            raise ValueError("term must not be empty")
        if not explanation.strip():
            raise ValueError("explanation must not be empty")

        normalized_term = term.strip().lower()
        resolved_term_type = (term_type or _DEFAULT_TERM_TYPE).strip() or _DEFAULT_TERM_TYPE
        clamped_confidence = max(0.0, min(1.0, float(confidence)))
        privacy_value = privacy_level.value if hasattr(privacy_level, "value") else str(privacy_level)
        decision = decision_id or _explanation_decision_id(session_id)
        candidate_event_ids = [str(event_id) for event_id in (source_event_ids or []) if str(event_id)]

        candidate = MemoryCandidate(
            memory_candidate_id=_explanation_candidate_id(session_id, normalized_term),
            decision_id=decision,
            session_id=session_id,
            source_event_ids=candidate_event_ids,
            candidate_type=MemoryCandidateType.MEETING_FACT,
            text=explanation.strip(),
            confidence=clamped_confidence,
            write_policy=MemoryWritePolicy.ELIGIBLE,
            privacy_level=PrivacyLevel(privacy_value),
            reason=f"LLM-detected term explanation for '{term.strip()}'.",
            metadata={
                "org_id": org_id,
                "subject_user_id": user_id,
                "term": term.strip(),
                "term_type": resolved_term_type,
                "entity": term.strip(),
                "canonical_entity": term.strip(),
                "normalized_entity": normalized_term,
                "topic": term.strip(),
                "source_capture_ref": (
                    f"transcript:{source_segment_id}" if source_segment_id else ""
                ),
                "memory_source": "unknown_term_explanation_v1",
                "detector_candidate_id": detector_candidate_id,
                "target_speaker_id": target_speaker_id,
                "tags": [
                    _EXPLANATION_TAG,
                    "unknown_term",
                    f"term_type:{resolved_term_type}",
                ],
            },
        )

        return self._memory_service.commit_candidate_once(
            candidate,
            org_id=org_id,
            user_id=user_id,
        )


def _explanation_candidate_id(session_id: str, normalized_term: str) -> str:
    seed = f"{session_id}:{normalized_term}"
    digest = sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"memcand_explain_{digest}"


def _explanation_decision_id(session_id: str) -> str:
    return f"explanation:{session_id}"
