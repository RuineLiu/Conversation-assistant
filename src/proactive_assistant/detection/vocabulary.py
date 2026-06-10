"""Personal vocabulary baseline for unknown-term detection.

This module pulls a per-user / per-session list of terms the assistant
should treat as already-known and therefore not propose for explanation.
The list is consumed by the LLM unknown-term detector as a hint, not as a
hard filter -- the model still decides what's worth explaining, but it
sees the user's existing vocabulary up-front so it stops re-explaining
``KPI``, ``OKR``, ``GMV`` etc. once the user has seen them.

Sources of known vocabulary:

1. User-scope memories that carry an ``entity`` or ``canonical_entity``
   (typically ``person_or_fact``, ``project_context``, ``user_preference``).
2. Session-scope memories tagged ``term_explanation`` (added by the
   detector itself after a term is explained inside this session).

The service is deliberately tolerant of a missing or empty memory store:
in that case it returns an empty list and the detector keeps working,
just without personalization. This keeps the demo path resilient when
the user has no history yet.
"""

from __future__ import annotations

from typing import Iterable

from proactive_assistant.memory import (
    MemoryQuery,
    MemoryScope,
    MemoryService,
    MemoryType,
)


DEFAULT_KNOWN_VOCAB_LIMIT = 64


_DEFAULT_KNOWN_MEMORY_TYPES: tuple[MemoryType, ...] = (
    MemoryType.PERSON_OR_FACT,
    MemoryType.PROJECT_CONTEXT,
    MemoryType.MEETING_FACT,
    MemoryType.USER_PREFERENCE,
)


class PersonalVocabularyService:
    """Compose a personalized known-vocabulary list for a session.

    Usage::

        vocab = PersonalVocabularyService(memory_service)
        known = vocab.get_known_vocabulary(
            session_id="session_001",
            org_id="org_001",
            user_id="user_001",
        )
        # -> ["KPI", "OKR", "Project Atlas", ...]
    """

    def __init__(
        self,
        memory_service: MemoryService | None,
        *,
        memory_types: Iterable[MemoryType] | None = None,
        limit: int = DEFAULT_KNOWN_VOCAB_LIMIT,
    ) -> None:
        self._memory_service = memory_service
        self._memory_types = tuple(memory_types) if memory_types is not None else _DEFAULT_KNOWN_MEMORY_TYPES
        self._limit = max(1, limit)

    def get_known_vocabulary(
        self,
        *,
        session_id: str,
        org_id: str = "default_org",
        user_id: str = "default_user",
    ) -> list[str]:
        """Return a deduplicated list of terms the user already knows."""

        if self._memory_service is None:
            return []
        records = self._memory_service.store.list_memories(
            MemoryQuery(
                org_id=org_id,
                user_id=user_id,
                session_id=session_id,
                memory_types=list(self._memory_types),
                limit=min(self._limit * 4, 200),
            )
        )
        seen: set[str] = set()
        terms: list[str] = []
        for record in records:
            # Session-scope records are only visible inside their own session;
            # user/org/global scope memories are visible across the session.
            scope = record.scope if isinstance(record.scope, str) else record.scope.value
            if scope == MemoryScope.SESSION.value and record.session_id != session_id:
                continue
            for raw_term in _candidate_terms_from_record(record):
                normalized = _normalize_term(raw_term)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                terms.append(raw_term.strip())
                if len(terms) >= self._limit:
                    return terms
        return terms

    def get_explained_terms_for_session(self, *, session_id: str) -> list[str]:
        """Return the subset of known vocabulary added inside this session.

        Used by the detector to highlight terms it already produced an
        explanation for in this meeting, so the LLM does not reissue them.
        """

        if self._memory_service is None:
            return []
        records = self._memory_service.store.list_memories(
            MemoryQuery(
                session_id=session_id,
                memory_types=[MemoryType.MEETING_FACT, MemoryType.PERSON_OR_FACT],
                tags=["term_explanation"],
                limit=self._limit,
            )
        )
        seen: set[str] = set()
        terms: list[str] = []
        for record in records:
            for raw_term in _candidate_terms_from_record(record):
                normalized = _normalize_term(raw_term)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                terms.append(raw_term.strip())
        return terms


def _candidate_terms_from_record(record: object) -> list[str]:
    """Pull plausible term strings from a memory record's metadata.

    Order matters: original-casing fields come first so they are seen by
    the caller's dedup pass before the lowercase ``normalized_entity``
    variant. This way the LLM receives "GMV" rather than "gmv" in its
    ``explained_in_session`` input.
    """

    metadata = getattr(record, "metadata", {}) or {}
    out: list[str] = []
    for key in ("canonical_entity", "entity", "topic", "normalized_entity"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    tags = metadata.get("term_aliases") if isinstance(metadata, dict) else None
    if isinstance(tags, list):
        for alias in tags:
            if isinstance(alias, str) and alias.strip():
                out.append(alias.strip())
    return out


def _normalize_term(term: str) -> str:
    return term.strip().lower()
