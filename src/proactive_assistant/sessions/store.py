from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Protocol, runtime_checkable

from proactive_assistant.sessions.contracts import AssistantSession, TranscriptSegmentRecord


class SessionAlreadyExistsError(ValueError):
    pass


class SessionNotFoundError(KeyError):
    pass


@runtime_checkable
class SessionRepository(Protocol):
    """Repository contract for sessions and transcript segments."""

    def create_session(self, session: AssistantSession) -> AssistantSession: ...

    def update_session(self, session: AssistantSession) -> AssistantSession: ...

    def get_session(self, session_id: str) -> AssistantSession: ...

    def list_sessions(self) -> list[AssistantSession]: ...

    def append_transcript(self, segment: TranscriptSegmentRecord) -> TranscriptSegmentRecord: ...

    def list_transcript(self, session_id: str) -> list[TranscriptSegmentRecord]: ...


SessionStore = SessionRepository


class InMemorySessionStore:
    """In-memory repository for tests and early product wiring.

    It stores copies at boundaries so service tests cannot mutate state by
    holding returned objects.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, AssistantSession] = {}
        self._transcripts: dict[str, list[TranscriptSegmentRecord]] = defaultdict(list)

    def create_session(self, session: AssistantSession) -> AssistantSession:
        if session.session_id in self._sessions:
            raise SessionAlreadyExistsError(f"session already exists: {session.session_id}")
        self._sessions[session.session_id] = deepcopy(session)
        return deepcopy(session)

    def update_session(self, session: AssistantSession) -> AssistantSession:
        if session.session_id not in self._sessions:
            raise SessionNotFoundError(f"session not found: {session.session_id}")
        self._sessions[session.session_id] = deepcopy(session)
        return deepcopy(session)

    def get_session(self, session_id: str) -> AssistantSession:
        try:
            return deepcopy(self._sessions[session_id])
        except KeyError as exc:
            raise SessionNotFoundError(f"session not found: {session_id}") from exc

    def list_sessions(self) -> list[AssistantSession]:
        return sorted((deepcopy(session) for session in self._sessions.values()), key=lambda item: item.created_at)

    def append_transcript(self, segment: TranscriptSegmentRecord) -> TranscriptSegmentRecord:
        if segment.session_id not in self._sessions:
            raise SessionNotFoundError(f"session not found: {segment.session_id}")
        self._transcripts[segment.session_id].append(deepcopy(segment))
        return deepcopy(segment)

    def list_transcript(self, session_id: str) -> list[TranscriptSegmentRecord]:
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"session not found: {session_id}")
        return sorted(
            (deepcopy(segment) for segment in self._transcripts[session_id]),
            key=lambda item: (item.start_ms, item.end_ms, item.segment_id),
        )
