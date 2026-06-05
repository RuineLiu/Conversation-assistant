"""Session and transcript core for product-facing proactive assistant flows."""

from proactive_assistant.sessions.contracts import (
    AssistantSession,
    SessionConfig,
    SessionContextSnapshot,
    SessionScene,
    SessionSource,
    SessionStatus,
    TranscriptSegmentInput,
    TranscriptSegmentRecord,
    TranscriptWindow,
)
from proactive_assistant.sessions.service import SessionService
from proactive_assistant.sessions.store import InMemorySessionStore, SessionAlreadyExistsError, SessionNotFoundError

__all__ = [
    "AssistantSession",
    "InMemorySessionStore",
    "SessionAlreadyExistsError",
    "SessionConfig",
    "SessionContextSnapshot",
    "SessionNotFoundError",
    "SessionScene",
    "SessionService",
    "SessionSource",
    "SessionStatus",
    "TranscriptSegmentInput",
    "TranscriptSegmentRecord",
    "TranscriptWindow",
]
