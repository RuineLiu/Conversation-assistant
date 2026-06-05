from uuid import uuid4

from proactive_assistant.prompting import PromptGenerationRequest, TranscriptWindowItem
from proactive_assistant.sessions.contracts import (
    AssistantSession,
    SessionConfig,
    SessionContextSnapshot,
    SessionStatus,
    TranscriptSegmentInput,
    TranscriptSegmentRecord,
    TranscriptWindow,
)
from proactive_assistant.sessions.store import SessionStore
from proactive_assistant.sessions.windowing import (
    recent_window_by_chars,
    recent_window_by_segments,
    recent_window_by_time,
)


class SessionService:
    def __init__(self, store: SessionStore) -> None:
        self._store = store

    def create_session(
        self,
        config: SessionConfig | None = None,
        *,
        session_id: str | None = None,
        start: bool = True,
    ) -> AssistantSession:
        resolved_config = config or SessionConfig()
        session = AssistantSession(
            session_id=session_id or f"session_{uuid4().hex[:12]}",
            title=resolved_config.title,
            scene=resolved_config.scene,
            source=resolved_config.source,
            locale=resolved_config.locale,
            pre_context=resolved_config.pre_context,
            privacy_constraints=resolved_config.privacy_constraints,
            metadata=resolved_config.metadata,
        )
        if start:
            session = session.start()
        return self._store.create_session(session)

    def get_session(self, session_id: str) -> AssistantSession:
        return self._store.get_session(session_id)

    def list_sessions(self) -> list[AssistantSession]:
        return self._store.list_sessions()

    def pause_session(self, session_id: str) -> AssistantSession:
        session = self._store.get_session(session_id)
        if session.status == SessionStatus.ENDED:
            raise ValueError("ended sessions cannot be paused")
        return self._store.update_session(session.pause())

    def resume_session(self, session_id: str) -> AssistantSession:
        session = self._store.get_session(session_id)
        if session.status == SessionStatus.ENDED:
            raise ValueError("ended sessions cannot be resumed")
        return self._store.update_session(session.start())

    def end_session(self, session_id: str) -> AssistantSession:
        session = self._store.get_session(session_id)
        return self._store.update_session(session.end())

    def append_transcript(
        self,
        session_id: str,
        segment: TranscriptSegmentInput,
        *,
        segment_id: str | None = None,
    ) -> TranscriptSegmentRecord:
        session = self._store.get_session(session_id)
        if session.status == SessionStatus.ENDED:
            raise ValueError("cannot append transcript to an ended session")
        if session.status == SessionStatus.CREATED:
            self._store.update_session(session.start())
        record = TranscriptSegmentRecord(
            **segment.model_dump(mode="python"),
            session_id=session_id,
            segment_id=segment_id or f"segment_{uuid4().hex[:12]}",
        )
        return self._store.append_transcript(record)

    def get_transcript(self, session_id: str) -> list[TranscriptSegmentRecord]:
        return self._store.list_transcript(session_id)

    def get_transcript_window(
        self,
        session_id: str,
        *,
        max_segments: int | None = None,
        max_chars: int | None = None,
        max_age_ms: int | None = None,
    ) -> TranscriptWindow:
        transcript = self._store.list_transcript(session_id)
        if max_chars is not None:
            return recent_window_by_chars(session_id, transcript, max_chars=max_chars)
        if max_age_ms is not None:
            return recent_window_by_time(session_id, transcript, max_age_ms=max_age_ms)
        return recent_window_by_segments(session_id, transcript, max_segments=max_segments or 12)

    def get_context_snapshot(
        self,
        session_id: str,
        *,
        max_segments: int = 12,
        memory_context: list[str] | None = None,
        memory_refs: list[str] | None = None,
    ) -> SessionContextSnapshot:
        session = self._store.get_session(session_id)
        transcript = self._store.list_transcript(session_id)
        recent = recent_window_by_segments(session_id, transcript, max_segments=max_segments)
        return SessionContextSnapshot(
            session_id=session.session_id,
            scene=session.scene,
            status=session.status,
            locale=session.locale,
            pre_context=session.pre_context,
            recent_transcript=recent,
            transcript_stats={
                "segment_count": len(transcript),
                "total_chars": sum(len(segment.text) for segment in transcript),
                "duration_ms": _duration_ms(transcript),
                "speaker_count": len({segment.speaker for segment in transcript}),
            },
            privacy_constraints=session.privacy_constraints,
            memory_context=memory_context or [],
            memory_refs=memory_refs or [],
            metadata=session.metadata,
        )

    def build_prompt_generation_request(
        self,
        session_id: str,
        *,
        max_segments: int = 12,
        memory_context: list[str] | None = None,
        memory_refs: list[str] | None = None,
    ) -> PromptGenerationRequest:
        snapshot = self.get_context_snapshot(
            session_id,
            max_segments=max_segments,
            memory_context=memory_context,
            memory_refs=memory_refs,
        )
        if not snapshot.recent_transcript.segments:
            raise ValueError("cannot build prompt request without transcript segments")
        return PromptGenerationRequest(
            session_id=snapshot.session_id,
            scenario_id=str(snapshot.scene),
            locale=snapshot.locale,
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
                "status": snapshot.status,
                "pre_context": snapshot.pre_context,
                "transcript_stats": snapshot.transcript_stats,
                "metadata": snapshot.metadata,
            },
            memory_context=snapshot.memory_context,
            privacy_constraints=snapshot.privacy_constraints,
        )


def _duration_ms(transcript: list[TranscriptSegmentRecord]) -> int:
    if not transcript:
        return 0
    return max(segment.end_ms for segment in transcript) - min(segment.start_ms for segment in transcript)
