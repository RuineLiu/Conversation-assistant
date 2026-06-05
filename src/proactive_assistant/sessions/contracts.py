from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SessionStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"


class SessionScene(StrEnum):
    MEETING_BUSINESS = "meeting_business"


class SessionSource(StrEnum):
    MANUAL_TRANSCRIPT = "manual_transcript"
    UPLOADED_AUDIO_TRANSCRIPT = "uploaded_audio_transcript"
    LIVE_ASR_FUTURE = "live_asr_future"


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    scene: SessionScene = SessionScene.MEETING_BUSINESS
    source: SessionSource = SessionSource.MANUAL_TRANSCRIPT
    locale: str = "zh-CN"
    title: str = ""
    pre_context: str = ""
    privacy_constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssistantSession(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    session_id: str
    title: str = ""
    status: SessionStatus = SessionStatus.CREATED
    scene: SessionScene = SessionScene.MEETING_BUSINESS
    source: SessionSource = SessionSource.MANUAL_TRANSCRIPT
    locale: str = "zh-CN"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    ended_at: datetime | None = None
    pre_context: str = ""
    privacy_constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def start(self, when: datetime | None = None) -> "AssistantSession":
        now = when or datetime.now(UTC)
        return self.model_copy(update={"status": SessionStatus.RUNNING, "started_at": self.started_at or now})

    def pause(self) -> "AssistantSession":
        return self.model_copy(update={"status": SessionStatus.PAUSED})

    def end(self, when: datetime | None = None) -> "AssistantSession":
        return self.model_copy(update={"status": SessionStatus.ENDED, "ended_at": when or datetime.now(UTC)})


class TranscriptSegmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    asr_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str | None = None
    is_final: bool = True
    source: SessionSource = SessionSource.MANUAL_TRANSCRIPT
    topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> "TranscriptSegmentInput":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class TranscriptSegmentRecord(TranscriptSegmentInput):
    session_id: str
    segment_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TranscriptWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    segments: list[TranscriptSegmentRecord]
    total_chars: int = Field(ge=0)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @classmethod
    def from_segments(cls, session_id: str, segments: list[TranscriptSegmentRecord]) -> "TranscriptWindow":
        if not segments:
            return cls(session_id=session_id, segments=[], total_chars=0)
        return cls(
            session_id=session_id,
            segments=segments,
            total_chars=sum(len(segment.text) for segment in segments),
            start_ms=min(segment.start_ms for segment in segments),
            end_ms=max(segment.end_ms for segment in segments),
        )


class SessionContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    session_id: str
    scene: SessionScene
    status: SessionStatus
    locale: str
    pre_context: str = ""
    recent_transcript: TranscriptWindow
    transcript_stats: dict[str, Any]
    privacy_constraints: list[str] = Field(default_factory=list)
    memory_context: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
