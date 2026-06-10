from __future__ import annotations

from enum import StrEnum
from queue import Empty, Queue
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class StreamingSpeechEventType(StrEnum):
    SESSION_STARTED = "session_started"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    CANCELED = "canceled"
    SESSION_STOPPED = "session_stopped"
    ERROR = "error"


class StreamingSpeechEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    event_type: StreamingSpeechEventType
    text: str = ""
    language: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    offset_ms: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    reason: str = ""
    raw_response: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


@runtime_checkable
class StreamingSpeechSession(Protocol):
    def write_audio(self, audio: bytes) -> None: ...

    def end_audio(self) -> None: ...

    def stop(self) -> None: ...

    def read_event(self, timeout_seconds: float = 0.1) -> StreamingSpeechEvent | None: ...


@runtime_checkable
class StreamingSpeechRecognizer(Protocol):
    def start_stream(
        self,
        *,
        language: str | None = None,
        audio_format: str = "pcm16k",
    ) -> StreamingSpeechSession: ...


class StreamingSpeechRecognitionService:
    def __init__(self, recognizer: StreamingSpeechRecognizer) -> None:
        self._recognizer = recognizer

    def start_stream(
        self,
        *,
        language: str | None = None,
        audio_format: str = "pcm16k",
    ) -> StreamingSpeechSession:
        return self._recognizer.start_stream(language=language, audio_format=audio_format)


class QueueBackedStreamingSpeechSession:
    def __init__(self) -> None:
        self._events: Queue[StreamingSpeechEvent] = Queue()
        self._stopped = False

    def _put(self, event: StreamingSpeechEvent) -> None:
        self._events.put(event)

    def read_event(self, timeout_seconds: float = 0.1) -> StreamingSpeechEvent | None:
        try:
            return self._events.get(timeout=timeout_seconds)
        except Empty:
            return None


class FakeStreamingSpeechRecognizer:
    def __init__(
        self,
        *,
        partial_text: str = "这个问题谁负责",
        final_text: str = "这个问题谁负责，下周五 deadline 前能不能定？",
        language: str = "zh-CN",
        confidence: float = 0.92,
    ) -> None:
        self.partial_text = partial_text
        self.final_text = final_text
        self.language = language
        self.confidence = confidence
        self.sessions: list[FakeStreamingSpeechSession] = []

    def start_stream(
        self,
        *,
        language: str | None = None,
        audio_format: str = "pcm16k",
    ) -> StreamingSpeechSession:
        session = FakeStreamingSpeechSession(
            partial_text=self.partial_text,
            final_text=self.final_text,
            language=language or self.language,
            confidence=self.confidence,
            audio_format=audio_format,
        )
        self.sessions.append(session)
        return session


class FakeStreamingSpeechSession(QueueBackedStreamingSpeechSession):
    def __init__(
        self,
        *,
        partial_text: str,
        final_text: str,
        language: str,
        confidence: float,
        audio_format: str,
    ) -> None:
        super().__init__()
        self.partial_text = partial_text
        self.final_text = final_text
        self.language = language
        self.confidence = confidence
        self.audio_format = audio_format
        self.received_audio: list[bytes] = []
        self._partial_sent = False
        self._final_sent = False
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.SESSION_STARTED,
                language=language,
                metadata={"provider": "fake_streaming", "audio_format": audio_format},
            )
        )

    def write_audio(self, audio: bytes) -> None:
        if not audio:
            return
        self.received_audio.append(audio)
        if not self._partial_sent and self.partial_text:
            self._partial_sent = True
            self._put(
                StreamingSpeechEvent(
                    event_type=StreamingSpeechEventType.PARTIAL_TRANSCRIPT,
                    text=self.partial_text,
                    language=self.language,
                    confidence=max(0.0, min(1.0, self.confidence - 0.1)),
                    metadata={"provider": "fake_streaming"},
                )
            )

    def end_audio(self) -> None:
        self._emit_final_and_stop()

    def stop(self) -> None:
        self._emit_final_and_stop()

    def _emit_final_and_stop(self) -> None:
        if self._stopped:
            return
        if not self._final_sent:
            self._final_sent = True
            self._put(
                StreamingSpeechEvent(
                    event_type=StreamingSpeechEventType.FINAL_TRANSCRIPT,
                    text=self.final_text,
                    language=self.language,
                    confidence=self.confidence,
                    offset_ms=0,
                    duration_ms=1800,
                    metadata={"provider": "fake_streaming"},
                )
            )
        self._stopped = True
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.SESSION_STOPPED,
                language=self.language,
                metadata={"provider": "fake_streaming"},
            )
        )
