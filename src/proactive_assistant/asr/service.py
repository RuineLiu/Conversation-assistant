from typing import Protocol, runtime_checkable

from proactive_assistant.asr.contracts import SpeechTranscriptionResult


class SpeechRecognitionError(RuntimeError):
    pass


@runtime_checkable
class SpeechRecognizer(Protocol):
    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        content_type: str | None = None,
    ) -> SpeechTranscriptionResult: ...


class SpeechRecognitionService:
    def __init__(self, recognizer: SpeechRecognizer) -> None:
        self._recognizer = recognizer

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        content_type: str | None = None,
    ) -> SpeechTranscriptionResult:
        if not audio:
            raise ValueError("audio payload is empty")
        return self._recognizer.transcribe(audio, language=language, content_type=content_type)


class FakeSpeechRecognizer:
    def __init__(
        self,
        result: SpeechTranscriptionResult | None = None,
        *,
        text: str = "这个问题谁负责，下周五 deadline 前能不能定？",
        language: str = "zh-CN",
        confidence: float = 0.92,
    ) -> None:
        self._result = result
        self._text = text
        self._language = language
        self._confidence = confidence
        self.requests: list[dict[str, object]] = []

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        content_type: str | None = None,
    ) -> SpeechTranscriptionResult:
        self.requests.append(
            {
                "audio_size": len(audio),
                "language": language,
                "content_type": content_type,
            }
        )
        if self._result is not None:
            return self._result
        return SpeechTranscriptionResult(
            provider="fake",
            text=self._text,
            language=language or self._language,
            confidence=self._confidence,
            raw_response={"source": "fake_speech_recognizer"},
        )
