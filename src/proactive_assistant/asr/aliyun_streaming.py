from __future__ import annotations

from typing import Any

from proactive_assistant.asr.service import SpeechRecognitionError
from proactive_assistant.asr.settings import AliyunSpeechSettings
from proactive_assistant.asr.streaming import (
    QueueBackedStreamingSpeechSession,
    StreamingSpeechEvent,
    StreamingSpeechEventType,
    StreamingSpeechSession,
)


class AliyunDashScopeStreamingRecognizer:
    """Aliyun DashScope realtime ASR recognizer.

    The product WebSocket endpoint sends raw PCM chunks encoded as
    16 kHz / 16-bit / mono. DashScope's realtime Paraformer SDK accepts
    the same bytes as ``format="pcm"`` frames.
    """

    def __init__(self, settings: AliyunSpeechSettings | None = None) -> None:
        self._settings = settings or AliyunSpeechSettings()
        if not self._settings.is_configured:
            raise SpeechRecognitionError("Aliyun ASR is not configured")

    def start_stream(
        self,
        *,
        language: str | None = None,
        audio_format: str = "pcm16k",
    ) -> StreamingSpeechSession:
        if audio_format != "pcm16k":
            raise SpeechRecognitionError(f"unsupported streaming audio format: {audio_format}")
        return AliyunDashScopeStreamingSession(settings=self._settings, language=language)


class AliyunDashScopeStreamingSession(QueueBackedStreamingSpeechSession):
    def __init__(self, *, settings: AliyunSpeechSettings, language: str | None = None) -> None:
        super().__init__()
        self._settings = settings
        self._language = language or settings.speech_language
        self._closed = False
        self._completed = False
        self._dashscope, callback_base, recognition_cls = _load_dashscope()
        self._dashscope.api_key = settings.api_key.get_secret_value() if settings.api_key else None
        self._dashscope.base_websocket_api_url = settings.endpoint
        self._callback = _AliyunRecognitionCallback(self, callback_base)
        self._recognition = recognition_cls(
            model=settings.model,
            callback=self._callback,
            format=settings.audio_format,
            sample_rate=settings.sample_rate,
        )
        try:
            self._recognition.start()
        except Exception as exc:
            raise SpeechRecognitionError(f"Aliyun ASR streaming start failed: {exc}") from exc

    def write_audio(self, audio: bytes) -> None:
        if self._closed or not audio:
            return
        try:
            self._recognition.send_audio_frame(audio)
        except Exception as exc:
            self._put_error(f"Aliyun ASR streaming write failed: {exc}")
            raise SpeechRecognitionError(f"Aliyun ASR streaming write failed: {exc}") from exc

    def end_audio(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._recognition.stop()
        except Exception as exc:
            self._put_error(f"Aliyun ASR streaming stop failed: {exc}")

    def stop(self) -> None:
        self.end_audio()

    def _on_open(self) -> None:
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.SESSION_STARTED,
                language=self._language,
                metadata={
                    "provider": "aliyun_dashscope",
                    "model": self._settings.model,
                    "region": self._settings.region,
                },
            )
        )

    def _on_event(self, result: object) -> None:
        sentence = _sentence_from_result(result)
        if not isinstance(sentence, dict):
            return
        text = str(sentence.get("text") or "").strip()
        if not text:
            return
        is_final = _is_sentence_end(result, sentence)
        self._put(
            StreamingSpeechEvent(
                event_type=(
                    StreamingSpeechEventType.FINAL_TRANSCRIPT
                    if is_final
                    else StreamingSpeechEventType.PARTIAL_TRANSCRIPT
                ),
                text=text,
                language=self._language,
                confidence=_confidence_from_sentence(sentence),
                offset_ms=_int_or_none(sentence.get("begin_time")),
                duration_ms=_duration_ms(sentence),
                raw_response=_result_payload(result),
                metadata={
                    "provider": "aliyun_dashscope",
                    "model": self._settings.model,
                    "request_id": _request_id(result),
                    "sentence_end": is_final,
                },
            )
        )

    def _on_complete(self) -> None:
        if self._completed:
            return
        self._completed = True
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.SESSION_STOPPED,
                language=self._language,
                metadata={"provider": "aliyun_dashscope", "model": self._settings.model},
            )
        )

    def _on_error(self, result: object) -> None:
        self._put_error(_error_reason(result))

    def _on_close(self) -> None:
        # DashScope invokes on_complete for normal stream closure. Keep close
        # as a no-op to avoid emitting duplicate SESSION_STOPPED events.
        return

    def _put_error(self, reason: str) -> None:
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.ERROR,
                language=self._language,
                reason=reason,
                metadata={"provider": "aliyun_dashscope", "model": self._settings.model},
            )
        )


def _load_dashscope() -> tuple[object, type, type]:
    try:
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback
    except ImportError as exc:
        raise SpeechRecognitionError("dashscope package is not installed; run `uv sync`") from exc
    return dashscope, RecognitionCallback, Recognition


def _AliyunRecognitionCallback(session: AliyunDashScopeStreamingSession, callback_base: type) -> object:
    class Callback(callback_base):  # type: ignore[misc, valid-type]
        def on_open(self) -> None:
            session._on_open()

        def on_event(self, result: object) -> None:
            session._on_event(result)

        def on_complete(self) -> None:
            session._on_complete()

        def on_error(self, result: object) -> None:
            session._on_error(result)

        def on_close(self) -> None:
            session._on_close()

    return Callback()


def _sentence_from_result(result: object) -> dict[str, Any] | None:
    getter = getattr(result, "get_sentence", None)
    if not callable(getter):
        return None
    sentence = getter()
    if isinstance(sentence, dict):
        return sentence
    if isinstance(sentence, list) and sentence and isinstance(sentence[-1], dict):
        return sentence[-1]
    return None


def _is_sentence_end(result: object, sentence: dict[str, Any]) -> bool:
    checker = getattr(result, "is_sentence_end", None)
    if callable(checker):
        try:
            return bool(checker(sentence))
        except Exception:
            pass
    return sentence.get("end_time") is not None


def _confidence_from_sentence(sentence: dict[str, Any]) -> float:
    raw = sentence.get("confidence")
    if isinstance(raw, int | float):
        return max(0.0, min(1.0, float(raw)))
    return 0.0


def _duration_ms(sentence: dict[str, Any]) -> int | None:
    begin = _int_or_none(sentence.get("begin_time"))
    end = _int_or_none(sentence.get("end_time"))
    if begin is None or end is None:
        return None
    return max(0, end - begin)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _request_id(result: object) -> str:
    getter = getattr(result, "get_request_id", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:
            return ""
    return str(getattr(result, "request_id", "") or "")


def _result_payload(result: object) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key in ("status_code", "request_id", "code", "message", "output", "usage"):
        value = getattr(result, key, None)
        if value is not None:
            payload[key] = value
    return payload


def _error_reason(result: object) -> str:
    code = str(getattr(result, "code", "") or "")
    message = str(getattr(result, "message", "") or "")
    if code and message:
        return f"{code}: {message}"
    return message or code or "Aliyun ASR streaming error"
