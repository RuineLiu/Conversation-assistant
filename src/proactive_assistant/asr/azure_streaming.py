from __future__ import annotations

from queue import Queue

from proactive_assistant.asr.service import SpeechRecognitionError
from proactive_assistant.asr.settings import AzureSpeechSettings
from proactive_assistant.asr.streaming import (
    QueueBackedStreamingSpeechSession,
    StreamingSpeechEvent,
    StreamingSpeechEventType,
    StreamingSpeechSession,
)


class AzureSpeechSDKStreamingRecognizer:
    """Azure Speech SDK streaming recognizer using PushAudioInputStream.

    The client must send raw PCM audio chunks: 16 kHz, 16-bit, mono.
    Browser/mobile encodings such as webm/opus need client-side decoding
    or a later server-side transcoding layer.
    """

    def __init__(self, settings: AzureSpeechSettings | None = None) -> None:
        self._settings = settings or AzureSpeechSettings()
        if not self._settings.is_configured:
            raise SpeechRecognitionError("Azure Speech is not configured")

    def start_stream(
        self,
        *,
        language: str | None = None,
        audio_format: str = "pcm16k",
    ) -> StreamingSpeechSession:
        if audio_format != "pcm16k":
            raise SpeechRecognitionError(f"unsupported streaming audio format: {audio_format}")
        return AzureSpeechSDKStreamingSession(settings=self._settings, language=language)


class AzureSpeechSDKStreamingSession(QueueBackedStreamingSpeechSession):
    def __init__(self, *, settings: AzureSpeechSettings, language: str | None = None) -> None:
        super().__init__()
        self._settings = settings
        self._language = language or settings.speech_language
        self._closed = False
        self._speechsdk = _load_speechsdk()
        self._push_stream = self._build_push_stream()
        self._recognizer = self._build_recognizer()
        self._connect_callbacks()
        try:
            self._recognizer.start_continuous_recognition_async().get()
        except Exception as exc:
            raise SpeechRecognitionError(f"Azure Speech streaming start failed: {exc}") from exc

    def write_audio(self, audio: bytes) -> None:
        if self._closed or not audio:
            return
        try:
            self._push_stream.write(audio)
        except Exception as exc:
            self._put_error(f"Azure Speech streaming write failed: {exc}")
            raise SpeechRecognitionError(f"Azure Speech streaming write failed: {exc}") from exc

    def end_audio(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._push_stream.close()
        except Exception:
            pass

    def stop(self) -> None:
        if not self._closed:
            self.end_audio()
        try:
            self._recognizer.stop_continuous_recognition_async().get()
        except Exception as exc:
            self._put_error(f"Azure Speech streaming stop failed: {exc}")

    def _build_push_stream(self) -> object:
        stream_format = self._speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1,
        )
        return self._speechsdk.audio.PushAudioInputStream(stream_format=stream_format)

    def _build_recognizer(self) -> object:
        speech_config = _speech_config(self._speechsdk, self._settings)
        speech_config.speech_recognition_language = self._language
        audio_config = self._speechsdk.audio.AudioConfig(stream=self._push_stream)
        return self._speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    def _connect_callbacks(self) -> None:
        self._recognizer.session_started.connect(
            lambda event: self._put(
                StreamingSpeechEvent(
                    event_type=StreamingSpeechEventType.SESSION_STARTED,
                    language=self._language,
                    raw_response=_event_payload(event),
                    metadata={"provider": "azure_speech_sdk"},
                )
            )
        )
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.session_stopped.connect(self._on_session_stopped)

    def _on_recognizing(self, event: object) -> None:
        result = getattr(event, "result", None)
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            return
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.PARTIAL_TRANSCRIPT,
                text=text,
                language=self._language,
                offset_ms=_ticks_to_ms(getattr(result, "offset", None)),
                duration_ms=_ticks_to_ms(getattr(result, "duration", None)),
                raw_response=_event_payload(event),
                metadata={"provider": "azure_speech_sdk"},
            )
        )

    def _on_recognized(self, event: object) -> None:
        result = getattr(event, "result", None)
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            return
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.FINAL_TRANSCRIPT,
                text=text,
                language=self._language,
                confidence=_confidence_from_result(result),
                offset_ms=_ticks_to_ms(getattr(result, "offset", None)),
                duration_ms=_ticks_to_ms(getattr(result, "duration", None)),
                raw_response=_event_payload(event),
                metadata={"provider": "azure_speech_sdk", "reason": str(getattr(result, "reason", ""))},
            )
        )

    def _on_canceled(self, event: object) -> None:
        details = getattr(event, "cancellation_details", None)
        reason = str(getattr(details, "reason", "") or getattr(event, "reason", "") or "canceled")
        error_details = str(getattr(details, "error_details", "") or "")
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.CANCELED,
                language=self._language,
                reason=error_details or reason,
                raw_response=_event_payload(event),
                metadata={"provider": "azure_speech_sdk"},
            )
        )

    def _on_session_stopped(self, event: object) -> None:
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.SESSION_STOPPED,
                language=self._language,
                raw_response=_event_payload(event),
                metadata={"provider": "azure_speech_sdk"},
            )
        )

    def _put_error(self, reason: str) -> None:
        self._put(
            StreamingSpeechEvent(
                event_type=StreamingSpeechEventType.ERROR,
                language=self._language,
                reason=reason,
                metadata={"provider": "azure_speech_sdk"},
            )
        )


def _load_speechsdk() -> object:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise SpeechRecognitionError(
            "azure-cognitiveservices-speech is not installed; run `uv sync` before using streaming ASR"
        ) from exc
    return speechsdk


def _speech_config(speechsdk: object, settings: AzureSpeechSettings) -> object:
    key = settings.speech_key.get_secret_value() if settings.speech_key is not None else None
    if settings.speech_endpoint:
        return speechsdk.SpeechConfig(subscription=key, endpoint=settings.speech_endpoint)
    if not settings.speech_region:
        raise SpeechRecognitionError("Azure Speech region is not configured")
    return speechsdk.SpeechConfig(subscription=key, region=settings.speech_region)


def _ticks_to_ms(value: object) -> int | None:
    if isinstance(value, int | float):
        return max(0, int(value / 10_000))
    return None


def _confidence_from_result(result: object) -> float:
    properties = getattr(result, "properties", None)
    if properties is None:
        return 0.0
    try:
        raw_json = properties.get_property("SpeechServiceResponse_JsonResult")
    except Exception:
        return 0.0
    if not raw_json:
        return 0.0
    # Avoid importing json on the hot path unless detailed output exists.
    import json

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return 0.0
    nbest = payload.get("NBest")
    if isinstance(nbest, list) and nbest and isinstance(nbest[0], dict):
        confidence = nbest[0].get("Confidence")
        if isinstance(confidence, int | float):
            return max(0.0, min(1.0, float(confidence)))
    return 0.0


def _event_payload(event: object) -> dict[str, object]:
    payload: dict[str, object] = {"event": event.__class__.__name__}
    session_id = getattr(event, "session_id", None)
    if session_id:
        payload["session_id"] = str(session_id)
    result = getattr(event, "result", None)
    if result is not None:
        payload["text"] = str(getattr(result, "text", "") or "")
        payload["reason"] = str(getattr(result, "reason", "") or "")
    return payload
