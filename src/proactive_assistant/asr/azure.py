import json
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from proactive_assistant.asr.contracts import SpeechTranscriptionResult
from proactive_assistant.asr.service import SpeechRecognitionError
from proactive_assistant.asr.settings import AzureSpeechSettings


class AzureSpeechRestRecognizer:
    """Azure Speech-to-text REST recognizer for short audio uploads.

    This is a backend-friendly bridge for uploaded chunks. True low-latency
    websocket streaming can be added later without changing product endpoints.
    """

    def __init__(self, settings: AzureSpeechSettings | None = None) -> None:
        self._settings = settings or AzureSpeechSettings()
        if not self._settings.is_configured:
            raise SpeechRecognitionError("Azure Speech is not configured")

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        content_type: str | None = None,
    ) -> SpeechTranscriptionResult:
        resolved_language = language or self._settings.speech_language
        request = Request(
            _speech_url(self._settings, resolved_language),
            data=audio,
            headers={
                "Ocp-Apim-Subscription-Key": self._speech_key(),
                "Content-Type": content_type or self._settings.speech_content_type,
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._settings.request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SpeechRecognitionError(f"Azure Speech HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise SpeechRecognitionError(f"Azure Speech request failed: {exc.reason}") from exc
        except (HTTPException, OSError) as exc:
            raise SpeechRecognitionError(f"Azure Speech request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SpeechRecognitionError("Azure Speech returned invalid JSON") from exc
        return _parse_azure_response(payload, language=resolved_language)

    def _speech_key(self) -> str:
        key = self._settings.speech_key
        if key is None:
            raise SpeechRecognitionError("Azure Speech key is not configured")
        return key.get_secret_value()


def _speech_url(settings: AzureSpeechSettings, language: str) -> str:
    endpoint = (settings.speech_endpoint or "").strip().rstrip("/")
    if endpoint:
        base = endpoint
        if "/speech/recognition/" in base:
            separator = "&" if "?" in base else "?"
            return f"{base}{separator}{urlencode({'language': language})}"
        return f"{base}/speech/recognition/conversation/cognitiveservices/v1?{urlencode({'language': language})}"
    region = (settings.speech_region or "").strip()
    return f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?{urlencode({'language': language})}"


def _parse_azure_response(payload: dict[str, object], *, language: str) -> SpeechTranscriptionResult:
    status = str(payload.get("RecognitionStatus", ""))
    if status and status.lower() not in {"success", "recognizedspeech"}:
        raise SpeechRecognitionError(f"Azure Speech recognition failed: {status}")
    text = str(payload.get("DisplayText") or payload.get("Text") or "").strip()
    if not text:
        raise SpeechRecognitionError("Azure Speech did not return recognized text")
    return SpeechTranscriptionResult(
        provider="azure_speech",
        text=text,
        language=language,
        confidence=_confidence(payload),
        duration_ms=_duration_ms(payload),
        raw_response=payload,
    )


def _confidence(payload: dict[str, object]) -> float:
    nbest = payload.get("NBest")
    if isinstance(nbest, list) and nbest:
        first = nbest[0]
        if isinstance(first, dict):
            confidence = first.get("Confidence")
            if isinstance(confidence, int | float):
                return max(0.0, min(1.0, float(confidence)))
    return 0.0


def _duration_ms(payload: dict[str, object]) -> int | None:
    duration = payload.get("Duration")
    if isinstance(duration, int):
        # Azure REST duration is expressed in 100ns units.
        return int(duration / 10_000)
    return None
