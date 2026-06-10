from http.client import RemoteDisconnected

import pytest

from proactive_assistant.asr import AzureSpeechRestRecognizer, AzureSpeechSettings, SpeechRecognitionError


def test_azure_speech_wraps_remote_disconnect_as_recognition_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def disconnect(*_args: object, **_kwargs: object) -> object:
        raise RemoteDisconnected("remote closed")

    monkeypatch.setattr("proactive_assistant.asr.azure.urlopen", disconnect)
    recognizer = AzureSpeechRestRecognizer(
        AzureSpeechSettings(speech_key="sk-test", speech_region="eastus")
    )

    with pytest.raises(SpeechRecognitionError, match="Azure Speech request failed"):
        recognizer.transcribe(b"fake-wav", language="zh-CN", content_type="audio/wav")
