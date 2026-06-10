"""Speech recognition adapters for product audio ingestion."""

from proactive_assistant.asr.azure import AzureSpeechRestRecognizer
from proactive_assistant.asr.azure_streaming import AzureSpeechSDKStreamingRecognizer
from proactive_assistant.asr.contracts import SpeechTranscriptionResult
from proactive_assistant.asr.service import FakeSpeechRecognizer, SpeechRecognitionError, SpeechRecognitionService, SpeechRecognizer
from proactive_assistant.asr.settings import AzureSpeechSettings
from proactive_assistant.asr.streaming import (
    FakeStreamingSpeechRecognizer,
    StreamingSpeechEvent,
    StreamingSpeechEventType,
    StreamingSpeechRecognitionService,
    StreamingSpeechRecognizer,
    StreamingSpeechSession,
)

__all__ = [
    "AzureSpeechRestRecognizer",
    "AzureSpeechSDKStreamingRecognizer",
    "AzureSpeechSettings",
    "FakeSpeechRecognizer",
    "FakeStreamingSpeechRecognizer",
    "SpeechRecognitionError",
    "SpeechRecognitionService",
    "SpeechRecognizer",
    "SpeechTranscriptionResult",
    "StreamingSpeechEvent",
    "StreamingSpeechEventType",
    "StreamingSpeechRecognitionService",
    "StreamingSpeechRecognizer",
    "StreamingSpeechSession",
]
