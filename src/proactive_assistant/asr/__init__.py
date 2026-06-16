"""Speech recognition adapters for product audio ingestion."""

from proactive_assistant.asr.azure import AzureSpeechRestRecognizer
from proactive_assistant.asr.azure_streaming import AzureSpeechSDKStreamingRecognizer
from proactive_assistant.asr.aliyun_streaming import AliyunDashScopeStreamingRecognizer
from proactive_assistant.asr.contracts import SpeechTranscriptionResult
from proactive_assistant.asr.partial import PartialTranscriptAggregator, SoftTranscriptSegment, normalize_partial_text
from proactive_assistant.asr.service import FakeSpeechRecognizer, SpeechRecognitionError, SpeechRecognitionService, SpeechRecognizer
from proactive_assistant.asr.settings import AliyunSpeechSettings, AzureSpeechSettings
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
    "AliyunDashScopeStreamingRecognizer",
    "AliyunSpeechSettings",
    "AzureSpeechSettings",
    "FakeSpeechRecognizer",
    "FakeStreamingSpeechRecognizer",
    "PartialTranscriptAggregator",
    "SpeechRecognitionError",
    "SpeechRecognitionService",
    "SpeechRecognizer",
    "SpeechTranscriptionResult",
    "SoftTranscriptSegment",
    "StreamingSpeechEvent",
    "StreamingSpeechEventType",
    "StreamingSpeechRecognitionService",
    "StreamingSpeechRecognizer",
    "StreamingSpeechSession",
    "normalize_partial_text",
]
