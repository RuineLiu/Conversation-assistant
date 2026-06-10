from __future__ import annotations

from types import SimpleNamespace

from proactive_assistant.asr import AliyunDashScopeStreamingRecognizer, AliyunSpeechSettings
from proactive_assistant.asr.aliyun_streaming import AliyunDashScopeStreamingSession
from proactive_assistant.asr.streaming import StreamingSpeechEventType
from proactive_assistant.product.api import _build_streaming_speech_service


class FakeRecognitionCallback:
    def on_open(self) -> None: ...

    def on_event(self, result: object) -> None: ...

    def on_complete(self) -> None: ...

    def on_error(self, result: object) -> None: ...

    def on_close(self) -> None: ...


class FakeRecognitionResult:
    def __init__(self, sentence: dict[str, object], *, request_id: str = "req_001") -> None:
        self._sentence = sentence
        self.request_id = request_id
        self.status_code = 200
        self.code = ""
        self.message = ""
        self.output = {"sentence": sentence}

    def get_sentence(self) -> dict[str, object]:
        return self._sentence

    def get_request_id(self) -> str:
        return self.request_id

    @staticmethod
    def is_sentence_end(sentence: dict[str, object]) -> bool:
        return sentence.get("end_time") is not None


class FakeRecognition:
    instances: list["FakeRecognition"] = []

    def __init__(self, *, model: str, callback: object, format: str, sample_rate: int) -> None:
        self.model = model
        self.callback = callback
        self.format = format
        self.sample_rate = sample_rate
        self.frames: list[bytes] = []
        FakeRecognition.instances.append(self)

    def start(self) -> None:
        self.callback.on_open()

    def send_audio_frame(self, audio: bytes) -> None:
        self.frames.append(audio)
        self.callback.on_event(FakeRecognitionResult({"text": "小张下周五", "begin_time": 0}))

    def stop(self) -> None:
        self.callback.on_event(
            FakeRecognitionResult({"text": "小张下周五确认DDL", "begin_time": 0, "end_time": 1200})
        )
        self.callback.on_complete()
        self.callback.on_close()


def test_aliyun_streaming_recognizer_maps_partial_and_final_events(monkeypatch) -> None:
    FakeRecognition.instances = []
    fake_dashscope = SimpleNamespace(api_key=None, base_websocket_api_url=None)
    monkeypatch.setattr(
        "proactive_assistant.asr.aliyun_streaming._load_dashscope",
        lambda: (fake_dashscope, FakeRecognitionCallback, FakeRecognition),
    )
    settings = AliyunSpeechSettings(
        api_key="sk-test",
        endpoint="wss://dashscope.example/ws",
        model="paraformer-test",
        sample_rate=16000,
    )
    recognizer = AliyunDashScopeStreamingRecognizer(settings)

    stream = recognizer.start_stream(language="zh-CN")
    opened = stream.read_event(0.1)
    stream.write_audio(b"\0" * 3200)
    partial = stream.read_event(0.1)
    stream.end_audio()
    final = stream.read_event(0.1)
    stopped = stream.read_event(0.1)

    assert fake_dashscope.api_key == "sk-test"
    assert fake_dashscope.base_websocket_api_url == "wss://dashscope.example/ws"
    assert FakeRecognition.instances[0].model == "paraformer-test"
    assert FakeRecognition.instances[0].format == "pcm"
    assert FakeRecognition.instances[0].frames == [b"\0" * 3200]
    assert opened is not None and opened.event_type == StreamingSpeechEventType.SESSION_STARTED
    assert partial is not None and partial.event_type == StreamingSpeechEventType.PARTIAL_TRANSCRIPT
    assert partial.text == "小张下周五"
    assert final is not None and final.event_type == StreamingSpeechEventType.FINAL_TRANSCRIPT
    assert final.text == "小张下周五确认DDL"
    assert final.duration_ms == 1200
    assert stopped is not None and stopped.event_type == StreamingSpeechEventType.SESSION_STOPPED


def test_product_streaming_speech_service_uses_aliyun_provider(monkeypatch) -> None:
    FakeRecognition.instances = []
    fake_dashscope = SimpleNamespace(api_key=None, base_websocket_api_url=None)
    monkeypatch.setattr(
        "proactive_assistant.asr.aliyun_streaming._load_dashscope",
        lambda: (fake_dashscope, FakeRecognitionCallback, FakeRecognition),
    )
    monkeypatch.setenv("ASR_PROVIDER", "aliyun")
    monkeypatch.setenv("ALIYUN_ASR_API_KEY", "sk-test")
    monkeypatch.setenv("ALIYUN_ASR_ENDPOINT", "wss://dashscope.example/ws")
    monkeypatch.setenv("ALIYUN_ASR_MODEL", "paraformer-test")

    service = _build_streaming_speech_service()

    assert service is not None
    stream = service.start_stream(language="zh-CN")
    event = stream.read_event(0.1)
    assert event is not None
    assert event.event_type == StreamingSpeechEventType.SESSION_STARTED


def test_azure_settings_can_still_be_constructed_by_field_name() -> None:
    from proactive_assistant.asr import AzureSpeechSettings

    settings = AzureSpeechSettings(speech_key="sk-test", speech_region="eastus")

    assert settings.is_configured
