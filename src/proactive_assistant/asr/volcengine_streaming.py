"""豆包大模型流式语音识别(火山引擎 v3 sauc bigmodel)的流式识别器。

协议:WebSocket 二进制帧(大端)。每帧 = 4 字节 header [+ 4 字节 sequence] +
4 字节 payload_size + payload。full client request 发 gzip(JSON 配置);音频帧发
gzip(PCM 块);最后一包用负 sequence。服务端返回 gzip(JSON),含 result.text 和
result.utterances(definite=True 表示一句确定)。

鉴权走 HTTP header:X-Api-App-Key / X-Api-Access-Key / X-Api-Resource-Id /
X-Api-Connect-Id。
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import threading
import uuid
from typing import Any

from proactive_assistant.asr.service import SpeechRecognitionError
from proactive_assistant.asr.settings import VolcengineSpeechSettings
from proactive_assistant.asr.streaming import (
    QueueBackedStreamingSpeechSession,
    StreamingSpeechEvent,
    StreamingSpeechEventType,
)

_PROTOCOL_VERSION = 0b0001
_HEADER_SIZE = 0b0001
_FULL_CLIENT_REQUEST = 0b0001
_AUDIO_ONLY_REQUEST = 0b0010
_FULL_SERVER_RESPONSE = 0b1001
_SERVER_ACK = 0b1011
_SERVER_ERROR = 0b1111
_FLAG_NONE = 0b0000
_FLAG_POS_SEQ = 0b0001
_FLAG_NEG_SEQ = 0b0010          # 最后一包,无序号
_FLAG_NEG_WITH_SEQ = 0b0011     # 最后一包,带序号
_SER_NONE = 0b0000
_SER_JSON = 0b0001
_COMP_NONE = 0b0000
_COMP_GZIP = 0b0001

_DEBUG = os.getenv("VOLC_ASR_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def _header(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
    return bytes(
        [
            (_PROTOCOL_VERSION << 4) | _HEADER_SIZE,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0x00,
        ]
    )


def _full_client_request(config: dict[str, Any], seq: int) -> bytes:
    payload = gzip.compress(json.dumps(config, ensure_ascii=False).encode("utf-8"))
    return (
        _header(_FULL_CLIENT_REQUEST, _FLAG_POS_SEQ, _SER_JSON, _COMP_GZIP)
        + struct.pack(">i", seq)
        + struct.pack(">I", len(payload))
        + payload
    )


def _audio_request(audio: bytes, seq: int, last: bool) -> bytes:
    payload = gzip.compress(audio)
    flags = _FLAG_NEG_WITH_SEQ if last else _FLAG_POS_SEQ
    seq_value = -seq if last else seq
    return (
        _header(_AUDIO_ONLY_REQUEST, flags, _SER_NONE, _COMP_GZIP)
        + struct.pack(">i", seq_value)
        + struct.pack(">I", len(payload))
        + payload
    )


def _parse_server_frame(data: bytes) -> dict[str, Any]:
    header_size = data[0] & 0x0F
    message_type = (data[1] >> 4) & 0x0F
    flags = data[1] & 0x0F
    compression = data[2] & 0x0F
    serialization = (data[2] >> 4) & 0x0F
    offset = max(4, header_size * 4)
    out: dict[str, Any] = {"message_type": message_type, "flags": flags}

    if flags in (_FLAG_POS_SEQ, _FLAG_NEG_WITH_SEQ):
        out["seq"] = struct.unpack(">i", data[offset : offset + 4])[0]
        offset += 4

    if message_type == _SERVER_ERROR:
        out["error_code"] = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4

    if offset + 4 <= len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        body = data[offset : offset + size]
        if compression == _COMP_GZIP and body:
            try:
                body = gzip.decompress(body)
            except Exception:
                pass
        if serialization == _SER_JSON and body:
            try:
                out["payload"] = json.loads(body.decode("utf-8"))
            except Exception:
                out["payload"] = body.decode("utf-8", "ignore")
        else:
            out["payload"] = body
    out["is_last"] = flags in (_FLAG_NEG_SEQ, _FLAG_NEG_WITH_SEQ)
    return out


class VolcengineBigModelSession(QueueBackedStreamingSpeechSession):
    def __init__(
        self,
        *,
        app_key: str,
        access_key: str,
        resource_id: str,
        endpoint: str,
        language: str,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self._seq = 1
        self._closed = False
        self._last_final = ""
        try:
            from websocket import create_connection
        except ImportError as exc:  # pragma: no cover
            raise SpeechRecognitionError("websocket-client not installed; run `uv add websocket-client`") from exc

        headers = [
            f"X-Api-App-Key: {app_key}",
            f"X-Api-Access-Key: {access_key}",
            f"X-Api-Resource-Id: {resource_id}",
            f"X-Api-Connect-Id: {uuid.uuid4()}",
        ]
        try:
            self._ws = create_connection(endpoint, header=headers, timeout=10, enable_multithread=True)
        except Exception as exc:
            raise SpeechRecognitionError(f"Volcengine ASR connect failed: {exc}") from exc

        config = {
            "user": {"uid": "proactive_assistant"},
            "audio": {"format": "pcm", "rate": sample_rate, "bits": 16, "channel": 1, "codec": "raw"},
            "request": {
                "model_name": "bigmodel",
                "enable_punc": True,
                "enable_itn": True,
                "show_utterances": True,
                "result_type": "single",
            },
        }
        try:
            self._ws.send_binary(_full_client_request(config, self._seq))
        except Exception as exc:
            raise SpeechRecognitionError(f"Volcengine ASR init send failed: {exc}") from exc

        self._put(StreamingSpeechEvent(event_type=StreamingSpeechEventType.SESSION_STARTED, language=language))
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while not self._closed:
            try:
                data = self._ws.recv()
            except Exception as exc:
                if not self._closed:
                    self._put(StreamingSpeechEvent(event_type=StreamingSpeechEventType.ERROR, reason=str(exc)))
                break
            if not data or isinstance(data, str):
                continue
            try:
                frame = _parse_server_frame(data)
            except Exception as exc:
                self._put(StreamingSpeechEvent(event_type=StreamingSpeechEventType.ERROR, reason=f"parse: {exc}"))
                continue
            if _DEBUG:
                print(f"[volc] frame mt={frame.get('message_type')} flags={frame.get('flags')} payload={str(frame.get('payload'))[:200]}", flush=True)
            if frame.get("message_type") == _SERVER_ERROR:
                self._put(
                    StreamingSpeechEvent(
                        event_type=StreamingSpeechEventType.ERROR,
                        reason=f"volc error code={frame.get('error_code')} {str(frame.get('payload'))[:200]}",
                    )
                )
                break
            payload = frame.get("payload")
            if isinstance(payload, dict):
                self._emit(payload)
            if frame.get("is_last"):
                self._put(StreamingSpeechEvent(event_type=StreamingSpeechEventType.SESSION_STOPPED))
                break

    def _emit(self, payload: dict[str, Any]) -> None:
        result = payload.get("result") or {}
        if isinstance(result, list):  # some responses wrap result in a list
            result = result[-1] if result else {}
        text = str(result.get("text") or "").strip()
        utterances = result.get("utterances") or []
        emitted_final = False
        for utt in utterances:
            if not isinstance(utt, dict) or not utt.get("definite"):
                continue
            utt_text = str(utt.get("text") or "").strip()
            if not utt_text or utt_text == self._last_final:
                continue
            self._last_final = utt_text
            start = utt.get("start_time")
            end = utt.get("end_time")
            self._put(
                StreamingSpeechEvent(
                    event_type=StreamingSpeechEventType.FINAL_TRANSCRIPT,
                    text=utt_text,
                    offset_ms=start if isinstance(start, int) and start >= 0 else None,
                    duration_ms=(end - start) if isinstance(start, int) and isinstance(end, int) and end >= start else None,
                    metadata={"provider": "volcengine"},
                )
            )
            emitted_final = True
        if not emitted_final and text:
            self._put(
                StreamingSpeechEvent(
                    event_type=StreamingSpeechEventType.PARTIAL_TRANSCRIPT,
                    text=text,
                    metadata={"provider": "volcengine"},
                )
            )

    def write_audio(self, audio: bytes) -> None:
        if self._closed or not audio:
            return
        self._seq += 1
        try:
            self._ws.send_binary(_audio_request(audio, self._seq, last=False))
        except Exception as exc:
            raise SpeechRecognitionError(f"Volcengine ASR write failed: {exc}") from exc

    def end_audio(self) -> None:
        if self._closed:
            return
        self._seq += 1
        try:
            self._ws.send_binary(_audio_request(b"\x00\x00", self._seq, last=True))
        except Exception:
            pass

    def stop(self) -> None:
        self._closed = True
        try:
            self._ws.close()
        except Exception:
            pass


class VolcengineBigModelStreamingRecognizer:
    def __init__(self, settings: VolcengineSpeechSettings) -> None:
        self._settings = settings

    def start_stream(self, *, language: str | None = None, audio_format: str = "pcm16k"):
        s = self._settings
        if not s.is_configured:
            raise SpeechRecognitionError("Volcengine ASR is not configured (need VOLC_ASR_APP_KEY + VOLC_ASR_ACCESS_KEY)")
        return VolcengineBigModelSession(
            app_key=s.app_key or "",
            access_key=s.access_key.get_secret_value() if s.access_key else "",
            resource_id=s.resource_id,
            endpoint=s.endpoint,
            language=language or s.speech_language,
            sample_rate=s.sample_rate,
        )
