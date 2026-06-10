"""Layer 1: Azure streaming ASR connectivity smoke test (no FastAPI server).

Isolates "can we talk to Azure Speech at all" from any product / WebSocket
wiring. It opens a real streaming session against the configured endpoint,
pushes audio, and prints every event the recognizer emits.

Usage:
    uv run python scripts/asr_smoke_azure.py path/to/audio_16k_mono.wav
    uv run python scripts/asr_smoke_azure.py            # uses 3s of silence

The WAV must be 16 kHz, 16-bit, mono PCM. A silence clip will connect and
return a SESSION_STARTED + (likely no transcript) + SESSION_STOPPED, which
is enough to prove credentials + endpoint + SDK all work.
"""

from __future__ import annotations

import sys
import time
import wave

from proactive_assistant.asr.azure_streaming import AzureSpeechSDKStreamingRecognizer
from proactive_assistant.asr.settings import AzureSpeechSettings
from proactive_assistant.asr.streaming import StreamingSpeechEventType


CHUNK_BYTES = 3200  # 100ms at 16kHz/16-bit/mono


def load_pcm16k(path: str | None) -> bytes:
    if path is None:
        # 3 seconds of silence
        return b"\x00\x00" * 16000 * 3
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if (rate, channels, width) != (16000, 1, 2):
        raise SystemExit(
            f"WAV must be 16kHz mono 16-bit PCM, got "
            f"rate={rate} channels={channels} sampwidth={width}. "
            f"Re-export it or run it through a converter first."
        )
    return frames


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    settings = AzureSpeechSettings()
    print(f"is_configured={settings.is_configured} "
          f"endpoint={'set' if settings.speech_endpoint else 'none'} "
          f"region={settings.speech_region or 'none'} "
          f"language={settings.speech_language}")
    if not settings.is_configured:
        raise SystemExit("Azure Speech not configured; check .env")

    audio = load_pcm16k(path)
    print(f"loaded {len(audio)} bytes of pcm16k "
          f"(~{len(audio) / (16000 * 2):.1f}s)")

    recognizer = AzureSpeechSDKStreamingRecognizer(settings)
    stream = recognizer.start_stream(language=settings.speech_language)
    print("stream started, pushing audio in 100ms chunks...")

    # Push audio paced roughly in real time so partials have a chance to fire.
    for offset in range(0, len(audio), CHUNK_BYTES):
        stream.write_audio(audio[offset:offset + CHUNK_BYTES])
        while (event := stream.read_event(0.0)) is not None:
            _print_event(event)
        time.sleep(0.1)

    stream.end_audio()
    print("audio ended, draining events...")

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        event = stream.read_event(0.2)
        if event is None:
            continue
        _print_event(event)
        if event.event_type in {
            StreamingSpeechEventType.SESSION_STOPPED.value,
            StreamingSpeechEventType.CANCELED.value,
            StreamingSpeechEventType.ERROR.value,
        }:
            break

    stream.stop()
    print("done.")
    return 0


def _print_event(event) -> None:  # type: ignore[no-untyped-def]
    et = event.event_type
    if et == StreamingSpeechEventType.PARTIAL_TRANSCRIPT.value:
        print(f"  [partial] {event.text}")
    elif et == StreamingSpeechEventType.FINAL_TRANSCRIPT.value:
        print(f"  [FINAL  ] {event.text}  (conf={event.confidence:.2f})")
    elif et == StreamingSpeechEventType.CANCELED.value:
        print(f"  [CANCELED] {event.reason}")
    elif et == StreamingSpeechEventType.ERROR.value:
        print(f"  [ERROR  ] {event.reason}")
    else:
        print(f"  [{et}]")


if __name__ == "__main__":
    raise SystemExit(main())
