"""Layer 1 (mic): Azure streaming ASR straight from your microphone.

Captures 16 kHz / mono / 16-bit PCM from the default input device and feeds
it into the project's AzureSpeechSDKStreamingSession via write_audio -- the
exact same path the WebSocket endpoint uses, minus FastAPI. Proves mic ->
Azure -> transcript works before involving the server.

Usage:
    uv run python scripts/asr_mic_azure.py            # talk, Ctrl-C to stop
    uv run python scripts/asr_mic_azure.py --seconds 15

Speak Mandarin (zh-CN per .env). Partial + final transcripts print live.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import sounddevice as sd

from proactive_assistant.asr.azure_streaming import AzureSpeechSDKStreamingRecognizer
from proactive_assistant.asr.settings import AzureSpeechSettings
from proactive_assistant.asr.streaming import StreamingSpeechEventType


SAMPLE_RATE = 16000
BLOCKSIZE = 1600  # 100ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="auto-stop after N seconds (0 = run until Ctrl-C)")
    args = parser.parse_args()

    settings = AzureSpeechSettings()
    print(f"is_configured={settings.is_configured} language={settings.speech_language}")
    if not settings.is_configured:
        raise SystemExit("Azure Speech not configured; check .env")

    _print_input_device()

    recognizer = AzureSpeechSDKStreamingRecognizer(settings)
    stream = recognizer.start_stream(language=settings.speech_language)
    print("stream started. speak now (Ctrl-C to stop)...\n")

    audio_q: queue.Queue[bytes] = queue.Queue()
    stop = threading.Event()

    def on_audio(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            print(f"  (audio status: {status})", file=sys.stderr)
        audio_q.put(bytes(indata))

    def drain_events() -> None:
        while not stop.is_set():
            event = stream.read_event(0.1)
            if event is not None:
                _print_event(event)

    reader = threading.Thread(target=drain_events, daemon=True)
    reader.start()

    started = time.monotonic()
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            dtype="int16",
            channels=1,
            callback=on_audio,
        ):
            while True:
                try:
                    chunk = audio_q.get(timeout=0.1)
                    stream.write_audio(chunk)
                except queue.Empty:
                    pass
                if args.seconds and (time.monotonic() - started) >= args.seconds:
                    break
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stream.end_audio()
        # let final transcripts land
        deadline = time.monotonic() + 8.0
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
        stop.set()
        stream.stop()
    print("done.")
    return 0


def _print_input_device() -> None:
    try:
        idx = sd.default.device[0]
        info = sd.query_devices(idx) if idx is not None else sd.query_devices(kind="input")
        print(f"input device: {info['name']} "
              f"(max in channels={info['max_input_channels']})")
    except Exception as exc:
        print(f"(could not query input device: {exc})")


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
