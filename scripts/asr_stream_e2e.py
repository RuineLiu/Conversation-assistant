"""Layer 2: end-to-end streaming ASR through the running FastAPI server.

Creates a session over HTTP, opens the WebSocket ASR stream, pushes a WAV
file as pcm16k chunks, and prints each event plus any prompt the product
flow generated from the recognized transcript.

Prereq: server running, e.g.
    uv run uvicorn proactive_assistant.product.api:create_app --factory --port 8001

Usage:
    uv run python scripts/asr_stream_e2e.py path/to/audio_16k_mono.wav
    uv run python scripts/asr_stream_e2e.py            # 3s silence (connectivity only)
    HOST=127.0.0.1 PORT=8001 uv run python scripts/asr_stream_e2e.py audio.wav
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
import wave

import httpx
import websockets


CHUNK_BYTES = 3200  # 100ms at 16kHz/16-bit/mono


def load_pcm16k(path: str | None) -> bytes:
    if path is None:
        return b"\x00\x00" * 16000 * 3
    with wave.open(path, "rb") as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if (rate, channels, width) != (16000, 1, 2):
        raise SystemExit(
            f"WAV must be 16kHz mono 16-bit PCM, got rate={rate} "
            f"channels={channels} sampwidth={width}."
        )
    return frames


async def main() -> int:
    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "8001")
    base = f"http://{host}:{port}"
    ws_base = f"ws://{host}:{port}"
    path = sys.argv[1] if len(sys.argv) > 1 else None
    audio = load_pcm16k(path)
    session_id = f"asr_demo_{uuid.uuid4().hex[:8]}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base}/sessions",
            json={
                "session_id": session_id,
                "config": {
                    "title": "ASR streaming demo",
                    "metadata": {"org_id": "org_demo", "subject_user_id": "user_demo"},
                },
            },
        )
        resp.raise_for_status()
        print(f"session created: {session_id}")

    ws_url = f"{ws_base}/sessions/{session_id}/asr/stream?speaker=Bao&language=zh-CN"
    print(f"connecting {ws_url}")
    async with websockets.connect(
        ws_url, max_size=8 * 1024 * 1024, ping_interval=20, ping_timeout=60
    ) as ws:
        async def push() -> None:
            for offset in range(0, len(audio), CHUNK_BYTES):
                await ws.send(audio[offset:offset + CHUNK_BYTES])
                await asyncio.sleep(0.1)  # pace ~real time
            await ws.send(json.dumps({"type": "stop"}))

        pusher = asyncio.create_task(push())
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                msg = json.loads(raw)
                _print_message(msg)
                if msg.get("type") in {"session_stopped", "canceled", "error"}:
                    break
        except asyncio.TimeoutError:
            print("timed out waiting for events")
        finally:
            pusher.cancel()
    print("done.")
    return 0


def _print_message(msg: dict) -> None:
    mtype = msg.get("type")
    if mtype == "stream_opened":
        print(f"[stream_opened] speaker={msg.get('speaker')} lang={msg.get('language')}")
        return
    tr = msg.get("transcription", {})
    if mtype == "partial_transcript":
        print(f"  [partial] {tr.get('text')}")
    elif mtype == "final_transcript":
        print(f"  [FINAL  ] {tr.get('text')}  (conf={tr.get('confidence')})")
        step = msg.get("transcript_step")
        if step:
            prompts = step.get("prompts", [])
            print(f"           -> {len(prompts)} prompt(s) generated, "
                  f"opportunity_count={step.get('opportunity_count')}")
            for p in prompts:
                print(f"              · [{p.get('prompt_category')}] "
                      f"title={p.get('glasses_title')!r} "
                      f"text={p.get('glasses_text')!r} "
                      f"surface={p.get('prd_surface')} "
                      f"show={p.get('should_display')}")
        if msg.get("transcript_error"):
            print(f"           !! transcript_error: {msg['transcript_error']}")
    else:
        print(f"  [{mtype}] {tr.get('reason') or ''}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
