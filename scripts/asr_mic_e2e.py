"""Layer 2 (mic): end-to-end streaming ASR from mic through the FastAPI server.

Captures mic audio (16 kHz mono 16-bit) and streams it to the running
server's WebSocket ASR endpoint. Prints live partial/final transcripts plus
any prompts the product flow generates -- this is the full demo loop:
your voice -> Azure -> meeting state -> detection -> prompt.

Prereq: server running
    uv run uvicorn proactive_assistant.product.api:create_app --factory --port 8001

Usage:
    uv run python scripts/asr_mic_e2e.py            # talk, Ctrl-C to stop
    uv run python scripts/asr_mic_e2e.py --seconds 20
    HOST=127.0.0.1 PORT=8001 uv run python scripts/asr_mic_e2e.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid

import httpx
import sounddevice as sd
import websockets


SAMPLE_RATE = 16000
BLOCKSIZE = 1600  # 100ms


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="auto-stop after N seconds (0 = until Ctrl-C)")
    parser.add_argument("--speaker", default="Bao")
    args = parser.parse_args()

    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "8001")
    base = f"http://{host}:{port}"
    ws_base = f"ws://{host}:{port}"
    session_id = f"asr_mic_{uuid.uuid4().hex[:8]}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base}/sessions",
            json={
                "session_id": session_id,
                "config": {
                    "title": "ASR mic demo",
                    "metadata": {"org_id": "org_demo", "subject_user_id": "user_demo"},
                },
            },
        )
        resp.raise_for_status()
    print(f"session created: {session_id}")

    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue[bytes] = asyncio.Queue()

    def on_audio(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))

    ws_url = (f"{ws_base}/sessions/{session_id}/asr/stream"
              f"?speaker={args.speaker}&language=zh-CN")
    print(f"connecting {ws_url}")
    print("speak now (Ctrl-C to stop)...\n")

    stop = asyncio.Event()

    async with websockets.connect(
        ws_url, max_size=8 * 1024 * 1024, ping_interval=20, ping_timeout=60
    ) as ws:
        stop_sent = False

        async def send_stop() -> None:
            nonlocal stop_sent
            if stop_sent:
                return
            stop_sent = True
            await ws.send(json.dumps({"type": "stop"}))

        async def push() -> None:
            elapsed = 0.0
            while not stop.is_set():
                chunk = await audio_q.get()
                await ws.send(chunk)
                elapsed += BLOCKSIZE / SAMPLE_RATE
                if args.seconds and elapsed >= args.seconds:
                    break
            await send_stop()

        async def recv() -> None:
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    msg = json.loads(raw)
                    _print_message(msg)
                    if msg.get("type") in {"session_stopped", "canceled", "error"}:
                        break
            except websockets.exceptions.ConnectionClosedOK:
                pass  # server closed cleanly (code 1000); normal end
            except asyncio.TimeoutError:
                print("(timed out waiting for events)")
            finally:
                stop.set()

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            dtype="int16",
            channels=1,
            callback=on_audio,
        ):
            pusher = asyncio.create_task(push())
            receiver = asyncio.create_task(recv())
            try:
                await receiver
            except KeyboardInterrupt:
                stop.set()
                await send_stop()
                await receiver
            finally:
                stop.set()
                if not pusher.done():
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
            print(f"           -> {len(prompts)} prompt(s), "
                  f"opportunity_count={step.get('opportunity_count')}")
            for p in prompts:
                print(f"              · [{p.get('prompt_category')}] "
                      f"title={p.get('glasses_title')!r} "
                      f"text={p.get('glasses_text')!r} "
                      f"surface={p.get('prd_surface')} show={p.get('should_display')}")
            _print_meeting_state(step)
        if msg.get("transcript_error"):
            print(f"           !! transcript_error: {msg['transcript_error']}")
    else:
        print(f"  [{mtype}] {tr.get('reason') or ''}")


def _print_meeting_state(step: dict) -> None:
    """Show what the pipeline understood, even when no prompt fired."""
    ms = step.get("meeting_state") or {}
    actions = ms.get("action_items", [])
    decisions = ms.get("decisions", [])
    risks = ms.get("risks", [])
    gaps = step.get("meeting_gaps", [])
    if not (actions or decisions or risks or gaps):
        print("           (meeting_state: nothing extracted)")
        return
    for a in actions:
        print(f"           ~ action: desc={a.get('desc')!r} "
              f"owner={a.get('owner')!r} deadline={a.get('deadline')!r} "
              f"status={a.get('status')!r}")
    for d in decisions:
        print(f"           ~ decision: topic={d.get('topic')!r} "
              f"conclusion={d.get('conclusion')!r}")
    for r in risks:
        print(f"           ~ risk: desc={r.get('desc')!r} closed={r.get('closed')}")
    for g in gaps:
        print(f"           ~ gap: type={g.get('gap_type')!r} text={g.get('text')!r}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
