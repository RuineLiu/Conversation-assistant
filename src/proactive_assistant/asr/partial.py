from __future__ import annotations

import re
import time
from dataclasses import dataclass
from hashlib import sha1

from proactive_assistant.asr.streaming import StreamingSpeechEvent


_TRIGGER_PATTERN = re.compile(
    r"(谁|什么|哪|哪里|怎么|为什么|是否|能不能|可以吗|吗|负责|跟进|定下来|"
    r"ddl|deadline|风险|阻塞|待办|todo|action|导演|作者|解释|是什么意思)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SoftTranscriptSegment:
    segment_id: str
    text: str
    speaker: str
    start_ms: int
    end_ms: int
    reason: str
    normalized_text: str


class PartialTranscriptAggregator:
    """Turn unstable ASR partials into deduped soft transcript segments.

    The aggregator is intentionally conservative about persistence: it only
    emits soft segments for preview-time prompt detection. Final ASR events
    remain the source of truth for transcript storage and memory writes.
    """

    def __init__(
        self,
        *,
        min_chars: int = 6,
        min_delta_chars: int = 5,
        min_emit_interval_ms: int = 700,
        max_soft_interval_ms: int = 1800,
    ) -> None:
        self._min_chars = max(1, min_chars)
        self._min_delta_chars = max(1, min_delta_chars)
        self._min_emit_interval_ms = max(0, min_emit_interval_ms)
        self._max_soft_interval_ms = max(self._min_emit_interval_ms, max_soft_interval_ms)
        self._last_emitted_norm = ""
        self._last_emitted_at = 0.0
        self._sequence = 0

    def observe(
        self,
        event: StreamingSpeechEvent,
        *,
        speaker: str,
        fallback_start_ms: int,
        fallback_end_ms: int,
    ) -> SoftTranscriptSegment | None:
        text = event.text.strip()
        normalized = normalize_partial_text(text)
        if len(normalized) < self._min_chars:
            return None
        if normalized == self._last_emitted_norm:
            return None

        now = time.monotonic()
        elapsed_ms = int((now - self._last_emitted_at) * 1000) if self._last_emitted_at else 10**9
        delta_chars = _new_text_delta(self._last_emitted_norm, normalized)
        has_trigger = bool(_TRIGGER_PATTERN.search(text))

        reason = ""
        if has_trigger and (delta_chars >= self._min_delta_chars or not self._last_emitted_norm):
            reason = "trigger_terms"
        elif elapsed_ms >= self._max_soft_interval_ms and delta_chars >= self._min_delta_chars:
            reason = "max_soft_interval"
        elif elapsed_ms >= self._min_emit_interval_ms and delta_chars >= self._min_delta_chars * 2:
            reason = "stable_growth"
        if not reason:
            return None

        self._sequence += 1
        self._last_emitted_norm = normalized
        self._last_emitted_at = now
        start_ms = event.offset_ms if event.offset_ms is not None else fallback_start_ms
        if event.duration_ms is not None:
            end_ms = start_ms + max(1, event.duration_ms)
        else:
            end_ms = max(start_ms + 1, fallback_end_ms)
        return SoftTranscriptSegment(
            segment_id=f"soft_{self._sequence:04d}_{sha1(normalized.encode('utf-8')).hexdigest()[:10]}",
            text=text,
            speaker=speaker or "unknown",
            start_ms=start_ms,
            end_ms=end_ms,
            reason=reason,
            normalized_text=normalized,
        )

    def matches_last_preview(self, text: str) -> bool:
        normalized = normalize_partial_text(text)
        if not normalized or not self._last_emitted_norm:
            return False
        return normalized.startswith(self._last_emitted_norm) or self._last_emitted_norm.startswith(normalized)


def normalize_partial_text(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"[\s,，。.!！?？:：;；、\"'“”‘’（）()\[\]{}<>《》]+", "", lowered)


def _new_text_delta(previous: str, current: str) -> int:
    if not previous:
        return len(current)
    if current.startswith(previous):
        return len(current) - len(previous)
    return len(current)
