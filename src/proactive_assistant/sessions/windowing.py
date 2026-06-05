from proactive_assistant.sessions.contracts import TranscriptSegmentRecord, TranscriptWindow


def recent_window_by_segments(
    session_id: str,
    segments: list[TranscriptSegmentRecord],
    *,
    max_segments: int,
) -> TranscriptWindow:
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")
    return TranscriptWindow.from_segments(session_id, segments[-max_segments:])


def recent_window_by_time(
    session_id: str,
    segments: list[TranscriptSegmentRecord],
    *,
    max_age_ms: int,
) -> TranscriptWindow:
    if max_age_ms <= 0:
        raise ValueError("max_age_ms must be positive")
    if not segments:
        return TranscriptWindow.from_segments(session_id, [])
    latest_end = max(segment.end_ms for segment in segments)
    cutoff = max(0, latest_end - max_age_ms)
    return TranscriptWindow.from_segments(
        session_id,
        [segment for segment in segments if segment.end_ms >= cutoff],
    )


def recent_window_by_chars(
    session_id: str,
    segments: list[TranscriptSegmentRecord],
    *,
    max_chars: int,
) -> TranscriptWindow:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    selected: list[TranscriptSegmentRecord] = []
    total = 0
    for segment in reversed(segments):
        segment_chars = len(segment.text)
        if selected and total + segment_chars > max_chars:
            break
        selected.append(segment)
        total += segment_chars
        if total >= max_chars:
            break
    selected.reverse()
    return TranscriptWindow.from_segments(session_id, selected)
