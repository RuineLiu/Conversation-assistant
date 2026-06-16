from proactive_assistant.asr import PartialTranscriptAggregator, StreamingSpeechEvent, StreamingSpeechEventType


def test_partial_aggregator_emits_soft_segment_for_trigger_terms() -> None:
    aggregator = PartialTranscriptAggregator(min_chars=4, min_emit_interval_ms=0)

    segment = aggregator.observe(
        StreamingSpeechEvent(
            event_type=StreamingSpeechEventType.PARTIAL_TRANSCRIPT,
            text="这个问题谁负责",
            language="zh-CN",
            confidence=0.82,
        ),
        speaker="Bao",
        fallback_start_ms=0,
        fallback_end_ms=900,
    )

    assert segment is not None
    assert segment.text == "这个问题谁负责"
    assert segment.speaker == "Bao"
    assert segment.reason == "trigger_terms"


def test_partial_aggregator_dedupes_same_partial_text() -> None:
    aggregator = PartialTranscriptAggregator(min_chars=4, min_emit_interval_ms=0)
    event = StreamingSpeechEvent(
        event_type=StreamingSpeechEventType.PARTIAL_TRANSCRIPT,
        text="这个问题谁负责",
        language="zh-CN",
    )

    assert aggregator.observe(event, speaker="Bao", fallback_start_ms=0, fallback_end_ms=800) is not None
    assert aggregator.observe(event, speaker="Bao", fallback_start_ms=0, fallback_end_ms=800) is None


def test_partial_aggregator_ignores_short_non_trigger_partial() -> None:
    aggregator = PartialTranscriptAggregator(min_chars=6, min_emit_interval_ms=0)

    segment = aggregator.observe(
        StreamingSpeechEvent(
            event_type=StreamingSpeechEventType.PARTIAL_TRANSCRIPT,
            text="小张",
            language="zh-CN",
        ),
        speaker="Bao",
        fallback_start_ms=0,
        fallback_end_ms=300,
    )

    assert segment is None


def test_partial_aggregator_does_not_emit_incomplete_filler_continuation() -> None:
    aggregator = PartialTranscriptAggregator(
        min_chars=4,
        min_emit_interval_ms=0,
        max_soft_interval_ms=0,
    )
    aggregator.observe(
        StreamingSpeechEvent(
            event_type=StreamingSpeechEventType.PARTIAL_TRANSCRIPT,
            text="你们谁还记得我们上周说的那个模型？",
            language="zh-CN",
        ),
        speaker="Bao",
        fallback_start_ms=0,
        fallback_end_ms=1000,
    )

    segment = aggregator.observe(
        StreamingSpeechEvent(
            event_type=StreamingSpeechEventType.PARTIAL_TRANSCRIPT,
            text="就是呃，你说的那",
            language="zh-CN",
        ),
        speaker="Bao",
        fallback_start_ms=1200,
        fallback_end_ms=1800,
    )

    assert segment is None
