import pytest
from pydantic import ValidationError

from proactive_assistant.sessions import (
    InMemorySessionStore,
    SessionAlreadyExistsError,
    SessionConfig,
    SessionService,
    SessionStatus,
    TranscriptSegmentInput,
)


def make_service() -> SessionService:
    return SessionService(InMemorySessionStore())


def segment(
    text: str,
    *,
    speaker: str = "Bao",
    start_ms: int = 0,
    end_ms: int = 1000,
) -> TranscriptSegmentInput:
    return TranscriptSegmentInput(
        speaker=speaker,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        asr_confidence=0.92,
    )


def test_create_session_starts_by_default() -> None:
    service = make_service()

    session = service.create_session(
        SessionConfig(
            title="Launch risk sync",
            pre_context="Discuss launch risks and owners.",
            privacy_constraints=["avoid customer names"],
        ),
        session_id="session_001",
    )

    assert session.session_id == "session_001"
    assert session.status == SessionStatus.RUNNING
    assert session.started_at is not None
    assert session.privacy_constraints == ["avoid customer names"]


def test_duplicate_session_id_is_rejected() -> None:
    service = make_service()
    service.create_session(session_id="session_001")

    with pytest.raises(SessionAlreadyExistsError):
        service.create_session(session_id="session_001")


def test_transcript_segment_rejects_invalid_time_range() -> None:
    with pytest.raises(ValidationError):
        segment("Invalid timing", start_ms=1000, end_ms=1000)


def test_append_and_read_transcript_sorted_by_time() -> None:
    service = make_service()
    service.create_session(session_id="session_001")

    service.append_transcript("session_001", segment("Second", start_ms=2000, end_ms=3000), segment_id="seg_002")
    service.append_transcript("session_001", segment("First", start_ms=0, end_ms=1000), segment_id="seg_001")

    transcript = service.get_transcript("session_001")
    assert [item.segment_id for item in transcript] == ["seg_001", "seg_002"]


def test_ended_session_rejects_transcript_append() -> None:
    service = make_service()
    service.create_session(session_id="session_001")
    service.end_session("session_001")

    with pytest.raises(ValueError, match="ended session"):
        service.append_transcript("session_001", segment("Too late"))


def test_recent_window_by_segments() -> None:
    service = make_service()
    service.create_session(session_id="session_001")
    for index in range(5):
        service.append_transcript(
            "session_001",
            segment(f"Segment {index}", start_ms=index * 1000, end_ms=index * 1000 + 500),
            segment_id=f"seg_{index}",
        )

    window = service.get_transcript_window("session_001", max_segments=2)
    assert [item.segment_id for item in window.segments] == ["seg_3", "seg_4"]
    assert window.start_ms == 3000
    assert window.end_ms == 4500


def test_recent_window_by_chars_keeps_recent_whole_segments() -> None:
    service = make_service()
    service.create_session(session_id="session_001")
    service.append_transcript("session_001", segment("alpha", start_ms=0, end_ms=500), segment_id="seg_1")
    service.append_transcript("session_001", segment("beta", start_ms=600, end_ms=900), segment_id="seg_2")
    service.append_transcript("session_001", segment("gamma", start_ms=1000, end_ms=1500), segment_id="seg_3")

    window = service.get_transcript_window("session_001", max_chars=9)
    assert [item.text for item in window.segments] == ["beta", "gamma"]


def test_recent_window_by_time() -> None:
    service = make_service()
    service.create_session(session_id="session_001")
    service.append_transcript("session_001", segment("old", start_ms=0, end_ms=500), segment_id="seg_1")
    service.append_transcript("session_001", segment("new", start_ms=3000, end_ms=3500), segment_id="seg_2")

    window = service.get_transcript_window("session_001", max_age_ms=1000)
    assert [item.segment_id for item in window.segments] == ["seg_2"]


def test_context_snapshot_preserves_memory_placeholders_and_stats() -> None:
    service = make_service()
    service.create_session(
        SessionConfig(pre_context="Weekly launch meeting", privacy_constraints=["no customer data"]),
        session_id="session_001",
    )
    service.append_transcript(
        "session_001",
        segment("Who owns the launch risk follow-up?", speaker="Bao", start_ms=0, end_ms=1200),
        segment_id="seg_1",
    )

    snapshot = service.get_context_snapshot(
        "session_001",
        memory_context=["Last week: Alex owned release notes."],
        memory_refs=["memory:release_notes_owner"],
    )

    assert snapshot.pre_context == "Weekly launch meeting"
    assert snapshot.privacy_constraints == ["no customer data"]
    assert snapshot.memory_refs == ["memory:release_notes_owner"]
    assert snapshot.transcript_stats["segment_count"] == 1
    assert snapshot.transcript_stats["speaker_count"] == 1


def test_build_prompt_generation_request_from_session_context() -> None:
    service = make_service()
    service.create_session(
        SessionConfig(pre_context="Weekly launch meeting", privacy_constraints=["no customer data"]),
        session_id="session_001",
    )
    service.append_transcript(
        "session_001",
        segment("Who owns the launch risk follow-up?", speaker="Bao", start_ms=0, end_ms=1200),
        segment_id="seg_1",
    )

    request = service.build_prompt_generation_request(
        "session_001",
        memory_context=["Last week: Alex owned release notes."],
    )

    assert request.session_id == "session_001"
    assert request.scenario_id == "meeting_business"
    assert request.transcript_window[0].transcript_id == "seg_1"
    assert request.privacy_constraints == ["no customer data"]
    assert request.memory_context == ["Last week: Alex owned release notes."]
    assert request.session_context["pre_context"] == "Weekly launch meeting"


def test_build_prompt_generation_request_requires_transcript() -> None:
    service = make_service()
    service.create_session(session_id="session_001")

    with pytest.raises(ValueError, match="without transcript"):
        service.build_prompt_generation_request("session_001")
