from typing import Any

from proactive_assistant.detection import (
    UnknownTermDetectionRequest,
    UnknownTermDetector,
    UnknownTermType,
)
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import PrivacyLevel, TranscriptWindowItem


def make_window(text: str = "我们这季度 GMV 增长了 20%。", transcript_id: str = "seg_0") -> list[TranscriptWindowItem]:
    return [
        TranscriptWindowItem(
            transcript_id=transcript_id,
            speaker="Bao",
            text=text,
            timestamp_ms=0,
        )
    ]


def make_detector(response: dict[str, Any]) -> tuple[UnknownTermDetector, FakeModelClient]:
    client = FakeModelClient(response)
    detector = UnknownTermDetector(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test", max_output_tokens=512),
    )
    return detector, client


def test_detector_returns_candidates_with_stable_ids() -> None:
    response = {
        "candidates": [
            {
                "term": "GMV",
                "term_type": "acronym",
                "explanation": "商品交易总额，电商核心指标。",
                "confidence": 0.86,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_0",
                "rationale": "电商术语，对非电商背景同事陌生。",
            }
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, client = make_detector(response)

    result = detector.detect(
        UnknownTermDetectionRequest(
            session_id="session_001",
            transcript_window=make_window(),
        )
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.term == "GMV"
    assert candidate.term_type == UnknownTermType.ACRONYM.value
    assert candidate.explanation == "商品交易总额，电商核心指标。"
    assert candidate.candidate_id.startswith("unkterm_")
    assert candidate.confidence == 0.86
    assert candidate.privacy_level == PrivacyLevel.LOW.value
    assert candidate.source_segment_id == "seg_0"
    assert result.model_usage is not None
    assert result.model_usage.provider == "fake"
    assert len(client.requests) == 1


def test_detector_filters_terms_already_in_known_vocabulary() -> None:
    response = {
        "candidates": [
            {
                "term": "OKR",
                "term_type": "acronym",
                "explanation": "目标与关键结果管理框架。",
                "confidence": 0.8,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_0",
                "rationale": "test",
            },
            {
                "term": "GMV",
                "term_type": "acronym",
                "explanation": "商品交易总额。",
                "confidence": 0.85,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_0",
                "rationale": "test",
            },
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, _ = make_detector(response)

    result = detector.detect(
        UnknownTermDetectionRequest(
            session_id="session_001",
            transcript_window=make_window(),
            known_vocabulary=["okr"],
        )
    )

    assert [c.term for c in result.candidates] == ["GMV"]


def test_detector_filters_terms_already_explained_in_session() -> None:
    response = {
        "candidates": [
            {
                "term": "GMV",
                "term_type": "acronym",
                "explanation": "商品交易总额。",
                "confidence": 0.85,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_0",
                "rationale": "test",
            }
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, _ = make_detector(response)

    result = detector.detect(
        UnknownTermDetectionRequest(
            session_id="session_001",
            transcript_window=make_window(),
            explained_in_session=["GMV"],
        )
    )

    assert result.candidates == []


def test_detector_dedupes_same_term_case_insensitively() -> None:
    response = {
        "candidates": [
            {
                "term": "GMV",
                "term_type": "acronym",
                "explanation": "商品交易总额。",
                "confidence": 0.85,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_0",
                "rationale": "test",
            },
            {
                "term": "gmv",
                "term_type": "acronym",
                "explanation": "电商核心指标。",
                "confidence": 0.6,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_0",
                "rationale": "test",
            },
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, _ = make_detector(response)

    result = detector.detect(
        UnknownTermDetectionRequest(
            session_id="session_001",
            transcript_window=make_window(),
        )
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].term == "GMV"


def test_detector_drops_candidates_with_unknown_source_segment_id() -> None:
    response = {
        "candidates": [
            {
                "term": "GMV",
                "term_type": "acronym",
                "explanation": "商品交易总额。",
                "confidence": 0.85,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_hallucinated",
                "rationale": "test",
            }
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, _ = make_detector(response)

    result = detector.detect(
        UnknownTermDetectionRequest(
            session_id="session_001",
            transcript_window=make_window(transcript_id="seg_0"),
        )
    )

    assert result.candidates == []


def test_detector_respects_max_candidates() -> None:
    response = {
        "candidates": [
            {
                "term": f"TERM{idx}",
                "term_type": "acronym",
                "explanation": f"解释{idx}。",
                "confidence": 0.7,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": "seg_0",
                "rationale": "test",
            }
            for idx in range(5)
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, _ = make_detector(response)

    result = detector.detect(
        UnknownTermDetectionRequest(
            session_id="session_001",
            transcript_window=make_window(),
            max_candidates=2,
        )
    )

    assert len(result.candidates) == 2


def test_detector_returns_empty_when_model_has_no_candidates() -> None:
    detector, _ = make_detector({"candidates": [], "detection_notes": "all known", "safety_flags": []})

    result = detector.detect(
        UnknownTermDetectionRequest(
            session_id="session_001",
            transcript_window=make_window(),
        )
    )

    assert result.candidates == []
    assert result.detection_notes == "all known"
