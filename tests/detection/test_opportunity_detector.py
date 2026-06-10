from typing import Any

from proactive_assistant.detection import (
    OpportunityDetectionRequest,
    OpportunityDetector,
)
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import TranscriptWindowItem


def make_window(text: str, transcript_id: str = "seg_0") -> list[TranscriptWindowItem]:
    return [
        TranscriptWindowItem(
            transcript_id=transcript_id,
            speaker="Bao",
            text=text,
            timestamp_ms=0,
        )
    ]


def make_detector(response: dict[str, Any]) -> tuple[OpportunityDetector, FakeModelClient]:
    client = FakeModelClient(response)
    detector = OpportunityDetector(
        model_client=client,
        settings=ModelGatewaySettings(default_model="gpt-test", fast_model="gpt-fast-test"),
    )
    return detector, client


def test_detects_action_missing_deadline_for_ddl_phrasing() -> None:
    response = {
        "opportunities": [
            {
                "prompt_category": "summary_gap_check",
                "gap_type": "action_missing_deadline",
                "captured_text": "小张把项目文档给我，但还没说ddl。",
                "source_segment_id": "seg_0",
                "owner": "小张",
                "deadline": "",
                "entity": "项目文档",
                "priority": "P1",
                "confidence": 0.84,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "rationale": "任务有负责人但缺截止时间(ddl)。",
            }
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, client = make_detector(response)

    result = detector.detect(
        OpportunityDetectionRequest(
            session_id="session_001",
            transcript_window=make_window("小张把项目文档给我吧，ddl你定一下。"),
        )
    )

    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.prompt_category == "summary_gap_check"
    assert c.gap_type == "action_missing_deadline"
    assert c.owner == "小张"
    assert c.entity == "项目文档"
    assert c.candidate_id.startswith("llmopp_")
    assert result.model_usage is not None
    assert len(client.requests) == 1


def test_drops_candidate_with_hallucinated_segment_id() -> None:
    response = {
        "opportunities": [
            {
                "prompt_category": "question_answer",
                "gap_type": "none",
                "captured_text": "这个数据怎么来的？",
                "source_segment_id": "seg_hallucinated",
                "owner": "",
                "deadline": "",
                "entity": "",
                "priority": "P1",
                "confidence": 0.8,
                "privacy_level": "low",
                "privacy_risk": 0.0,
                "rationale": "提问。",
            }
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, _ = make_detector(response)

    result = detector.detect(
        OpportunityDetectionRequest(
            session_id="session_001",
            transcript_window=make_window("这个数据怎么来的？", transcript_id="seg_0"),
        )
    )

    assert result.candidates == []


def test_dedupes_same_segment_category_gap() -> None:
    one = {
        "prompt_category": "summary_gap_check",
        "gap_type": "open_risk",
        "captured_text": "客户报价材料还没完成是个风险。",
        "source_segment_id": "seg_0",
        "owner": "",
        "deadline": "",
        "entity": "客户报价",
        "priority": "P1",
        "confidence": 0.8,
        "privacy_level": "low",
        "privacy_risk": 0.1,
        "rationale": "风险未闭环。",
    }
    response = {
        "opportunities": [one, dict(one, confidence=0.6, rationale="重复。")],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, _ = make_detector(response)

    result = detector.detect(
        OpportunityDetectionRequest(
            session_id="session_001",
            transcript_window=make_window("客户报价材料还没完成。"),
        )
    )

    assert len(result.candidates) == 1


def test_respects_max_candidates() -> None:
    response = {
        "opportunities": [
            {
                "prompt_category": "suggestion",
                "gap_type": "none",
                "captured_text": f"建议点 {i}",
                "source_segment_id": "seg_0",
                "owner": "",
                "deadline": "",
                "entity": f"topic{i}",
                "priority": "P2",
                "confidence": 0.7,
                "privacy_level": "low",
                "privacy_risk": 0.0,
                "rationale": "建议。",
            }
            for i in range(5)
        ],
        "detection_notes": "",
        "safety_flags": [],
    }
    detector, _ = make_detector(response)

    result = detector.detect(
        OpportunityDetectionRequest(
            session_id="session_001",
            transcript_window=make_window("我们接下来怎么推进？"),
            max_candidates=2,
        )
    )

    assert len(result.candidates) == 2


def test_returns_empty_when_no_opportunities() -> None:
    detector, _ = make_detector(
        {"opportunities": [], "detection_notes": "nothing actionable", "safety_flags": []}
    )

    result = detector.detect(
        OpportunityDetectionRequest(
            session_id="session_001",
            transcript_window=make_window("今天天气不错，我们继续吧。"),
        )
    )

    assert result.candidates == []
    assert result.detection_notes == "nothing actionable"
