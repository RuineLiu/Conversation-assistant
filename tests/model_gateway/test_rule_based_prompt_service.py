from proactive_assistant.orchestration import build_prompt_generation_request
from proactive_assistant.prompting import RuleBasedPromptGenerationService
from proactive_assistant.sessions import InMemorySessionStore, SessionConfig, SessionService, TranscriptSegmentInput


def make_request(text: str, *, memory_context: list[str] | None = None):  # type: ignore[no-untyped-def]
    service = SessionService(InMemorySessionStore())
    service.create_session(
        SessionConfig(title="Launch risk sync", pre_context="讨论风险、负责人和下一步。"),
        session_id="session_001",
    )
    service.append_transcript(
        "session_001",
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=900,
            text=text,
            asr_confidence=0.94,
        ),
        segment_id="seg_0",
    )
    snapshot = service.get_context_snapshot("session_001", memory_context=memory_context or [])
    opportunity = snapshot.metadata.get("unused")
    assert opportunity is None

    from proactive_assistant.detection import PromptOpportunityDetector

    detected = PromptOpportunityDetector().detect(snapshot)
    return build_prompt_generation_request(snapshot, detected.opportunities[0])


def test_rule_based_prompt_service_generates_gap_check_prompt_without_model() -> None:
    request = make_request("这个问题谁负责，下周五 deadline 前能不能定？")

    result = RuleBasedPromptGenerationService().generate_prompt(request)

    assert result.should_prompt is True
    assert result.prompt_category == "summary_gap_check"
    assert result.content_granularity == 2
    assert result.glasses_title == "待明确"
    assert "负责人" in result.glasses_text
    assert "截止时间" in result.glasses_text
    assert result.source_refs == ["transcript:seg_0"]
    assert result.model_usage is not None
    assert result.model_usage.provider == "rules"


def test_rule_based_prompt_service_does_not_fabricate_missing_fact() -> None:
    request = make_request("腾讯是哪一年成立的？")

    result = RuleBasedPromptGenerationService().generate_prompt(request)

    assert result.prompt_category == "person_or_fact"
    assert result.glasses_title == "需要查证"
    assert "避免编造事实" in result.app_detail_text


def test_rule_based_prompt_service_uses_memory_for_fact_recall() -> None:
    request = make_request("上次会议关于 release notes 的结论是什么？", memory_context=["上次会议：Alex 负责 release notes。"])

    result = RuleBasedPromptGenerationService().generate_prompt(request)

    assert result.prompt_category == "person_or_fact"
    assert result.glasses_title == "可用记忆"
    assert "Alex" in result.glasses_text
