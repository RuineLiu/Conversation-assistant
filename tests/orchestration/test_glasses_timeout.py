"""P2-3: glasses-bound prompt generation enforces a hard deadline.

When the model takes longer than ``glasses_prompt_timeout_seconds`` to
respond, the orchestrator returns a SUPPRESSED candidate with
``safety_flag="llm_timeout"`` instead of letting a stale popup land on
the wearable.

App surfaces are not affected because the demo path explicitly accepts
longer waits when the user opened a tab on their own.
"""

import time
from typing import Any

from proactive_assistant.detection import (
    CandidateTimingAction,
    DetectionRuleMatch,
    PromptOpportunity,
    PromptPriority,
)
from proactive_assistant.model_gateway import (
    FakeModelClient,
    ModelGatewayTimeoutError,
    ModelRequest,
    ModelResponse,
)
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.prompting import (
    ContentGranularity,
    PRDSurface,
    PrivacyLevel,
    PromptCategory,
    PromptGenerationService,
)
from proactive_assistant.schemas.scenario import ActivityPhase
from proactive_assistant.sessions import (
    InMemorySessionStore,
    SessionConfig,
    SessionService,
    TranscriptSegmentInput,
)


class SlowFakeModelClient:
    """Fake client that sleeps ``sleep_seconds`` before each response."""

    def __init__(self, response: dict[str, Any], sleep_seconds: float) -> None:
        self._inner = FakeModelClient(response)
        self._sleep = sleep_seconds
        self.requests: list[ModelRequest] = []

    def generate_structured(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        time.sleep(self._sleep)
        return self._inner.generate_structured(request)


def _make_snapshot(text: str = "这个问题谁负责？"):  # type: ignore[no-untyped-def]
    service = SessionService(InMemorySessionStore())
    service.create_session(SessionConfig(title="Risk sync"), session_id="session_001")
    service.append_transcript(
        "session_001",
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=900,
            text=text,
            asr_confidence=0.93,
        ),
        segment_id="seg_0",
    )
    return service.get_context_snapshot("session_001")


def _make_glasses_opportunity() -> PromptOpportunity:
    return PromptOpportunity(
        opportunity_id="opp_001",
        session_id="session_001",
        trigger_segment_ids=["seg_0"],
        captured_text="负责人？",
        prompt_category=PromptCategory.QUESTION_ANSWER,
        activity_phase=ActivityPhase.IN_ACTIVITY,
        candidate_timing_action=CandidateTimingAction.DURING_ACTIVITY,
        suggested_content_granularity=ContentGranularity.ONE_LINE_ANSWER,
        priority=PromptPriority.P1,
        confidence=0.7,
        privacy_level=PrivacyLevel.LOW,
        privacy_risk=0.1,
        reason="test",
        rule_matches=[
            DetectionRuleMatch(rule_name="q", matched_terms=["?"], confidence_delta=0.0, reason="t")
        ],
    )


def _valid_response() -> dict[str, Any]:
    return {
        "should_prompt": True,
        "prompt_category": "question_answer",
        "content_granularity": 2,
        "glasses_title": "负责人",
        "glasses_text": "请确认负责人。",
        "app_detail_text": "需要明确负责人。",
        "source_refs": ["transcript:seg_0"],
        "confidence": 0.8,
        "privacy_level": "low",
        "privacy_risk": 0.1,
        "rationale": "test",
        "safety_flags": [],
    }


def test_glasses_prompt_times_out_when_model_slower_than_deadline() -> None:
    slow_client = SlowFakeModelClient(_valid_response(), sleep_seconds=0.3)
    prompt_service = PromptGenerationService(
        model_client=slow_client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    orchestrator = PromptOrchestrator(
        prompt_service=prompt_service,
        glasses_prompt_timeout_seconds=0.05,
    )

    candidate = orchestrator.generate_candidate(_make_snapshot(), _make_glasses_opportunity())

    assert candidate.status == "suppressed"
    assert candidate.prompt_result is not None
    assert candidate.prompt_result.should_prompt is False
    assert "llm_timeout" in candidate.prompt_result.safety_flags
    assert candidate.metadata["safety_flag_reason"] == "llm_timeout"
    assert candidate.metadata["timeout_seconds"] == 0.05


def test_glasses_prompt_passes_when_model_under_deadline() -> None:
    quick_client = SlowFakeModelClient(_valid_response(), sleep_seconds=0.01)
    prompt_service = PromptGenerationService(
        model_client=quick_client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    orchestrator = PromptOrchestrator(
        prompt_service=prompt_service,
        glasses_prompt_timeout_seconds=1.0,
    )

    candidate = orchestrator.generate_candidate(_make_snapshot(), _make_glasses_opportunity())

    assert candidate.status == "generated"
    assert candidate.prompt_result is not None
    assert candidate.prompt_result.should_prompt is True
    assert "llm_timeout" not in (candidate.prompt_result.safety_flags or [])


def test_app_surface_opportunity_ignores_glasses_timeout() -> None:
    """App-surface generations should not be bound by the glasses deadline."""

    slow_client = SlowFakeModelClient(_valid_response(), sleep_seconds=0.3)
    prompt_service = PromptGenerationService(
        model_client=slow_client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    orchestrator = PromptOrchestrator(
        prompt_service=prompt_service,
        glasses_prompt_timeout_seconds=0.05,
    )
    opportunity = _make_glasses_opportunity().model_copy(
        update={
            "candidate_timing_action": CandidateTimingAction.AFTER_ACTIVITY.value,
            "priority": PromptPriority.P0.value,
        }
    )

    candidate = orchestrator.generate_candidate(_make_snapshot(), opportunity)

    assert candidate.status == "generated"
    assert "llm_timeout" not in (candidate.prompt_result.safety_flags or [])
