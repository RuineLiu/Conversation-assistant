"""End-to-end test for B2: unknown-term explanation closes the loop into memory.

Flow under test::

    transcript "GMV" 出现
        -> UnknownTermDetector (LLM) 给出 explanation
        -> PromptOpportunityDetector emits CONCEPT_EXPLANATION opportunity
        -> PromptOrchestrator fast-paths into PromptCandidate (no second LLM)
        -> ProductAssistantService logs decision
        -> ExplanationMemoryWriter commits a session-scope MEETING_FACT
        -> PersonalVocabularyService sees "GMV" in explained_in_session
        -> Next transcript turn carrying "GMV" feeds explained_in_session to LLM
"""

from typing import Any

from proactive_assistant.detection import (
    PromptOpportunityDetector,
    UnknownTermDetector,
)
from proactive_assistant.detection.vocabulary import PersonalVocabularyService
from proactive_assistant.memory import InMemoryMemoryStore, MemoryQuery, MemoryService
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.product import ProductAssistantService
from proactive_assistant.prompting import (
    PromptGenerationService,
    RuleBasedPromptGenerationService,
)
from proactive_assistant.runtime import PromptRuntimeService
from proactive_assistant.sessions import (
    InMemorySessionStore,
    SessionConfig,
    SessionService,
    TranscriptSegmentInput,
)


def _unknown_term_response(term: str, explanation: str, source_segment_id: str) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "term": term,
                "term_type": "acronym",
                "explanation": explanation,
                "confidence": 0.86,
                "privacy_level": "low",
                "privacy_risk": 0.05,
                "source_segment_id": source_segment_id,
                "rationale": "电商背景术语，对非电商同事陌生。",
            }
        ],
        "detection_notes": "",
        "safety_flags": [],
    }


def _build_product_service(
    *,
    detector_responses: list[dict[str, Any]],
):  # type: ignore[no-untyped-def]
    memory_service = MemoryService(InMemoryMemoryStore())

    # The detector's FakeModelClient cycles through responses by call order.
    iter_responses = iter(detector_responses)

    def detector_response(_request):
        try:
            return next(iter_responses)
        except StopIteration:
            return {"candidates": [], "detection_notes": "no more", "safety_flags": []}

    detector_client = FakeModelClient(detector_response)
    unknown_term_detector = UnknownTermDetector(
        model_client=detector_client,
        settings=ModelGatewaySettings(default_model="gpt-test", max_output_tokens=512),
    )
    vocabulary_service = PersonalVocabularyService(memory_service)
    opportunity_detector = PromptOpportunityDetector(
        max_opportunities=5,
        unknown_term_detector=unknown_term_detector,
        vocabulary_service=vocabulary_service,
    )
    # PromptGenerationService client should never be called when the fast
    # path triggers. If it ever is, the test will assert on its requests.
    prompt_client = FakeModelClient({})
    prompt_service = PromptGenerationService(
        model_client=prompt_client,
        settings=ModelGatewaySettings(default_model="gpt-test", max_output_tokens=512),
    )
    orchestrator = PromptOrchestrator(
        prompt_service=prompt_service,
        fallback_prompt_service=RuleBasedPromptGenerationService(),
        detector=opportunity_detector,
    )
    product = ProductAssistantService(
        session_service=SessionService(InMemorySessionStore()),
        prompt_orchestrator=orchestrator,
        runtime_service=PromptRuntimeService(),
        memory_service=memory_service,
    )
    return product, memory_service, detector_client, prompt_client


def test_first_mention_writes_explanation_memory_and_second_mention_skips_llm() -> None:
    """The full B2 contract.

    First transcript turn carrying "GMV" triggers detector → product
    flow writes a session-scope memory record. Second turn carrying
    "GMV" runs the detector again, but the LLM now sees the term in
    explained_in_session and returns empty; the product flow emits no
    new candidate for it, AND the original memory record is not
    duplicated.
    """

    # First call: detector returns GMV. Second call: detector returns
    # nothing because the term is now in explained_in_session.
    product, memory_service, detector_client, prompt_client = _build_product_service(
        detector_responses=[
            _unknown_term_response(
                term="GMV",
                explanation="商品交易总额，电商核心指标。",
                source_segment_id="seg_0",
            ),
            {"candidates": [], "detection_notes": "explained already", "safety_flags": []},
        ],
    )
    session = product.create_session(
        SessionConfig(title="Q3 review", metadata={"org_id": "org_001", "subject_user_id": "user_001"}),
        session_id="session_001",
    )

    step_one = product.append_transcript_and_generate_prompts(
        session.session_id,
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=900,
            text="我们这季度 GMV 增长了 20%。",
            asr_confidence=0.94,
        ),
        segment_id="seg_0",
    )

    # The fast-path candidate produced a SHOWN decision carrying the
    # explanation, and no PromptGenerationService call was made.
    concept_prompts = [
        prompt for prompt in step_one.prompts if prompt.prompt_category == "concept_explanation"
    ]
    assert len(concept_prompts) == 1
    payload = concept_prompts[0]
    assert payload.glasses_title == "GMV"
    assert payload.glasses_text == "商品交易总额，电商核心指标。"
    assert prompt_client.requests == []  # second LLM call skipped

    # Memory got a session-scope MEETING_FACT tagged term_explanation.
    listed = memory_service.store.list_memories(
        MemoryQuery(session_id="session_001", limit=20)
    )
    explanation_records = [
        record for record in listed
        if "term_explanation" in record.tags
    ]
    assert len(explanation_records) == 1
    record = explanation_records[0]
    assert record.metadata["canonical_entity"] == "GMV"
    assert record.text == "商品交易总额，电商核心指标。"
    # Vocabulary now picks up GMV in explained_in_session.
    vocab = PersonalVocabularyService(memory_service)
    assert "GMV" in vocab.get_explained_terms_for_session(session_id="session_001")

    # Second turn carrying GMV: detector LLM returns nothing because the
    # term is in explained_in_session. No new concept_explanation prompt
    # gets produced.
    step_two = product.append_transcript_and_generate_prompts(
        session.session_id,
        TranscriptSegmentInput(
            speaker="Alex",
            start_ms=1_000,
            end_ms=2_000,
            text="我看了下 GMV 走势图，下季度还会涨。",
            asr_confidence=0.94,
        ),
        segment_id="seg_1",
    )

    second_turn_concepts = [
        prompt for prompt in step_two.prompts if prompt.prompt_category == "concept_explanation"
    ]
    assert second_turn_concepts == []

    # Detector was called twice (once per turn). Both calls were issued.
    assert len(detector_client.requests) == 2

    # Memory still has exactly one explanation record (idempotency).
    listed_after = memory_service.store.list_memories(
        MemoryQuery(session_id="session_001", limit=20)
    )
    explanation_records_after = [
        record for record in listed_after
        if "term_explanation" in record.tags
    ]
    assert len(explanation_records_after) == 1
    assert explanation_records_after[0].memory_id == record.memory_id


def test_memory_write_failure_does_not_break_realtime_prompt_flow() -> None:
    """If the writer raises, the candidate should still ship to the user.

    Demo path tolerance: a memory-side hiccup must not crash the
    transcript step.
    """

    product, memory_service, detector_client, _ = _build_product_service(
        detector_responses=[
            _unknown_term_response(
                term="GMV",
                explanation="商品交易总额。",
                source_segment_id="seg_0",
            ),
        ],
    )
    # Sabotage the writer: make commit_explanation always raise.
    class _ExplodingWriter:
        def commit_explanation(self, **_kwargs):
            raise RuntimeError("synthetic memory store failure")

    product.explanation_writer = _ExplodingWriter()
    session = product.create_session(
        SessionConfig(title="Q3 review", metadata={"org_id": "org_001", "subject_user_id": "user_001"}),
        session_id="session_001",
    )

    step = product.append_transcript_and_generate_prompts(
        session.session_id,
        TranscriptSegmentInput(
            speaker="Bao",
            start_ms=0,
            end_ms=900,
            text="我们这季度 GMV 增长了 20%。",
            asr_confidence=0.94,
        ),
        segment_id="seg_0",
    )

    concept_prompts = [
        prompt for prompt in step.prompts if prompt.prompt_category == "concept_explanation"
    ]
    assert len(concept_prompts) == 1
    assert concept_prompts[0].glasses_text == "商品交易总额。"
    # No memory record was persisted.
    listed = memory_service.store.list_memories(
        MemoryQuery(session_id="session_001", limit=10)
    )
    assert listed == []
