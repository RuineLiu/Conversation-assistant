"""P1 tests: ``create_default_product_service`` should default-wire the
LLM unknown-term detector + personal vocabulary so the demo runs without
extra setup. ``PROACTIVE_UNKNOWN_TERM_DETECTOR=off`` should disable it
cleanly for environments that want rules-only behavior.
"""

import os
from unittest.mock import patch

from proactive_assistant.detection import UnknownTermDetector
from proactive_assistant.detection.vocabulary import PersonalVocabularyService
from proactive_assistant.memory import MemoryService
from proactive_assistant.memory.query_understanding import QueryUnderstandingService
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import PromptGenerationService
from proactive_assistant.product.api import (
    _build_opportunity_detector,
    _build_prompt_orchestrator,
    _build_query_understanding_service,
    _build_storage_services,
    create_default_product_service,
)


def test_build_opportunity_detector_default_wires_llm_arm_and_vocabulary() -> None:
    fake_client = FakeModelClient({"candidates": [], "detection_notes": "", "safety_flags": []})
    memory_service, _, _ = _make_memory_service_for_test()

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROACTIVE_UNKNOWN_TERM_DETECTOR", None)
        detector = _build_opportunity_detector(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
            memory_service=memory_service,
        )

    # The detector exposes the arms it was built with via private attrs;
    # we check those because the public surface intentionally hides them.
    assert isinstance(detector._unknown_term_detector, UnknownTermDetector)
    assert isinstance(detector._vocabulary_service, PersonalVocabularyService)


def test_build_opportunity_detector_disabled_by_env_var() -> None:
    fake_client = FakeModelClient({"candidates": [], "detection_notes": "", "safety_flags": []})
    memory_service, _, _ = _make_memory_service_for_test()

    with patch.dict(os.environ, {"PROACTIVE_UNKNOWN_TERM_DETECTOR": "off"}):
        detector = _build_opportunity_detector(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
            memory_service=memory_service,
        )

    assert detector._unknown_term_detector is None
    assert detector._vocabulary_service is None


def test_build_opportunity_detector_no_model_client_yields_rules_only_detector() -> None:
    memory_service, _, _ = _make_memory_service_for_test()

    detector = _build_opportunity_detector(
        model_client=None,
        settings=None,
        memory_service=memory_service,
    )

    assert detector._unknown_term_detector is None
    assert detector._vocabulary_service is None


def test_build_opportunity_detector_no_memory_service_runs_without_personalization() -> None:
    """When memory_service is None (e.g. a degraded environment), the LLM
    arm still gets built but with vocabulary_service=None so it works
    without personalization."""

    fake_client = FakeModelClient({"candidates": [], "detection_notes": "", "safety_flags": []})

    with patch.dict(os.environ, {"PROACTIVE_UNKNOWN_TERM_DETECTOR": "on"}):
        detector = _build_opportunity_detector(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
            memory_service=None,
        )

    assert isinstance(detector._unknown_term_detector, UnknownTermDetector)
    assert detector._vocabulary_service is None


def test_build_storage_services_memory_backend_returns_memory_service() -> None:
    """P1 required _build_storage_services to always return a MemoryService
    so the orchestrator can wire vocabulary. Verify that contract holds."""

    with patch.dict(os.environ, {"PROACTIVE_STORAGE_BACKEND": "memory"}):
        _, _, memory_service = _build_storage_services()

    assert isinstance(memory_service, MemoryService)


def test_build_query_understanding_service_default_enabled() -> None:
    fake_client = FakeModelClient(
        {
            "intent": "open_recall",
            "target_entity": "",
            "target_entity_confidence": 0.0,
            "anaphora_resolved": False,
            "time_window_start": "",
            "time_window_end": "",
            "time_is_relative": False,
            "referenced_speaker": "",
            "confidence": 0.9,
            "rationale": "test",
        }
    )

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROACTIVE_QUERY_UNDERSTANDING", None)
        os.environ.pop("OPENAI_FAST_MODEL", None)
        os.environ.pop("PROACTIVE_OPENAI_FAST_MODEL", None)
        service = _build_query_understanding_service(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
        )

    assert isinstance(service, QueryUnderstandingService)
    assert service._settings.fast_model == "gpt-test"


def test_build_query_understanding_service_disabled_by_env_var() -> None:
    fake_client = FakeModelClient(
        {
            "intent": "open_recall",
            "target_entity": "",
            "target_entity_confidence": 0.0,
            "anaphora_resolved": False,
            "time_window_start": "",
            "time_window_end": "",
            "time_is_relative": False,
            "referenced_speaker": "",
            "confidence": 0.9,
            "rationale": "test",
        }
    )

    with patch.dict(os.environ, {"PROACTIVE_QUERY_UNDERSTANDING": "off"}):
        service = _build_query_understanding_service(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
        )

    assert service is None


def test_build_storage_services_threads_query_understanding_into_memory_service() -> None:
    query_understanding = QueryUnderstandingService(
        model_client=FakeModelClient(
            {
                "intent": "open_recall",
                "target_entity": "",
                "target_entity_confidence": 0.0,
                "anaphora_resolved": False,
                "time_window_start": "",
                "time_window_end": "",
                "time_is_relative": False,
                "referenced_speaker": "",
                "confidence": 0.9,
                "rationale": "test",
            }
        ),
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )

    with patch.dict(os.environ, {"PROACTIVE_STORAGE_BACKEND": "memory"}):
        _, _, memory_service = _build_storage_services(query_understanding=query_understanding)

    assert memory_service.query_understanding is query_understanding


def test_build_prompt_orchestrator_threads_detector_through_to_orchestrator() -> None:
    fake_client = FakeModelClient({"candidates": [], "detection_notes": "", "safety_flags": []})
    prompt_service = PromptGenerationService(
        model_client=fake_client,
        settings=ModelGatewaySettings(default_model="gpt-test"),
    )
    memory_service, _, _ = _make_memory_service_for_test()

    with patch.dict(os.environ, {"PROACTIVE_UNKNOWN_TERM_DETECTOR": "on"}):
        os.environ.pop("OPENAI_FAST_MODEL", None)
        os.environ.pop("PROACTIVE_OPENAI_FAST_MODEL", None)
        orchestrator = _build_prompt_orchestrator(
            prompt_service=prompt_service,
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
            memory_service=memory_service,
        )

    assert isinstance(orchestrator._detector._unknown_term_detector, UnknownTermDetector)
    assert isinstance(orchestrator._detector._vocabulary_service, PersonalVocabularyService)
    assert orchestrator._detector._unknown_term_detector._settings.fast_model == "gpt-test"


def test_create_default_product_service_in_memory_mode_has_unknown_term_arm() -> None:
    """End-to-end check of the demo entry point: with default env, the
    constructed service should have an LLM-arm-equipped detector."""

    with patch.dict(
        os.environ,
        {
            "PROACTIVE_STORAGE_BACKEND": "memory",
            "PROACTIVE_PROMPT_MODE": "llm_with_rule_fallback",
        },
    ):
        os.environ.pop("PROACTIVE_UNKNOWN_TERM_DETECTOR", None)
        service = create_default_product_service(
            ModelGatewaySettings(default_model="gpt-test", openai_api_key="sk-test"),
        )

    detector = service.prompt_orchestrator._detector
    assert isinstance(detector._unknown_term_detector, UnknownTermDetector)
    assert isinstance(detector._vocabulary_service, PersonalVocabularyService)
    # And the vocabulary service must back onto the SAME memory store
    # the product service is using, so the B2 explanation-writer loop
    # is visible to the next detection cycle.
    assert detector._vocabulary_service._memory_service is service.memory


def test_create_default_product_service_in_memory_mode_has_query_understanding() -> None:
    with patch.dict(
        os.environ,
        {
            "PROACTIVE_STORAGE_BACKEND": "memory",
            "PROACTIVE_PROMPT_MODE": "llm_with_rule_fallback",
        },
    ):
        os.environ.pop("PROACTIVE_QUERY_UNDERSTANDING", None)
        service = create_default_product_service(
            ModelGatewaySettings(default_model="gpt-test", openai_api_key="sk-test"),
        )

    assert isinstance(service.memory.query_understanding, QueryUnderstandingService)


def _make_memory_service_for_test() -> tuple[MemoryService, None, None]:
    from proactive_assistant.memory import InMemoryMemoryStore

    return MemoryService(InMemoryMemoryStore()), None, None


def test_build_opportunity_detector_default_wires_llm_opportunity_arm() -> None:
    from proactive_assistant.detection import OpportunityDetector

    fake_client = FakeModelClient({"opportunities": [], "detection_notes": "", "safety_flags": []})
    memory_service, _, _ = _make_memory_service_for_test()

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROACTIVE_OPPORTUNITY_DETECTOR", None)
        detector = _build_opportunity_detector(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
            memory_service=memory_service,
        )

    assert isinstance(detector._opportunity_detector, OpportunityDetector)
    assert detector._rule_first_detection is True


def test_build_opportunity_detector_rule_first_can_be_disabled() -> None:
    fake_client = FakeModelClient({"opportunities": [], "detection_notes": "", "safety_flags": []})
    memory_service, _, _ = _make_memory_service_for_test()

    with patch.dict(os.environ, {"PROACTIVE_RULE_FIRST_DETECTION": "off"}):
        detector = _build_opportunity_detector(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
            memory_service=memory_service,
        )

    assert detector._rule_first_detection is False


def test_build_opportunity_detector_opportunity_arm_disabled_by_env() -> None:
    fake_client = FakeModelClient({"opportunities": [], "detection_notes": "", "safety_flags": []})
    memory_service, _, _ = _make_memory_service_for_test()

    with patch.dict(os.environ, {"PROACTIVE_OPPORTUNITY_DETECTOR": "off"}):
        detector = _build_opportunity_detector(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
            memory_service=memory_service,
        )

    assert detector._opportunity_detector is None


def test_build_opportunity_detector_arms_are_independent() -> None:
    """Unknown-term off but opportunity on: only the opportunity arm builds."""
    from proactive_assistant.detection import OpportunityDetector

    fake_client = FakeModelClient({"opportunities": [], "detection_notes": "", "safety_flags": []})
    memory_service, _, _ = _make_memory_service_for_test()

    with patch.dict(
        os.environ,
        {"PROACTIVE_UNKNOWN_TERM_DETECTOR": "off", "PROACTIVE_OPPORTUNITY_DETECTOR": "on"},
    ):
        detector = _build_opportunity_detector(
            model_client=fake_client,
            settings=ModelGatewaySettings(default_model="gpt-test"),
            memory_service=memory_service,
        )

    assert detector._unknown_term_detector is None
    assert isinstance(detector._opportunity_detector, OpportunityDetector)
