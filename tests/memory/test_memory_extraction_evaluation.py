from pathlib import Path

from proactive_assistant.memory import (
    evaluate_memory_extraction_fixtures,
    load_memory_extraction_fixture_cases,
)


FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "memory_extraction" / "cases.json"


def test_load_memory_extraction_fixture_cases() -> None:
    cases = load_memory_extraction_fixture_cases(FIXTURE_PATH)

    assert len(cases) == 5
    assert cases[0].case_id == "action_owner_deadline_zh"
    assert cases[0].expected_candidates[0].owner == "张三"


def test_evaluate_memory_extraction_fixtures_in_fixture_mode() -> None:
    cases = load_memory_extraction_fixture_cases(FIXTURE_PATH)

    report = evaluate_memory_extraction_fixtures(cases)

    assert report.ok is True
    assert report.mode == "fixture"
    assert report.case_count == 5
    assert report.pass_count == 5
    assert report.failed_count == 0
    assert report.candidate_count == 4
    action = report.results[0]
    assert action.passed is True
    assert action.expected_checks[0].matched_candidate_id is not None
    assert action.candidates[0].owner == "张三"
    privacy = next(result for result in report.results if result.case_id == "privacy_sensitive_pricing")
    assert privacy.passed is True
    assert "180 万" not in " ".join(candidate.text for candidate in privacy.candidates)
