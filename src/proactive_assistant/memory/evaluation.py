from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.memory.extraction import MemoryExtractionRequest, MemoryExtractionService
from proactive_assistant.model_gateway import FakeModelClient, ModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import ModelUsageMetadata, TranscriptWindowItem
from proactive_assistant.runtime import MemoryCandidate


class MemoryExtractionExpectedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_type: str
    text_contains: list[str] = Field(default_factory=list)
    owner: str = ""
    deadline: str = ""
    status: str = ""
    entity_contains: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    write_policy: str | None = None


class MemoryExtractionFixtureCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    description: str = ""
    locale: str = "zh-CN"
    transcript_window: list[TranscriptWindowItem] = Field(min_length=1)
    session_context: dict[str, Any] = Field(default_factory=dict)
    meeting_state: dict[str, Any] = Field(default_factory=dict)
    privacy_constraints: list[str] = Field(default_factory=list)
    max_candidates: int = Field(default=8, ge=1, le=20)
    expected_candidates: list[MemoryExtractionExpectedCandidate] = Field(default_factory=list)
    forbidden_text_contains: list[str] = Field(default_factory=list)
    expected_min_candidates: int = Field(default=0, ge=0)
    fixture_model_output: dict[str, Any] | None = None


class MemoryExtractionCandidateEval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_candidate_id: str
    candidate_type: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    write_policy: str
    owner: str = ""
    deadline: str = ""
    status: str = ""
    entity: str = ""
    source_refs: list[str] = Field(default_factory=list)


class MemoryExtractionExpectedCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_index: int
    passed: bool
    reason: str
    matched_candidate_id: str | None = None


class MemoryExtractionCaseEvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    passed: bool
    candidate_count: int
    candidates: list[MemoryExtractionCandidateEval] = Field(default_factory=list)
    expected_checks: list[MemoryExtractionExpectedCheckResult] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    extraction_notes: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    model_usage: ModelUsageMetadata | None = None


class MemoryExtractionEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    mode: str
    case_count: int
    pass_count: int
    failed_count: int
    candidate_count: int
    failed_cases: list[str] = Field(default_factory=list)
    results: list[MemoryExtractionCaseEvalResult] = Field(default_factory=list)


def load_memory_extraction_fixture_cases(path: Path) -> list[MemoryExtractionFixtureCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
    else:
        cases = payload
    if not isinstance(cases, list):
        raise ValueError("memory extraction fixture file must contain a list or {cases: [...]}")
    return [MemoryExtractionFixtureCase.model_validate(item) for item in cases]


def evaluate_memory_extraction_fixtures(
    cases: list[MemoryExtractionFixtureCase],
    *,
    mode: str = "fixture",
    model_client: ModelClient | None = None,
    settings: ModelGatewaySettings | None = None,
    model: str | None = None,
) -> MemoryExtractionEvaluationReport:
    results = [
        evaluate_memory_extraction_case(
            case,
            mode=mode,
            model_client=model_client,
            settings=settings,
            model=model,
        )
        for case in cases
    ]
    pass_count = sum(1 for result in results if result.passed)
    candidate_count = sum(result.candidate_count for result in results)
    failed_cases = [result.case_id for result in results if not result.passed]
    return MemoryExtractionEvaluationReport(
        ok=pass_count == len(results),
        mode=mode,
        case_count=len(results),
        pass_count=pass_count,
        failed_count=len(results) - pass_count,
        candidate_count=candidate_count,
        failed_cases=failed_cases,
        results=results,
    )


def evaluate_memory_extraction_case(
    case: MemoryExtractionFixtureCase,
    *,
    mode: str = "fixture",
    model_client: ModelClient | None = None,
    settings: ModelGatewaySettings | None = None,
    model: str | None = None,
) -> MemoryExtractionCaseEvalResult:
    client = _client_for_case(case, mode=mode, model_client=model_client)
    service = MemoryExtractionService(model_client=client, settings=settings)
    extraction = service.extract_candidates(_request_for_case(case), model=model)
    candidates = [_candidate_eval(candidate) for candidate in extraction.candidates]
    expected_checks = [_check_expected_candidate(index, expected, candidates) for index, expected in enumerate(case.expected_candidates)]
    failed_checks = [check.reason for check in expected_checks if not check.passed]
    if len(candidates) < case.expected_min_candidates:
        failed_checks.append(f"candidate_count_below_expected_min:{len(candidates)}<{case.expected_min_candidates}")
    for forbidden in case.forbidden_text_contains:
        if any(forbidden in candidate.text for candidate in candidates):
            failed_checks.append(f"forbidden_text_present:{forbidden}")
    return MemoryExtractionCaseEvalResult(
        case_id=case.case_id,
        passed=not failed_checks,
        candidate_count=len(candidates),
        candidates=candidates,
        expected_checks=expected_checks,
        failed_checks=failed_checks,
        extraction_notes=extraction.extraction_notes,
        safety_flags=list(extraction.safety_flags),
        model_usage=extraction.model_usage,
    )


def _client_for_case(
    case: MemoryExtractionFixtureCase,
    *,
    mode: str,
    model_client: ModelClient | None,
) -> ModelClient:
    if model_client is not None:
        return model_client
    if mode == "fixture":
        if case.fixture_model_output is None:
            raise ValueError(f"fixture mode requires fixture_model_output for case: {case.case_id}")
        return FakeModelClient(case.fixture_model_output)
    raise ValueError("live mode requires an injected model_client")


def _request_for_case(case: MemoryExtractionFixtureCase) -> MemoryExtractionRequest:
    return MemoryExtractionRequest(
        session_id=f"eval_{case.case_id}",
        scenario_id="meeting_business",
        locale=case.locale,
        transcript_window=case.transcript_window,
        session_context=case.session_context,
        meeting_state=case.meeting_state,
        privacy_constraints=case.privacy_constraints,
        max_candidates=case.max_candidates,
    )


def _candidate_eval(candidate: MemoryCandidate) -> MemoryExtractionCandidateEval:
    metadata = candidate.metadata
    return MemoryExtractionCandidateEval(
        memory_candidate_id=candidate.memory_candidate_id,
        candidate_type=str(candidate.candidate_type),
        text=candidate.text,
        confidence=candidate.confidence,
        write_policy=str(candidate.write_policy),
        owner=str(metadata.get("owner", "")),
        deadline=str(metadata.get("deadline", "")),
        status=str(metadata.get("status", "")),
        entity=str(metadata.get("entity", metadata.get("canonical_entity", ""))),
        source_refs=[str(item) for item in metadata.get("source_refs", [])],
    )


def _check_expected_candidate(
    index: int,
    expected: MemoryExtractionExpectedCandidate,
    candidates: list[MemoryExtractionCandidateEval],
) -> MemoryExtractionExpectedCheckResult:
    for candidate in candidates:
        if _candidate_matches(expected, candidate):
            return MemoryExtractionExpectedCheckResult(
                expected_index=index,
                passed=True,
                reason="matched",
                matched_candidate_id=candidate.memory_candidate_id,
            )
    return MemoryExtractionExpectedCheckResult(
        expected_index=index,
        passed=False,
        reason=f"missing_expected_candidate:{expected.candidate_type}",
    )


def _candidate_matches(
    expected: MemoryExtractionExpectedCandidate,
    candidate: MemoryExtractionCandidateEval,
) -> bool:
    if candidate.candidate_type != expected.candidate_type:
        return False
    if expected.write_policy is not None and candidate.write_policy != expected.write_policy:
        return False
    if expected.owner and expected.owner != candidate.owner and expected.owner not in candidate.text:
        return False
    if expected.deadline and expected.deadline != candidate.deadline and expected.deadline not in candidate.text:
        return False
    if expected.status and expected.status != candidate.status:
        return False
    if expected.source_refs and not set(expected.source_refs).issubset(set(candidate.source_refs)):
        return False
    if any(term not in candidate.text for term in expected.text_contains):
        return False
    if any(term not in candidate.entity for term in expected.entity_contains):
        return False
    return True
