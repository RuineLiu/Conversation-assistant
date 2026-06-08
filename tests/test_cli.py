import json
from pathlib import Path

import pytest

from proactive_assistant.cli import main
from proactive_assistant.io import read_jsonl
from proactive_assistant.model_gateway.smoke import (
    EmbeddingSmokeTestResult,
    MemoryCompressionSmokeTestResult,
    MemoryExtractionSmokeTestResult,
    PromptSmokeTestResult,
)
from proactive_assistant.schemas.persona import Persona


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "persona"
    / "sample_persona_records.jsonl"
)


def test_normalize_personas_command_writes_internal_persona_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "personas.normalized.jsonl"

    exit_code = main(
        [
            "normalize-personas",
            "--input",
            str(FIXTURE_PATH),
            "--output",
            str(output),
        ]
    )

    records = list(read_jsonl(output))
    personas = [Persona.model_validate(record) for record in records]
    assert exit_code == 0
    assert [persona.source_persona_id for persona in personas] == [
        "persona_alpha",
        "persona_beta",
    ]


def test_enrich_personas_command_writes_enriched_jsonl_and_report(tmp_path: Path) -> None:
    normalized = tmp_path / "personas.normalized.jsonl"
    enriched = tmp_path / "personas.enriched.rule_based.jsonl"
    report = tmp_path / "persona_enrichment_coverage.json"

    main(["normalize-personas", "--input", str(FIXTURE_PATH), "--output", str(normalized)])
    exit_code = main(
        [
            "enrich-personas",
            "--input",
            str(normalized),
            "--output",
            str(enriched),
            "--report",
            str(report),
        ]
    )

    personas = [Persona.model_validate(record) for record in read_jsonl(enriched)]
    report_payload = json.loads(report.read_text())
    assert exit_code == 0
    assert personas[1].proactive_preferences.in_activity_tolerance.source.value == "rule_based"
    assert report_payload["total_personas"] == 2
    assert "in_activity_tolerance" in report_payload["field_stats"]


def test_merge_llm_proposals_command_writes_merged_jsonl_and_report(tmp_path: Path) -> None:
    normalized = tmp_path / "personas.normalized.jsonl"
    proposals = tmp_path / "llm_proposals.jsonl"
    merged = tmp_path / "personas.enriched.llm_merged.jsonl"
    report = tmp_path / "persona_enrichment_coverage.json"

    main(["normalize-personas", "--input", str(FIXTURE_PATH), "--output", str(normalized)])
    proposals.write_text(
        json.dumps(
            {
                "persona_id": "P_persona_beta",
                "schema_version": "llm_enrichment_contract_v0",
                "proposals": [
                    {
                        "field_name": "in_activity_tolerance",
                        "value": 0.24,
                        "confidence": 0.72,
                        "evidence": ["dislikes interruption while studying"],
                        "rationale": "The persona prefers uninterrupted focus during study.",
                    }
                ],
                "model_metadata": {"model": "fixture-model"},
            }
        )
        + "\n"
    )

    exit_code = main(
        [
            "merge-llm-proposals",
            "--personas",
            str(normalized),
            "--proposals",
            str(proposals),
            "--output",
            str(merged),
            "--report",
            str(report),
        ]
    )

    personas = [Persona.model_validate(record) for record in read_jsonl(merged)]
    assert exit_code == 0
    assert personas[1].proactive_preferences.in_activity_tolerance.source.value == "llm_inferred"
    assert json.loads(report.read_text())["source_totals"]["llm_inferred"] > 0


def test_report_personas_command_writes_report(tmp_path: Path) -> None:
    normalized = tmp_path / "personas.normalized.jsonl"
    report = tmp_path / "persona_enrichment_coverage.json"

    main(["normalize-personas", "--input", str(FIXTURE_PATH), "--output", str(normalized)])
    exit_code = main(["report-personas", "--input", str(normalized), "--output", str(report)])

    payload = json.loads(report.read_text())
    assert exit_code == 0
    assert payload["total_personas"] == 2
    assert "pending_totals" in payload


def test_smoke_openai_prompt_command_prints_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured = {}

    def fake_smoke_runner(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return PromptSmokeTestResult(
            ok=True,
            provider="fake",
            model=kwargs["model"],
            latency_ms=3,
            should_prompt=True,
            prompt_category="summary_gap_check",
            content_granularity=2,
            glasses_title="负责人待确认",
            glasses_text="这个风险还没有明确 owner。",
            source_refs=["transcript:transcript_smoke_001"],
            confidence=0.84,
            privacy_level="low",
            privacy_risk=0.08,
        )

    monkeypatch.setattr("proactive_assistant.cli.run_openai_prompt_smoke_test", fake_smoke_runner)

    exit_code = main(
        [
            "smoke-openai-prompt",
            "--model",
            "gpt-test",
            "--base-url",
            "http://127.0.0.1:58081",
            "--api-style",
            "chat_completions",
            "--chat-response-format",
            "json_schema",
            "--max-output-tokens",
            "300",
            "--timeout-seconds",
            "9",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["model"] == "gpt-test"
    assert payload["prompt_category"] == "summary_gap_check"
    assert captured["model"] == "gpt-test"
    assert captured["settings"].openai_base_url == "http://127.0.0.1:58081"
    assert captured["settings"].model_api_style == "chat_completions"
    assert captured["settings"].chat_response_format == "json_schema"
    assert captured["settings"].max_output_tokens == 300
    assert captured["settings"].request_timeout_seconds == 9


def test_smoke_openai_memory_extraction_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = {}

    def fake_smoke_runner(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return MemoryExtractionSmokeTestResult(
            ok=True,
            provider="fake",
            model=kwargs["model"],
            latency_ms=4,
            candidate_count=2,
            quality_gate_passed=True,
            quality_checks={
                "has_candidates": True,
                "has_action_item": True,
                "has_owner_zhangsan": True,
                "has_deadline_next_friday": True,
                "has_decision": True,
                "has_source_refs": True,
                "no_blocked_candidates": True,
            },
            candidates=[],
        )

    monkeypatch.setattr("proactive_assistant.cli.run_openai_memory_extraction_smoke_test", fake_smoke_runner)

    exit_code = main(
        [
            "smoke-openai-memory-extraction",
            "--model",
            "gpt-test",
            "--base-url",
            "http://127.0.0.1:58081",
            "--api-style",
            "chat_completions",
            "--chat-response-format",
            "json_schema",
            "--max-output-tokens",
            "700",
            "--timeout-seconds",
            "12",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["model"] == "gpt-test"
    assert payload["candidate_count"] == 2
    assert payload["quality_gate_passed"] is True
    assert captured["model"] == "gpt-test"
    assert captured["settings"].openai_base_url == "http://127.0.0.1:58081"
    assert captured["settings"].model_api_style == "chat_completions"
    assert captured["settings"].chat_response_format == "json_schema"
    assert captured["settings"].max_output_tokens == 700
    assert captured["settings"].request_timeout_seconds == 12


def test_smoke_openai_memory_compression_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = {}

    def fake_smoke_runner(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return MemoryCompressionSmokeTestResult(
            ok=True,
            provider="fake",
            model=kwargs["model"],
            latency_ms=5,
            chunk_summary="本段确认负责人、截止时间和法务接口。",
            key_points=["张三负责客户报价确认"],
            open_questions=["是否需要额外审批未确认"],
            candidate_count=2,
            compression_quality=0.82,
            coverage_score=0.78,
            loss_risk_score=0.18,
            source_refs=["transcript:seg_0"],
            quality_gate_passed=True,
            quality_checks={
                "has_summary": True,
                "has_source_refs": True,
                "candidate_refs_are_covered": True,
                "quality_not_low": True,
                "loss_risk_not_high": True,
                "has_memory_candidates": True,
            },
        )

    monkeypatch.setattr("proactive_assistant.cli.run_openai_memory_compression_smoke_test", fake_smoke_runner)

    exit_code = main(
        [
            "smoke-openai-memory-compression",
            "--model",
            "gpt-test",
            "--base-url",
            "http://127.0.0.1:58081",
            "--api-style",
            "chat_completions",
            "--chat-response-format",
            "json_schema",
            "--max-output-tokens",
            "900",
            "--timeout-seconds",
            "15",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["model"] == "gpt-test"
    assert payload["candidate_count"] == 2
    assert payload["quality_gate_passed"] is True
    assert captured["model"] == "gpt-test"
    assert captured["settings"].openai_base_url == "http://127.0.0.1:58081"
    assert captured["settings"].model_api_style == "chat_completions"
    assert captured["settings"].chat_response_format == "json_schema"
    assert captured["settings"].max_output_tokens == 900
    assert captured["settings"].request_timeout_seconds == 15


def test_smoke_openai_embedding_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = {}

    def fake_smoke_runner(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return EmbeddingSmokeTestResult(
            ok=True,
            provider="fake_embeddings",
            model=kwargs["model"],
            latency_ms=2,
            input_count=2,
            dimensions=3,
            vector_preview=[0.1, 0.2, 0.3],
        )

    monkeypatch.setattr("proactive_assistant.cli.run_openai_embedding_smoke_test", fake_smoke_runner)

    exit_code = main(
        [
            "smoke-openai-embedding",
            "--model",
            "embedding-test",
            "--base-url",
            "http://127.0.0.1:58081",
            "--timeout-seconds",
            "8",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["model"] == "embedding-test"
    assert payload["dimensions"] == 3
    assert captured["model"] == "embedding-test"
    assert captured["settings"].openai_base_url == "http://127.0.0.1:58081"
    assert captured["settings"].request_timeout_seconds == 8


def test_evaluate_memory_extraction_fixtures_command_prints_report(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "evaluate-memory-extraction-fixtures",
            "--mode",
            "fixture",
            "--json-indent",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["case_count"] == 5
    assert payload["pass_count"] == 5
    assert payload["failed_cases"] == []
    assert payload["results"][0]["case_id"] == "action_owner_deadline_zh"


def test_merge_llm_proposals_command_rejects_duplicate_persona_ids(tmp_path: Path) -> None:
    normalized = tmp_path / "personas.normalized.jsonl"
    proposals = tmp_path / "llm_proposals.jsonl"
    merged = tmp_path / "personas.enriched.llm_merged.jsonl"

    main(["normalize-personas", "--input", str(FIXTURE_PATH), "--output", str(normalized)])
    payload = {
        "persona_id": "P_persona_beta",
        "schema_version": "llm_enrichment_contract_v0",
        "proposals": [
            {
                "field_name": "in_activity_tolerance",
                "value": 0.24,
                "confidence": 0.72,
                "evidence": ["dislikes interruption while studying"],
                "rationale": "The persona prefers uninterrupted focus during study.",
            }
        ],
    }
    proposals.write_text(json.dumps(payload) + "\n" + json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="Duplicate LLM proposals"):
        main(
            [
                "merge-llm-proposals",
                "--personas",
                str(normalized),
                "--proposals",
                str(proposals),
                "--output",
                str(merged),
            ]
        )
