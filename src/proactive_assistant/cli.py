from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from proactive_assistant.adapters import PersonaDatasetAdapter
from proactive_assistant.enrichment import (
    RuleBasedPersonaEnricher,
    merge_llm_enrichment,
    parse_llm_enrichment_response,
)
from proactive_assistant.io import read_jsonl, write_jsonl
from proactive_assistant.memory import evaluate_memory_extraction_fixtures, load_memory_extraction_fixture_cases
from proactive_assistant.model_gateway import (
    ModelGatewayError,
    ModelGatewaySettings,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from proactive_assistant.model_gateway.smoke import (
    run_openai_embedding_smoke_test,
    run_openai_memory_compression_smoke_test,
    run_openai_memory_extraction_smoke_test,
    run_openai_prompt_smoke_test,
)
from proactive_assistant.reports import build_persona_coverage_report
from proactive_assistant.schemas.llm_enrichment import LLMEnrichmentResponse
from proactive_assistant.schemas.persona import Persona


DEFAULT_NORMALIZED_OUTPUT = Path("data/processed/personas.normalized.jsonl")
DEFAULT_RULE_OUTPUT = Path("data/processed/personas.enriched.rule_based.jsonl")
DEFAULT_REPORT_OUTPUT = Path("data/reports/persona_enrichment_coverage.json")
DEFAULT_LLM_MERGED_OUTPUT = Path("data/processed/personas.enriched.llm_merged.jsonl")
DEFAULT_MEMORY_EXTRACTION_FIXTURES = Path("tests/fixtures/memory_extraction/cases.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proactive-assistant",
        description="Local data pipeline utilities for proactive assistant research.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser(
        "normalize-personas",
        help="Convert local PERSONA-like records into internal Persona JSONL.",
    )
    normalize.add_argument("--input", required=True, type=Path)
    normalize.add_argument("--output", default=DEFAULT_NORMALIZED_OUTPUT, type=Path)
    normalize.set_defaults(handler=handle_normalize_personas)

    enrich = subparsers.add_parser(
        "enrich-personas",
        help="Run deterministic rule-based enrichment on normalized Persona JSONL.",
    )
    enrich.add_argument("--input", required=True, type=Path)
    enrich.add_argument("--output", default=DEFAULT_RULE_OUTPUT, type=Path)
    enrich.add_argument("--report", default=DEFAULT_REPORT_OUTPUT, type=Path)
    enrich.set_defaults(handler=handle_enrich_personas)

    merge = subparsers.add_parser(
        "merge-llm-proposals",
        help="Merge local LLM proposal JSONL into normalized/enriched Persona JSONL.",
    )
    merge.add_argument("--personas", required=True, type=Path)
    merge.add_argument("--proposals", required=True, type=Path)
    merge.add_argument("--output", default=DEFAULT_LLM_MERGED_OUTPUT, type=Path)
    merge.add_argument("--report", default=DEFAULT_REPORT_OUTPUT, type=Path)
    merge.set_defaults(handler=handle_merge_llm_proposals)

    report = subparsers.add_parser(
        "report-personas",
        help="Write a coverage report for a Persona JSONL file.",
    )
    report.add_argument("--input", required=True, type=Path)
    report.add_argument("--output", default=DEFAULT_REPORT_OUTPUT, type=Path)
    report.set_defaults(handler=handle_report_personas)

    smoke = subparsers.add_parser(
        "smoke-openai-prompt",
        help="Run one live OpenAI structured-output prompt smoke test.",
    )
    smoke.add_argument("--model", default=None, help="Override OPENAI_MODEL for this smoke test.")
    smoke.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL, e.g. http://{addr}:58081.")
    smoke.add_argument(
        "--api-style",
        choices=["responses", "chat_completions"],
        default=None,
        help="Use responses for OpenAI Responses API or chat_completions for OpenAI-compatible APIs.",
    )
    smoke.add_argument(
        "--chat-response-format",
        choices=["json_schema", "json_object", "none"],
        default=None,
    )
    smoke.add_argument("--max-output-tokens", default=None, type=int)
    smoke.add_argument("--timeout-seconds", default=None, type=float)
    smoke.add_argument("--json-indent", default=None, type=int)
    smoke.set_defaults(handler=handle_smoke_openai_prompt)

    smoke_memory = subparsers.add_parser(
        "smoke-openai-memory-extraction",
        help="Run one live OpenAI structured-output memory extraction smoke test.",
    )
    smoke_memory.add_argument("--model", default=None, help="Override OPENAI_MODEL for this smoke test.")
    smoke_memory.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL, e.g. http://{addr}:58081.")
    smoke_memory.add_argument(
        "--api-style",
        choices=["responses", "chat_completions"],
        default=None,
        help="Use responses for OpenAI Responses API or chat_completions for OpenAI-compatible APIs.",
    )
    smoke_memory.add_argument(
        "--chat-response-format",
        choices=["json_schema", "json_object", "none"],
        default="json_object",
    )
    smoke_memory.add_argument("--max-output-tokens", default=None, type=int)
    smoke_memory.add_argument("--timeout-seconds", default=None, type=float)
    smoke_memory.add_argument("--json-indent", default=None, type=int)
    smoke_memory.set_defaults(handler=handle_smoke_openai_memory_extraction)

    smoke_compression = subparsers.add_parser(
        "smoke-openai-memory-compression",
        help="Run one live OpenAI structured-output memory compression smoke test.",
    )
    smoke_compression.add_argument("--model", default=None, help="Override OPENAI_MODEL for this smoke test.")
    smoke_compression.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL, e.g. http://{addr}:58081.")
    smoke_compression.add_argument(
        "--api-style",
        choices=["responses", "chat_completions"],
        default=None,
    )
    smoke_compression.add_argument(
        "--chat-response-format",
        choices=["json_schema", "json_object", "none"],
        default="json_object",
    )
    smoke_compression.add_argument("--max-output-tokens", default=None, type=int)
    smoke_compression.add_argument("--timeout-seconds", default=None, type=float)
    smoke_compression.add_argument("--json-indent", default=None, type=int)
    smoke_compression.set_defaults(handler=handle_smoke_openai_memory_compression)

    smoke_embedding = subparsers.add_parser(
        "smoke-openai-embedding",
        help="Run one live OpenAI-compatible embedding API smoke test.",
    )
    smoke_embedding.add_argument("--model", default=None, help="Override OPENAI_EMBEDDING_MODEL for this smoke test.")
    smoke_embedding.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL, e.g. http://{addr}:58081.")
    smoke_embedding.add_argument("--timeout-seconds", default=None, type=float)
    smoke_embedding.add_argument("--json-indent", default=None, type=int)
    smoke_embedding.set_defaults(handler=handle_smoke_openai_embedding)

    eval_memory = subparsers.add_parser(
        "evaluate-memory-extraction-fixtures",
        help="Evaluate memory extraction quality on fixture cases.",
    )
    eval_memory.add_argument("--input", default=DEFAULT_MEMORY_EXTRACTION_FIXTURES, type=Path)
    eval_memory.add_argument(
        "--mode",
        choices=["fixture", "live"],
        default="fixture",
        help="fixture uses embedded fixture_model_output; live calls the configured model gateway.",
    )
    eval_memory.add_argument("--model", default=None, help="Override OPENAI_MODEL for live evaluation.")
    eval_memory.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL, e.g. http://{addr}:58081.")
    eval_memory.add_argument(
        "--api-style",
        choices=["responses", "chat_completions"],
        default=None,
    )
    eval_memory.add_argument(
        "--chat-response-format",
        choices=["json_schema", "json_object", "none"],
        default="json_object",
    )
    eval_memory.add_argument("--max-output-tokens", default=None, type=int)
    eval_memory.add_argument("--timeout-seconds", default=None, type=float)
    eval_memory.add_argument("--json-indent", default=None, type=int)
    eval_memory.set_defaults(handler=handle_evaluate_memory_extraction_fixtures)

    return parser


def handle_normalize_personas(args: argparse.Namespace) -> None:
    adapter = PersonaDatasetAdapter()
    personas = adapter.from_path(args.input)
    adapter.write_normalized_jsonl(personas, args.output)
    print(json.dumps({"personas_written": len(personas), "output": str(args.output)}))


def handle_enrich_personas(args: argparse.Namespace) -> None:
    personas = load_personas_jsonl(args.input)
    enricher = RuleBasedPersonaEnricher()
    enriched = [enricher.enrich(persona).enriched_persona for persona in personas]
    write_personas_jsonl(enriched, args.output)
    write_report(enriched, args.report)
    print(
        json.dumps(
            {
                "personas_written": len(enriched),
                "output": str(args.output),
                "report": str(args.report),
            }
        )
    )


def handle_merge_llm_proposals(args: argparse.Namespace) -> None:
    personas = load_personas_jsonl(args.personas)
    proposals = load_llm_proposals_jsonl(args.proposals)
    proposals_by_persona = index_llm_proposals_by_persona(proposals)

    merged: list[Persona] = []
    unmatched_proposals = set(proposals_by_persona)
    for persona in personas:
        proposal = proposals_by_persona.get(persona.internal_persona_id)
        if proposal is None:
            proposal = proposals_by_persona.get(persona.source_persona_id)
        if proposal is None:
            merged.append(persona)
            continue
        unmatched_proposals.discard(proposal.persona_id)
        merged.append(merge_llm_enrichment(persona, proposal).enriched_persona)

    if unmatched_proposals:
        raise ValueError(f"LLM proposals reference unknown personas: {sorted(unmatched_proposals)}")

    write_personas_jsonl(merged, args.output)
    write_report(merged, args.report)
    print(
        json.dumps(
            {
                "personas_written": len(merged),
                "output": str(args.output),
                "report": str(args.report),
            }
        )
    )


def handle_report_personas(args: argparse.Namespace) -> None:
    personas = load_personas_jsonl(args.input)
    write_report(personas, args.output)
    print(json.dumps({"personas_reported": len(personas), "report": str(args.output)}))


def handle_smoke_openai_prompt(args: argparse.Namespace) -> None:
    settings = _settings_from_smoke_args(args)
    result = run_openai_prompt_smoke_test(settings=settings, model=args.model)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=args.json_indent))


def handle_smoke_openai_memory_extraction(args: argparse.Namespace) -> None:
    settings = _settings_from_smoke_args(args)
    result = run_openai_memory_extraction_smoke_test(settings=settings, model=args.model)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=args.json_indent))


def handle_smoke_openai_memory_compression(args: argparse.Namespace) -> None:
    settings = _settings_from_smoke_args(args)
    result = run_openai_memory_compression_smoke_test(settings=settings, model=args.model)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=args.json_indent))


def handle_smoke_openai_embedding(args: argparse.Namespace) -> None:
    settings = _settings_from_smoke_args(args)
    result = run_openai_embedding_smoke_test(settings=settings, model=args.model)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=args.json_indent))


def handle_evaluate_memory_extraction_fixtures(args: argparse.Namespace) -> None:
    settings = _settings_from_smoke_args(args)
    cases = load_memory_extraction_fixture_cases(args.input)
    model_client = _model_client_from_settings(settings) if args.mode == "live" else None
    report = evaluate_memory_extraction_fixtures(
        cases,
        mode=args.mode,
        model_client=model_client,
        settings=settings,
        model=args.model,
    )
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=args.json_indent))


def _settings_from_smoke_args(args: argparse.Namespace) -> ModelGatewaySettings:
    settings = ModelGatewaySettings()
    updates = {}
    if getattr(args, "max_output_tokens", None) is not None:
        updates["max_output_tokens"] = args.max_output_tokens
    if getattr(args, "timeout_seconds", None) is not None:
        updates["request_timeout_seconds"] = args.timeout_seconds
    base_url = _non_empty_arg(getattr(args, "base_url", None))
    if base_url is not None:
        updates["openai_base_url"] = base_url
    if getattr(args, "api_style", None) is not None:
        updates["model_api_style"] = args.api_style
    if getattr(args, "chat_response_format", None) is not None:
        updates["chat_response_format"] = args.chat_response_format
    if updates:
        settings = settings.model_copy(update=updates)
    return settings


def _non_empty_arg(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _model_client_from_settings(settings: ModelGatewaySettings):
    if settings.openai_api_key is None:
        raise ModelGatewayError("OPENAI_API_KEY or PROACTIVE_OPENAI_API_KEY is required for live evaluation")
    if settings.model_api_style == "chat_completions" or settings.openai_base_url is not None:
        return OpenAIChatCompletionsClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            response_format=settings.chat_response_format,
            max_tokens_param=settings.chat_max_tokens_param,
        )
    return OpenAIResponsesClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )


def load_personas_jsonl(path: Path) -> list[Persona]:
    return [Persona.model_validate(record) for record in read_jsonl(path)]


def write_personas_jsonl(personas: list[Persona], path: Path) -> None:
    write_jsonl(path, (persona.model_dump(mode="json") for persona in personas))


def load_llm_proposals_jsonl(path: Path) -> list[LLMEnrichmentResponse]:
    return [parse_llm_enrichment_response(record) for record in read_jsonl(path)]


def index_llm_proposals_by_persona(
    proposals: list[LLMEnrichmentResponse],
) -> dict[str, LLMEnrichmentResponse]:
    indexed: dict[str, LLMEnrichmentResponse] = {}
    duplicates: set[str] = set()
    for proposal in proposals:
        if proposal.persona_id in indexed:
            duplicates.add(proposal.persona_id)
            continue
        indexed[proposal.persona_id] = proposal
    if duplicates:
        raise ValueError(f"Duplicate LLM proposals for personas: {sorted(duplicates)}")
    return indexed


def write_report(personas: list[Persona], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = build_persona_coverage_report(personas)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
