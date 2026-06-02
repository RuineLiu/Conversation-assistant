# Proactive Assistant

This repository implements a simulation-first proactive assistant research framework.

The first milestones are intentionally narrow:

1. Define validated data schemas for personas, scenarios, transcripts, interventions, feedback, and episode trajectories.
2. Add a local PERSONA dataset adapter that converts already-downloaded records into the internal `Persona` schema.
3. Add deterministic persona enrichment as a transparent baseline before introducing LLM inference.
4. Define a strict LLM enrichment contract and merge layer without calling model APIs.
5. Keep all assistant inputs constrained to audio-derived transcripts and metadata.
6. Avoid model calls, reinforcement learning, database setup, and dashboard work until the schema and adapter layers are stable.

## Current Scope

Implemented modules:

```text
src/proactive_assistant/adapters/
  persona_dataset.py
src/proactive_assistant/io/
  jsonl.py
src/proactive_assistant/enrichment/
  llm_contract.py
  merge.py
  rule_based.py
src/proactive_assistant/schemas/
  enrichment.py
  llm_enrichment.py
  persona.py
  scenario.py
  transcript.py
  intervention.py
  feedback.py
  episode.py
```

## Setup

```bash
uv venv --python python3.12
uv pip install -e ".[dev]"
uv run pytest
```

Parquet input support is optional:

```bash
uv pip install -e ".[parquet]"
```

## PERSONA Adapter

The adapter does not download the gated SynthLabsAI/PERSONA dataset and does not call an LLM. It reads local `.jsonl`, `.json`, or `.parquet` files, groups records by persona id, and emits validated `Persona` objects.

Unknown proactive-assistant fields are kept as low-confidence placeholders:

```text
source = "llm_inferred"
confidence = 0.0
```

Those fields are also listed in `source_metadata.pending_enrichment_fields` so later LLM enrichment or simulator learning can replace them explicitly.

## Persona Enrichment

`RuleBasedPersonaEnricher` is a deterministic baseline for turning visible persona text and preference samples into proactive-assistant fields. It is deliberately conservative:

1. It only uses local `Persona.source_metadata` text evidence.
2. It does not call external models.
3. It does not overwrite high-confidence `source_derived` fields.
4. It records matched terms and evidence text for each update.

This gives later LLM enrichment a testable baseline and an audit trail.

## LLM Enrichment Contract

The LLM contract is implemented without making any model calls. Untrusted JSON is parsed into `LLMEnrichmentResponse`, then merged through `merge_llm_enrichment`.

Merge rules:

1. LLM proposals cannot overwrite `manual` fields.
2. LLM proposals cannot overwrite high-confidence `source_derived` fields.
3. Low-confidence proposals are skipped.
4. `rule_based` fields are replaced only when LLM confidence improves.
5. Every merge records updated and skipped fields in persona metadata.

## Design Notes

The schema layer separates simulator-only hidden state from policy-visible observations. Persona records may include hidden proactive preferences for simulation, but the assistant policy should only receive questionnaire priors, interaction history, and transcript-derived context.
