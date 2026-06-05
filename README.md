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
src/proactive_assistant/detection/
  contracts.py
  rules.py
  service.py
src/proactive_assistant/model_gateway/
  clients.py
  settings.py
  smoke.py
src/proactive_assistant/meeting_state/
  contracts.py
  rules.py
  service.py
src/proactive_assistant/memory/
  contracts.py
  service.py
  store.py
src/proactive_assistant/orchestration/
  contracts.py
  service.py
src/proactive_assistant/prompting/
  builder.py
  contracts.py
  service.py
src/proactive_assistant/product/
  contracts.py
  service.py
src/proactive_assistant/persistence/
  sqlite.py
src/proactive_assistant/repositories/
  __init__.py
src/proactive_assistant/runtime/
  contracts.py
  service.py
  store.py
src/proactive_assistant/sessions/
  contracts.py
  service.py
  store.py
  windowing.py
src/proactive_assistant/schemas/
  enrichment.py
  llm_enrichment.py
  persona.py
  scenario.py
  transcript.py
  intervention.py
  feedback.py
  episode.py
src/proactive_assistant/simulation/
  actions.py
  agents.py
  api.py
  coordinates.py
  engine.py
  events.py
  map.py
  objects.py
  state.py
  transcripts.py
```

## Setup

```bash
uv sync --dev
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

## Model Gateway And Prompt Contract

The first production-facing module isolates model calls behind a stable gateway. Business code should call `PromptGenerationService` instead of calling a model SDK directly.

Environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.2
export OPENAI_FAST_MODEL=gpt-5-mini
export OPENAI_MAX_OUTPUT_TOKENS=1200
export OPENAI_REQUEST_TIMEOUT_SECONDS=30
```

The same settings can also use `PROACTIVE_` prefixes, for example `PROACTIVE_OPENAI_MODEL`.

For OpenAI-compatible Chat Completions endpoints like `OpenAI(base_url="http://{addr}:58081", api_key=...)`, use:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="http://{addr}:58081"
export OPENAI_API_STYLE="chat_completions"
export OPENAI_MODEL="gpt-5.5"
```

Supported model names are provider-side configuration, not hard-coded in this repo. Examples from the current API notes include `gpt-4o`, `gpt-4o-mini`, `gpt-5.5`, `doubao-seed-1-6-251015`, `qwen3-max`, `deepseek-v4-pro`, and `deepseek-v4-flash`.

The prompt contract supports PRD-fit prompt categories and content granularity:

```text
question_answer
concept_explanation
person_or_fact
suggestion
summary_gap_check

0 = no action
1 = icon/title only
2 = captured text + one-line answer
3 = captured text + concise bullets
4 = detailed answer + context refs
```

Offline tests use `FakeModelClient`, so `uv run pytest` never calls OpenAI.

Live OpenAI smoke test:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="<your-model>"

uv run proactive-assistant smoke-openai-prompt \
  --model "$OPENAI_MODEL" \
  --json-indent 2
```

You can also store local values in `.env`; `.env` and `.env.*` are ignored by git. The smoke command runs one structured-output prompt generation request and prints a compact JSON result with model, latency, category, content granularity, text fields, source refs, confidence, and privacy risk.

Live smoke test against the OpenAI-compatible Chat Completions endpoint:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="http://{addr}:58081"
export OPENAI_API_STYLE="chat_completions"

uv run proactive-assistant smoke-openai-prompt \
  --model "gpt-5.5" \
  --base-url "$OPENAI_BASE_URL" \
  --api-style chat_completions \
  --json-indent 2
```

If the compatible endpoint does not support `response_format={"type":"json_schema"}`, try `--chat-response-format json_object`. The parser can still read a JSON object from plain or fenced model text, but strict schema support is preferred.

Minimal service usage:

```python
from proactive_assistant.model_gateway import OpenAIResponsesClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.prompting import (
    PromptGenerationRequest,
    PromptGenerationService,
    TranscriptWindowItem,
)

settings = ModelGatewaySettings()
service = PromptGenerationService(
    model_client=OpenAIResponsesClient(
        api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    ),
    settings=settings,
)

result = service.generate_prompt(
    PromptGenerationRequest(
        session_id="meeting_001",
        transcript_window=[
            TranscriptWindowItem(
                transcript_id="transcript_001",
                speaker="Bao",
                text="Who owns the launch risk follow-up?",
            )
        ],
    )
)
```

## Session And Transcript Core

`src/proactive_assistant/sessions/` is the product-facing session and transcript base layer. It does not run ASR, generate prompts, summarize sessions, or write long-term memory. It records session metadata and append-only transcript segments, then builds clean context for the model gateway.

Core concepts:

```text
AssistantSession          session metadata and lifecycle
TranscriptSegmentRecord   append-only transcript segment
TranscriptWindow          recent transcript slice
SessionContextSnapshot    prompt-ready session context with memory placeholders
SessionService            business API over a SessionStore
InMemorySessionStore      deterministic early repository implementation
```

Minimal usage:

```python
from proactive_assistant.sessions import (
    InMemorySessionStore,
    SessionConfig,
    SessionService,
    TranscriptSegmentInput,
)

service = SessionService(InMemorySessionStore())
session = service.create_session(
    SessionConfig(
        title="Launch risk sync",
        pre_context="Discuss launch risks and owners.",
        privacy_constraints=["no customer data"],
    ),
    session_id="session_001",
)

service.append_transcript(
    session.session_id,
    TranscriptSegmentInput(
        speaker="Bao",
        start_ms=0,
        end_ms=1200,
        text="Who owns the launch risk follow-up?",
        asr_confidence=0.92,
    ),
)

prompt_request = service.build_prompt_generation_request(session.session_id)
```

The session layer carries `memory_context` and `memory_refs` in `SessionContextSnapshot`; the product flow can now populate them through `MemoryService` retrieval.

## Repository Interfaces

Persistence boundaries are defined by explicit repository protocols, not by a generic ORM-style repository. Business services depend on these protocols while current tests and local flows use in-memory implementations.

```text
SessionRepository   sessions and transcript segments
RuntimeRepository   prompt decisions, feedback, rewards, memory candidates
MemoryRepository    long-term memory records and deterministic retrieval
```

The legacy `SessionStore`, `RuntimeStore`, and `MemoryStore` names remain as compatibility aliases for the same protocols. The in-memory implementations satisfy the repository contracts:

```text
InMemorySessionStore  -> SessionRepository
InMemoryRuntimeStore  -> RuntimeRepository
InMemoryMemoryStore   -> MemoryRepository
```

SQLite persistence implements these protocols without changing `SessionService`, `PromptRuntimeService`, `MemoryService`, or `ProductAssistantService`.

## SQLite Persistence V0

`src/proactive_assistant/persistence/sqlite.py` provides a local SQLite backend for product experiments that need durable sessions, transcripts, prompt decisions, feedback, rewards, memory candidates, and memory records.

It uses JSON-first storage: each table stores indexed columns for common filters plus the full validated Pydantic payload. This keeps v0 close to the current contracts while leaving room to normalize hot fields later.

To run the Product Backend API with SQLite persistence:

```bash
export PROACTIVE_STORAGE_BACKEND=sqlite
export PROACTIVE_SQLITE_PATH=data/local/proactive.db

uv run uvicorn proactive_assistant.product.api:app --reload --port 8001
```

If `PROACTIVE_STORAGE_BACKEND` is unset, the API keeps using in-memory stores for local smoke tests and unit tests.

## Memory Core

`src/proactive_assistant/memory/` defines the first long-term memory contract and deterministic store. It is separate from `meeting_state`, which is short-lived working dialogue state, and separate from runtime `memory_candidates`, which are only proposed writes after feedback.

The v0 memory core includes:

```text
MemoryRecord
MemoryQuery
MemorySearchResult
MemoryRecordUpdate
MemoryType
MemoryScope
MemorySource
RetentionPolicy
MemoryWriteStatus
InMemoryMemoryStore
```

Supported store operations:

```text
add_memory
get_memory
list_memories
search_memories
update_memory
archive_memory
```

`MemoryService` adds the candidate-to-memory lifecycle on top of the store:

```text
propose_from_candidate
commit_candidate
confirm_memory
reject_memory
archive_memory
search_context
```

Runtime `MemoryCandidate` mappings are explicit:

```text
USER_PREFERENCE      -> MemoryType.USER_PREFERENCE / user scope / until_revoked
NEGATIVE_PREFERENCE  -> MemoryType.NEGATIVE_PREFERENCE / user scope / until_revoked
PRIVACY_PREFERENCE   -> MemoryType.PRIVACY_PREFERENCE / user scope / high privacy / until_revoked
MEETING_FACT         -> MemoryType.MEETING_FACT / session scope / 90d
ACTION_ITEM          -> MemoryType.ACTION_ITEM / session scope / 90d

ELIGIBLE             -> active memory
NEEDS_CONFIRMATION   -> pending_confirmation memory
BLOCKED              -> rejected memory
```

Memory write path consolidation v1 turns feedback-generated `MemoryCandidate` objects into retrieval-ready `MemoryRecord` metadata. It preserves raw candidate text and upstream structured fields, then fills missing deterministic fields when possible:

```text
owner / assignee
deadline / normalized_deadline
status
entity / canonical_entity / normalized_entity
prompt_category / prd_surface / display mode / duration policy
source refs / provenance
feedback_affinity
memory_schema_version = memory_consolidation_v1
```

The write path never overwrites explicit structured metadata supplied by upstream modules. These normalized fields are what enable retrieval v1 to answer exact owner/deadline/status lookups without relying only on keyword overlap.

Meeting state snapshot memory writing v1 can turn the current `MeetingState` into memory candidates and optionally commit them. It currently exports action items, decisions, risks, and tracked gaps. This path is explicit rather than automatic: product callers trigger it through the service or API when a session reaches a checkpoint such as meeting end.

The first retrieval implementation is deterministic. It filters by org, user, session visibility, type, scope, privacy level, source ids, tags, and write status, then ranks by keyword overlap, importance, confidence, and recency. It does not call an embedding model, vector database, or LLM yet.

Memory retrieval v1 adds an explainable deterministic ranker on top of the same store contract:

```text
query + prompt category + meeting context
-> target entity resolution from active meeting entities
-> intent classification
-> exact lookup path for deadline / owner / status
-> open recall path for rationale and general context
-> normalized ranking features
-> diversity filtering
-> use_policy for prompt context, policy hints, or ref-only privacy handling
```

Exact lookup is intentionally conservative: weak target matches become `display_ref_only` or no recall instead of fabricating a deadline, owner, or status. High-privacy memories on glasses surfaces are also returned as references only, not as full text context.

Minimal usage:

```python
from proactive_assistant.memory import (
    InMemoryMemoryStore,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryType,
)

store = InMemoryMemoryStore()
store.add_memory(
    MemoryRecord(
        memory_id="mem_001",
        memory_type=MemoryType.USER_PREFERENCE,
        scope=MemoryScope.USER,
        text="用户偏好非常简短的眼镜端提示。",
        org_id="org_001",
        user_id="user_001",
        source=MemorySource.MANUAL,
        confidence=0.9,
        importance=0.8,
        tags=["prompt_style"],
    )
)

results = store.search_memories(
    MemoryQuery(user_id="user_001", query_text="简短 提示")
)
```

Minimal candidate lifecycle usage:

```python
from proactive_assistant.memory import InMemoryMemoryStore, MemoryQuery, MemoryService

memory_service = MemoryService(InMemoryMemoryStore())
memory = memory_service.commit_candidate(memory_candidate, org_id="org_001", user_id="user_001")

if memory.write_status == "pending_confirmation":
    memory_service.confirm_memory(memory.memory_id)

context = memory_service.search_context(
    MemoryQuery(session_id="session_001", query_text="owner deadline")
)
```

This layer is now connected to the product flow. Feedback-generated memory candidates are committed into long-term memory through `MemoryService`, and active memories can be retrieved into the next prompt snapshot.

## Meeting State Tracker

`src/proactive_assistant/meeting_state/` maintains live meeting structure from transcript segments. It is not long-term memory; it is working dialogue state that can later be persisted into episodic memory.

The v0 tracker is rule-based and extracts:

```text
MeetingUtterance
OpenQuestion
ActionItem
Decision
Risk
MentionedRef
MeetingGap
```

It scans structure-level gaps that define meeting proactivity:

```text
unanswered_question
action_missing_owner
action_missing_deadline
action_missing_next_step
decision_missing_conclusion
open_risk
end_summary_needed
```

Minimal usage:

```python
from proactive_assistant.meeting_state import MeetingStateTracker

tracker = MeetingStateTracker()
state = tracker.create_state("session_001", scheduled_end_ms=45 * 60 * 1000)
update = tracker.update_from_segment(state, transcript_segment)
```

This module is the foundation for replacing broad keyword triggering with gap-based meeting understanding. It does not call models, retrieve memory, or write persistent storage.

In the product flow, structured meeting gaps can be adapted into prompt opportunities. The adapter only emits state-derived opportunities for structured gaps such as missing owner, missing deadline, missing decision conclusion, open risk, or end-summary needs. Raw unanswered questions remain the responsibility of realtime detection to avoid duplicate prompts from the same utterance.

## Realtime Prompt Opportunity Detection

`src/proactive_assistant/detection/` detects realtime prompt opportunities from recent transcript windows. It is a lightweight routing/filter layer, not a fact-answering module, Suggest generator, summary generator, memory retriever, or model caller.

The v0 detector is rule-based and covers the PRD-fit meeting/business categories:

```text
question_answer
person_or_fact
suggestion
summary_gap_check
```

It emits `PromptOpportunity` records with:

```text
trigger_segment_ids
captured_text
prompt_category
activity_phase
candidate_timing_action
suggested_content_granularity
priority: P0 | P1 | P2
confidence
privacy_level / privacy_risk
safety_flags
```

Minimal usage:

```python
from proactive_assistant.detection import PromptOpportunityDetector

snapshot = service.get_context_snapshot("session_001")
result = PromptOpportunityDetector().detect(snapshot)
```

The detector is intentionally conservative with sensitive business context. High privacy risk downgrades priority and content granularity so later modules can avoid showing detailed content in glasses popups.

## Prompt Orchestration

`src/proactive_assistant/orchestration/` connects the product-facing realtime chain:

```text
SessionContextSnapshot -> PromptOpportunityDetector -> PromptGenerationService -> PromptCandidate
```

The orchestrator does not call OpenAI directly, write memory, render UI, or train a policy. It maps each detected opportunity into a PRD-fit `PromptGenerationRequest`, including:

```text
prompt_category_candidate
target_content_granularity
prd_surface
display_mode
duration_policy
source capture context
privacy risk context
```

Default routing rules are intentionally explicit and easy to replace later:

1. `after_activity` opportunities go to `app_summary_tab`.
2. High-privacy or `P2` opportunities go to `app_prompt_tab` with manual-style display mode.
3. Low-risk urgent opportunities go to `glasses_popup` with a bounded display duration.
4. Model `should_prompt=false` outputs become suppressed candidates instead of errors.
5. Invalid model outputs become failed candidates and do not break the orchestration result.

Minimal offline usage:

```python
from proactive_assistant.model_gateway import FakeModelClient
from proactive_assistant.model_gateway.settings import ModelGatewaySettings
from proactive_assistant.orchestration import PromptOrchestrator
from proactive_assistant.prompting import PromptGenerationService

prompt_service = PromptGenerationService(
    model_client=FakeModelClient(
        {
            "should_prompt": True,
            "prompt_category": "summary_gap_check",
            "content_granularity": 2,
            "glasses_title": "Owner missing",
            "glasses_text": "Launch risk owner is still unconfirmed.",
            "app_detail_text": "Confirm owner, deadline, and next action.",
            "source_refs": ["transcript:transcript_001"],
            "confidence": 0.84,
            "privacy_level": "low",
            "privacy_risk": 0.08,
            "rationale": "The transcript contains an unresolved ownership gap.",
            "safety_flags": [],
        }
    ),
    settings=ModelGatewaySettings(default_model="gpt-test"),
)

snapshot = service.get_context_snapshot("session_001")
result = PromptOrchestrator(prompt_service=prompt_service).run(snapshot)
```

## Product Flow Service

`src/proactive_assistant/product/` is the product-facing entrypoint over the realtime backend chain:

```text
append transcript
-> update live meeting state
-> retrieve active memory context
-> build session snapshot
-> detect prompt opportunities
-> merge structured meeting gaps as extra opportunities
-> generate prompt candidates
-> log prompt decisions
-> return product prompt payloads
```

Feedback goes through the same layer:

```text
record feedback
-> update reward observation
-> propose memory candidates
-> commit candidates into long-term memory
```

The service keeps API/UI callers away from internal detector, orchestration, runtime ledger, and memory store details. It still does not run real ASR, create a database, or require OpenAI during tests.

`ProductTranscriptStepResult` also returns the current `meeting_state`, latest `meeting_gaps`, and retrieved `MemoryContext`, so API/UI callers can inspect unresolved meeting structure and memory retrieval without invoking separate services.

Minimal usage:

```python
from proactive_assistant.product import ProductAssistantService
from proactive_assistant.runtime import FeedbackSignalType

step = product_service.append_transcript_and_generate_prompts(
    "session_001",
    transcript_segment,
)

feedback = product_service.record_feedback(
    step.prompts[0].decision_id,
    FeedbackSignalType.ACCEPT,
)
```

## Product Backend API

`src/proactive_assistant/product/api.py` exposes the product flow as a FastAPI backend for frontend, device, or integration clients. It uses the same `ProductAssistantService` chain and defaults to the configured model gateway.

Run locally with the OpenAI-compatible Chat Completions endpoint:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="http://{addr}:58081"
export OPENAI_API_STYLE="chat_completions"
export OPENAI_MODEL="gpt-5.5"

uv run uvicorn proactive_assistant.product.api:app --reload --port 8001
```

Core endpoints:

```text
GET  /health
POST /sessions
GET  /sessions
GET  /sessions/{session_id}
POST /sessions/{session_id}/transcript
GET  /sessions/{session_id}/meeting-state
GET  /sessions/{session_id}/memory-candidates
GET  /sessions/{session_id}/memory-context
POST /sessions/{session_id}/memory-snapshot
GET  /sessions/{session_id}/prompts
GET  /prompts?session_id={session_id}
POST /prompt-decisions/{decision_id}/feedback
POST /memories/{memory_id}/confirm
POST /memories/{memory_id}/reject
POST /memories/{memory_id}/archive
```

The transcript endpoint returns frontend-ready prompt payloads plus live `meeting_state`, latest `meeting_gaps`, and retrieved memory context. It does not expose full internal snapshots or prompt candidate objects.

Minimal API flow:

```bash
curl -X POST http://127.0.0.1:8001/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "session_001",
    "config": {
      "title": "Launch risk sync",
      "pre_context": "讨论风险和负责人。",
      "privacy_constraints": ["avoid customer data"],
      "metadata": {"participants": ["Bao", "Alex"], "scheduled_end_ms": 1800000}
    }
  }'

curl -X POST http://127.0.0.1:8001/sessions/session_001/transcript \
  -H 'Content-Type: application/json' \
  -d '{
    "segment_id": "seg_0",
    "segment": {
      "speaker": "Bao",
      "start_ms": 0,
      "end_ms": 900,
      "text": "这个问题谁负责，下周五 deadline 前能不能定？",
      "asr_confidence": 0.94
    }
  }'
```

## Persona Pipeline CLI

The local CLI turns the persona data chain into reproducible files:

```bash
uv run proactive-assistant normalize-personas \
  --input tests/fixtures/persona/sample_persona_records.jsonl \
  --output data/processed/personas.normalized.jsonl

uv run proactive-assistant enrich-personas \
  --input data/processed/personas.normalized.jsonl \
  --output data/processed/personas.enriched.rule_based.jsonl \
  --report data/reports/persona_enrichment_coverage.json

uv run proactive-assistant report-personas \
  --input data/processed/personas.enriched.rule_based.jsonl \
  --output data/reports/persona_enrichment_coverage.json
```

Raw and processed data paths are ignored by git except for `.gitkeep` placeholders.

## Simulation API

Start the phase-1 in-memory simulation API locally:

```bash
uv run uvicorn proactive_assistant.simulation.api:app --reload
```

Core endpoints:

```text
GET  /health
POST /simulations
GET  /simulations/{simulation_id}/state
POST /simulations/{simulation_id}/step
POST /simulations/{simulation_id}/instructions
POST /simulations/{simulation_id}/assistant/interventions
GET  /simulations/{simulation_id}/logs
```

## 2D Sandbox Viewer

The first frontend visualization is an independent Phaser + Vite app. It reads the in-memory simulation API, renders the tile map and object layer, and provides lightweight controls for stepping the world.

Run the backend API in one terminal:

```bash
uv run uvicorn proactive_assistant.simulation.api:app --reload
```

Run the viewer in another terminal:

```bash
cd apps/sandbox-viewer
npm install
npm run dev
```

Open the Vite URL shown by the dev server, usually `http://127.0.0.1:5173`.

Build check:

```bash
cd apps/sandbox-viewer
npm run build
```

Current viewer boundary:

1. Renders the conference-room tile grid, walls, interaction zones, objects, agents, paths, dialogue bubbles, and recent event log.
2. Calls the backend `POST /simulations`, `GET /state`, `POST /step`, and `POST /instructions` endpoints through a Vite `/api` proxy.
3. Keeps agent behavior deterministic; the `Demo Actions` button only sends hard-coded structured JSON actions for integration testing.
4. Does not include LLM planning, assistant policy learning, avatar art, feedback training UI, or a commuting scene yet.

## 2D Sandbox Direction

The simulation visualization direction is now a Smallville-style 2D tile sandbox rather than a 3D scene. See [2D Sandbox Simulation Plan](docs/2d_sandbox_simulation_plan.md) for the phase-1 architecture and implementation boundary.

The prompt optimization experiment should fit the PRD while treating colleague comments as reference material. See [PRD-Fit Prompt Timing And Content Granularity Spec](docs/prd_fit_prompt_optimization_spec.md) for the v1 meeting/business experiment contract.

## Design Notes

The schema layer separates simulator-only hidden state from policy-visible observations. Persona records may include hidden proactive preferences for simulation, but the assistant policy should only receive questionnaire priors, interaction history, and transcript-derived context.
