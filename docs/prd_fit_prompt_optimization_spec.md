# PRD-Fit Prompt Timing And Content Granularity Spec

This document defines the first experiment specification for optimizing proactive prompt timing and content granularity. It is intentionally PRD-fit: the PRD product flow is treated as the source of truth, while colleague comments, whiteboard diagrams, and prior flowcharts are reference material for risks, examples, and evaluation details.

The v1 scope is meeting and business conversation only. Student classroom and study scenarios are explicitly out of scope for this spec, even when they appear in reference comments.

## Source Priority

1. Product PRD: required surface, session flow, prompt panel states, transcript, prompt list, summary, memory, and Suggest behavior.
2. Current research direction: optimize prompt timing and content granularity through simulation, persona preference fitting, and reward logging.
3. Existing codebase: reuse current schemas and simulation contracts where possible.
4. Colleague comments and diagrams: reference only; use them to identify risk, but do not treat them as mandatory product requirements.

## PRD Mapping Layer

### Glasses Prompt Panel

The glasses prompt panel is the primary intervention surface. In simulation, it maps to `AssistantInterventionAction` plus display metadata.

| PRD state | Product meaning | Experiment meaning | Required log signal |
| --- | --- | --- | --- |
| 启动态 | A prompt is beginning, usually title + loading/progress effect. | Candidate intervention is selected but content may still be pending. | `decision_id`, `timing_action`, `prompt_category`, `selection_tick` |
| 弹窗态 | A focused prompt appears with icon, captured text, AI text, and auto-close countdown. | Active intervention with measurable interruption cost. | `shown_tick`, `duration_policy`, `display_mode`, `content_granularity`, `captured_text_ref` |
| 常驻态 | A lightweight persistent prompt remains available. | Low-urgency intervention, lower interruption cost, higher revisit opportunity. | `visible_duration`, `reopen_count`, `ignored_until_tick` |

The PRD countdown choices map to `duration_policy`: `auto`, `3s`, `5s`, `8s`.

The PRD display choices map to `display_mode`: `auto`, `wrist_turn`, `head_raise`.

### App Session Flow

The app is the session management and review surface. It is not the policy decision maker.

| PRD surface | Product behavior | Experiment mapping |
| --- | --- | --- |
| Home pre-text panel | Optional schedule recommendation, manual input, attachment upload. | Scenario setup context and prior memory; not real-time reward. |
| Control panel | Timer, scene indicator, play/pause. | Episode lifecycle and `tick` control. |
| Session list | Entry point for past sessions, default filter by conversation prompt. | Episode archive and comparison set. |
| Recognition language selector | Language choice for ASR. | Metadata only in v1; no real ASR. |
| Prompt scenario selector | General/interview-like prompt context. | `meeting_business` scenario family in v1. |
| Display mode selector | Auto / manual wrist turn / manual head raise. | `display_mode` decision feature. |
| Prompt duration selector | Auto / 3s / 5s / 8s. | `duration_policy` decision feature. |
| Real-time transcript tab | Speaker, timestamp, transcript clip. | `TranscriptEvent` stream and context refs. |
| Prompt tab | Categories such as Q&A, explanation, person, suggestion. | `prompt_category` and prompt list evaluation. |
| Summary tab | End-session summary. | `after_activity` and `summary_gap_check`. |
| Memory storage | Persist session summary and useful facts. | `memory_write_candidate` after episode; interface only in v1. |
| Suggest | Extracted follow-up suggestions. | Post-session action item / gap-check output. |

## Experiment Abstraction Layer

### V1 Meeting And Business Task Classes

Only these four task classes are covered in v1.

| Task class | Trigger pattern | Assistant goal | Example output |
| --- | --- | --- | --- |
| `question_answer` | A speaker asks or implies a factual question. | Answer without derailing the conversation. | “Tencent was founded in 1998.” |
| `memory_recall` | The user needs prior context, names, decisions, or meeting history. | Retrieve or summarize remembered context. | “Last meeting assigned this to Bao.” |
| `communication_gap` | A goal, risk, action item, or missing confirmation appears. | Surface a concise gap or next action. | “Budget owner is still unconfirmed.” |
| `post_summary_suggest` | The session ends or a topic is closed. | Summarize decisions, gaps, and suggestions. | “Summary: 3 action items, 1 unresolved dependency.” |

The `memory_recall` task class maps to `person_or_fact` when the retrieved item is a person/fact and to `summary_gap_check` when it is a meeting-state recap.

### Activity Phase

Use the existing phase vocabulary where possible.

| Phase | Timing action | Typical PRD surface | Intended frequency |
| --- | --- | --- | --- |
| `pre_activity` | `before_activity` | Glasses 启动态 or app pre-text. | Low frequency; preparation only. |
| `in_activity` | `during_activity` | Glasses 弹窗态 or 常驻态. | Controlled by interruption budget. |
| `post_activity` | `after_activity` | Summary tab, Suggest, memory storage. | Usually once per task/topic/session. |
| `uncertain` | `manual` or `no_action` | Prompt tab pending list. | Use when confidence is insufficient. |

### Prompt Category

The prompt categories must align with the PRD prompt tab.

| Category | Meaning | PRD reference |
| --- | --- | --- |
| `question_answer` | Direct answer to a current question. | 问答 |
| `concept_explanation` | Explanation of a term, concept, or unclear point. | 解释 |
| `person_or_fact` | Person, company, number, date, or memory fact. | 人物 / fact memory |
| `suggestion` | Suggested next step, risk, or communication move. | 建议 / Suggest |
| `summary_gap_check` | Summary, missing item, action item, unresolved gap. | 总结 / 查漏补缺 |

### Timing Action

The timing action is the first dimension of the policy decision.

| Timing action | Use when | Avoid when |
| --- | --- | --- |
| `before_activity` | The user is about to start a meeting task and preparation has high value. | The meeting is already flowing or the prompt would repeat obvious agenda text. |
| `during_activity` | The transcript reveals a question, gap, confusion, or useful fact. | The user is speaking intensely, privacy risk is high, or the value is uncertain. |
| `after_activity` | A topic/session ends and summary/gap-check value is high. | The task is not actually closed or the user is still negotiating. |
| `manual` | The user explicitly requests help or opens prompt details. | It should not be counted as proactive timing success. |
| `no_action` | Expected value is below interruption/privacy/redundancy cost. | The system is highly confident the user needs immediate help. |

### Content Granularity

The content granularity is the second dimension of the policy decision.

| Level | Name | Glasses content | App detail behavior |
| --- | --- | --- | --- |
| `0` | No action | Nothing shown. | No prompt card. |
| `1` | Icon/title only | Icon + short title. | Detail opens only if clicked. |
| `2` | One-line answer | Captured text + one-line AI answer. | Detail page contains context refs. |
| `3` | Concise bullets | Captured text + 2-3 bullets. | Detail page contains transcript clip and rationale. |
| `4` | Detailed answer | Captured text + detailed answer/context refs. | Full detail view, best for app or manual open. |

Default policy: glasses should prefer levels `1-3`; level `4` should be reserved for manual open, post-session summary, or low-interruption moments.

## Interfaces To Preserve And Extend Later

The current code already has the core concepts. Do not replace them in the next implementation step.

| Current contract | Continue using for | Later fields to add |
| --- | --- | --- |
| `CandidateIntervention` | Candidate prompt scoring before selection. | `prompt_category`, `prd_surface`, `duration_policy`, `display_mode`, `privacy_level` |
| `PolicyDecision` | Final selected action and content level. | `timing_action`, `content_granularity`, `source_capture_ref`, `policy_features` |
| `FeedbackEvent` | Reward input and preference fitting. | `implicit_signals`, `task_outcome_refs`, `stability_bucket` |
| `AssistantInterventionAction` | Actual simulation action injected into the world. | `prompt_category`, `display_mode`, `duration_policy`, `source_capture_ref` |

Field definitions for later implementation:

```json
{
  "prompt_category": "question_answer | concept_explanation | person_or_fact | suggestion | summary_gap_check",
  "content_granularity": "0 | 1 | 2 | 3 | 4",
  "prd_surface": "glasses_starting | glasses_popup | glasses_persistent | app_prompt_tab | app_summary_tab | app_suggest",
  "display_mode": "auto | wrist_turn | head_raise",
  "duration_policy": "auto | 3s | 5s | 8s",
  "source_capture_ref": "transcript:<event_id> | memory:<memory_id> | object:<object_id>",
  "privacy_level": "low | medium | high"
}
```

No LLM calls are allowed in v1 experiments. Simulator, policy, and content generator should be deterministic fixtures or rules until the data contract is stable.

## Policy Decision Record

Every proactive decision should be logged, including `no_action`, so offline analysis can learn both intervention and suppression behavior.

Minimum decision log:

```json
{
  "decision_id": "dec_000123",
  "episode_id": "meeting_episode_001",
  "persona_id": "persona_alpha",
  "scenario_id": "business_meeting_t1",
  "tick": 42,
  "activity_phase": "in_activity",
  "available_timing_actions": ["during_activity", "no_action"],
  "chosen_timing_action": "during_activity",
  "prompt_category": "communication_gap",
  "content_granularity": 2,
  "display_mode": "auto",
  "duration_policy": "5s",
  "source_capture_ref": "transcript:transcript_0042_001",
  "estimated_help_value": 0.76,
  "estimated_interrupt_cost": 0.31,
  "estimated_privacy_risk": 0.12,
  "policy_version": "deterministic_prd_fit_v0"
}
```

## Feedback Observation Layer

Feedback must combine explicit, implicit, and task-level signals.

### Explicit Feedback

| Signal | Meaning | Reward direction |
| --- | --- | --- |
| `accept` | User accepts or acts on the prompt. | Positive. |
| `dismiss` | User closes the prompt. | Negative unless redundant but harmless. |
| `ignore` | Prompt expires or is not revisited. | Mild negative or neutral. |
| `snooze` | User wants it later. | Timing negative, content may still be useful. |
| `ask_followup` | User asks for more detail. | Content granularity may be too low; helpfulness positive. |
| `verbal_reject` | User rejects or complains. | Strong negative. |

### Implicit Feedback

| Signal | Meaning |
| --- | --- |
| `visible_duration_ms` | How long the prompt stayed visible. |
| `manual_open_detail` | User clicked to inspect full AI detail. |
| `reopen_count` | User returned to a prompt. |
| `close_within_ms` | Fast close indicates possible annoyance or low value. |
| `followup_query_count` | More questions after prompt; may mean useful but too brief. |
| `conversation_pause_ms` | Long pause after prompt may indicate disruption. |

### Task Outcome

| Signal | Meaning |
| --- | --- |
| `task_progress` | Meeting objective, answer, or action item advanced. |
| `gap_closed` | Previously unresolved gap becomes confirmed. |
| `summary_used` | User reads or accepts summary/Suggest. |
| `conversation_continues` | Prompt did not break conversation flow. |

## Reward Function

Default reward formula:

```text
reward =
  1.0 * accept
  + 0.8 * helpfulness
  + 0.7 * task_progress
  + 0.5 * timing_fit
  + 0.5 * content_fit
  - 1.0 * annoyance
  - 0.8 * flow_break
  - 0.7 * redundancy
  - 1.2 * privacy_risk
  - 0.3 * latency_penalty
```

Metric ranges:

| Metric | Range | Interpretation |
| --- | --- | --- |
| `accept` | `0.0-1.0` | Explicit or inferred acceptance. |
| `helpfulness` | `0.0-1.0` | Whether the prompt helped answer, recall, or decide. |
| `task_progress` | `0.0-1.0` | Whether the meeting advanced. |
| `timing_fit` | `-1.0-1.0` | Positive means timely; negative means too early/late/disruptive. |
| `content_fit` | `-1.0-1.0` | Positive means enough detail; negative means too terse/verbose/wrong. |
| `annoyance` | `0.0-1.0` | User irritation or interruption burden. |
| `flow_break` | `0.0-1.0` | Conversation disruption. |
| `redundancy` | `0.0-1.0` | Obvious or duplicate content. |
| `privacy_risk` | `0.0-1.0` | Sensitive company/customer/internal data risk. |
| `latency_penalty` | `0.0-1.0` | Prompt arrived too late or blocked flow. |

Privacy risk is intentionally weighted higher than normal annoyance because colleague comments flagged customer data, internal company data, and meeting privacy as sensitive.

## Preference Stability And Later RL Interface

The first implementation should not train a model. It should produce stable logs that later support contextual bandits or online RL.

Preference stability for one persona is reached when, across multiple meeting/business episodes:

1. Negative feedback rate changes less than a configured threshold across consecutive windows.
2. Chosen timing-action distribution becomes stable.
3. Chosen content-granularity distribution becomes stable.
4. Reward variance decreases without suppressing useful prompts to `no_action` by default.

Recommended default thresholds for later implementation:

```text
window_size = 20 decisions
negative_feedback_delta <= 0.05
timing_distribution_jsd <= 0.08
granularity_distribution_jsd <= 0.08
min_prompt_rate >= 0.10
```

After stability is reached, cluster personas by their stable policy distributions:

- timing preference: before-heavy, during-help, post-summary, low-proactivity.
- content preference: title-only, concise-answer, bullet-detail, detail-on-demand.
- sensitivity: privacy-sensitive, flow-sensitive, latency-sensitive, redundancy-sensitive.

These clusters should later inform the onboarding questionnaire and cold-start policy.

## V1 Non-Goals

- No student classroom scenario.
- No real ASR or language recognition implementation.
- No image-based major classification or external visual recognition.
- No LLM simulator, LLM policy, or LLM content generation.
- No real online RL training.
- No production memory database.
- No product UI rebuild beyond what is necessary to visualize simulation experiments.

## Acceptance Criteria

The spec is complete when:

1. Every PRD core surface has a mapping to experiment state, action, or log fields.
2. Meeting/business task classes are concrete and limited to v1 scope.
3. Timing and content granularity are represented as one joint policy decision.
4. Reward inputs cover explicit feedback, implicit feedback, and task outcome.
5. Existing schema names are reused instead of replaced.
6. Comments and flowcharts are clearly marked as reference-only.
7. Future implementation can add deterministic policy and reward logging without making a new product decision.
