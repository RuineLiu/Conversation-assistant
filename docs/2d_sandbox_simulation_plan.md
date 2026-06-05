# 2D Sandbox Simulation Plan

This document replaces the previous 3D scene viewer direction. The new target is a Smallville-style 2D sandbox simulation for proactive assistant research.

## Goal

Build a top-down 2D conference-room simulation where agents move, interact with objects, speak, receive proactive assistant interventions, and provide structured feedback.

The first useful system should support interruption-timing experiments before it attempts high-fidelity LLM planning or reinforcement learning.

## Core Decisions

1. Use a 2D tile-based sandbox, not a 3D scene.
2. Implement only the meeting-room scenario in the first version.
3. Start with 2-3 agents.
4. Use procedural placeholder tiles and sprites first; replace art later.
5. Use Phaser.js + TypeScript for the frontend.
6. Use Python FastAPI for the backend API.
7. Keep the backend simulation tick/state/action/log system authoritative.
8. Use deterministic rule-based agents first.
9. Leave interfaces for LLM memory, planning, reflection, and reinforcement learning, but do not call models in phase 1.
10. Treat the proactive assistant as a separate simulated system, not as a frontend decoration.

## Non-Goals For Phase 1

- No 3D model or 3D rendering.
- No static single-image scene.
- No commuting scenario.
- No real audio ingestion.
- No real-time ASR.
- No LLM calls.
- No reinforcement learning training loop.
- No avatar-quality character art requirement.
- No production database.

## Repository Layout

Planned additions:

```text
apps/sandbox-viewer/
  index.html
  package.json
  src/
    main.ts
    game/
      ConferenceRoomScene.ts
      tileMap.ts
      agents.ts
      overlays.ts
      apiClient.ts

src/proactive_assistant/simulation/
  __init__.py
  actions.py
  agents.py
  api.py
  engine.py
  events.py
  map.py
  objects.py
  state.py
  transcripts.py

tests/simulation/
  test_actions.py
  test_engine.py
  test_map.py
  test_api.py
```

## 2D Map Model

The map should be Tiled-style even if we generate it procedurally in phase 1.

Initial map:

- Tile size: `32px`.
- Logical scale: `1 tile = 0.30m`.
- Room footprint: approximately `26 x 18 tiles`, mapping to `7.8m x 5.4m`.
- The simulation keeps metric metadata so later layouts can use exact physical dimensions.

Layers:

```text
floor
walls
furniture
objects
collision
spawn_points
interaction_zones
debug
```

Required objects:

- `door`
- `conference_table`
- `chair`
- `screen`
- `whiteboard`
- `light_switch`
- `laptop`
- `documents`
- `cup`
- `remote`

Each object should have:

```json
{
  "object_id": "screen_1",
  "type": "screen",
  "label": "Presentation screen",
  "tile": [3, 5],
  "size": [2, 3],
  "blocks_movement": true,
  "interaction_tiles": [[5, 5], [5, 6]],
  "state": {
    "power": "off",
    "display_mode": "blank"
  }
}
```

## Walkability

The backend owns walkability.

Rules:

1. Agents cannot move through wall tiles.
2. Agents cannot move through blocking object tiles.
3. Agents can stand on interaction tiles adjacent to objects.
4. Agents can reserve a tile while moving to prevent overlap.
5. The frontend may animate movement, but the backend decides the path and final position.

Pathfinding:

- Phase 1: grid BFS or A*.
- Diagonal movement: disabled.
- Path output: ordered tile list.

## Agent Model

Phase 1 agents are deterministic but structured to support later LLM behavior.

```json
{
  "agent_id": "agent_alex",
  "persona_id": "persona_001",
  "display_name": "Alex",
  "role": "presenter",
  "tile": [8, 9],
  "facing": "east",
  "current_goal": "prepare presentation",
  "current_action": null,
  "path": [],
  "public_mood": "focused",
  "assistant_state": {
    "wearing_device": true,
    "last_intervention_tick": null,
    "interruption_load": 0.0
  }
}
```

Persona influence in phase 1:

- `interruption_tolerance`
- `detail_preference`
- `proactivity_preference`
- `meeting_role`
- `task_focus`

These can be read from the existing persona schema/enrichment outputs when available; otherwise phase 1 can use small fixture personas.

## Action Schema

All actions are structured JSON. Free text is allowed only inside explicit text fields such as `utterance` or `content`.

### Move

```json
{
  "type": "move_to",
  "agent_id": "agent_alex",
  "target_tile": [11, 8],
  "reason": "approach laptop"
}
```

### Speak

```json
{
  "type": "speak",
  "agent_id": "agent_alex",
  "utterance": "Let's start with the quarterly numbers.",
  "audience": ["agent_bao", "agent_chris"]
}
```

### Interact With Object

```json
{
  "type": "interact_object",
  "agent_id": "agent_alex",
  "object_id": "screen_1",
  "interaction": "turn_on"
}
```

### Update Goal

```json
{
  "type": "update_goal",
  "agent_id": "agent_alex",
  "goal": "answer budget question"
}
```

### Assistant Intervention

```json
{
  "type": "assistant_intervention",
  "intervention_id": "int_00042",
  "target_agent_id": "agent_alex",
  "timing_policy": "before_activity",
  "channel": "smart_glasses_overlay",
  "content": "You planned to share the budget slide before Q&A.",
  "detail_level": 0.45,
  "context_refs": ["transcript_event_011", "object:laptop_1"]
}
```

### Feedback To Assistant

```json
{
  "type": "feedback_to_assistant",
  "agent_id": "agent_alex",
  "intervention_id": "int_00042",
  "feedback": "accept",
  "task_disruption": 0.1,
  "reason": "timely reminder"
}
```

Allowed feedback values:

```text
accept
ignore
dismiss
snooze
ask_followup
verbal_reject
task_disruption
```

## World State

The world state should be serializable and deterministic.

```json
{
  "simulation_id": "meeting_demo_001",
  "tick": 42,
  "scenario_id": "conference_room_meeting",
  "map": {
    "width": 26,
    "height": 18,
    "tile_size": 32
  },
  "agents": {},
  "objects": {},
  "active_dialogue": [],
  "active_interventions": [],
  "recent_events": []
}
```

The backend should expose full state for debugging in phase 1. Later phases can add policy-visible observation filters.

## Tick Loop

Each backend step runs:

1. Accept queued user instructions.
2. Generate or read transcript events.
3. Let deterministic agent policies propose actions.
4. Let assistant policy observe the world/transcript and optionally propose interventions.
5. Validate actions.
6. Apply one action per agent where possible.
7. Update object states.
8. Update paths and positions.
9. Generate feedback events for interventions.
10. Append all events to logs.
11. Return updated world state.

Phase 1 should be turn/tick based, not continuous real-time.

## Assistant Model

The assistant is represented as a separate system.

Inputs:

- Current world state.
- Recent transcript events.
- Target agent persona fields.
- Target agent current goal.
- Recent intervention history.

Outputs:

- No-op.
- Structured assistant intervention.

Initial policies:

- `before_activity`: reminder before a planned action.
- `during_activity`: help while agent is stuck or delayed.
- `after_activity`: summary/checklist after action completion.

The first policy should be deterministic and configurable. Reinforcement learning comes later.

## Transcript Events

Meeting transcript events should exist even before real audio ingestion.

```json
{
  "event_id": "transcript_event_011",
  "tick": 38,
  "speaker_agent_id": "agent_bao",
  "text": "Can we confirm the action items before the client joins?",
  "topic": "action_items",
  "derived_intents": ["request_summary"]
}
```

The assistant can use transcript events as context for interventions.

## API Surface

Phase 1 FastAPI endpoints:

```text
GET  /health
POST /simulations
GET  /simulations/{simulation_id}/state
POST /simulations/{simulation_id}/step
POST /simulations/{simulation_id}/instructions
POST /simulations/{simulation_id}/assistant/interventions
GET  /simulations/{simulation_id}/logs
```

Step request:

```json
{
  "ticks": 1
}
```

Instruction request:

```json
{
  "instruction": "Ask Alex to start the presentation.",
  "target_agent_id": "agent_alex"
}
```

Logs response should include:

- movement events
- dialogue events
- object interaction events
- assistant interventions
- feedback events
- validation failures

## Frontend Requirements

Use Phaser.js for:

- tile map rendering
- agent sprites
- movement animation
- collision/debug overlay
- object state visualization
- dialogue bubbles
- assistant intervention bubbles
- trajectory trails
- current public goals

Frontend should not decide world state.

Frontend loop:

1. Fetch initial world state.
2. Render map/object/agents.
3. User clicks `Step`.
4. Frontend calls backend step API.
5. Animate changes from previous state to new state.
6. Render logs and overlays.

## Phase 1 Acceptance Criteria

Phase 1 is done when:

1. A Phaser page opens with a top-down tile conference room.
2. At least 2 agents are visible.
3. Agents can move through backend-provided paths.
4. Collision prevents walking through the table, walls, chairs, and key objects.
5. Agents can interact with at least screen, whiteboard, laptop, documents, and door.
6. Backend `/step` updates state deterministically.
7. Logs record movement, dialogue, object interaction, interventions, and feedback.
8. Assistant interventions can be injected and visualized.
9. Rule-based assistant can trigger at least one before/during/after intervention.
10. Existing persona tests still pass.

## Implementation Order

### Step 1: Backend simulation kernel

Add schemas, world state, map, objects, actions, validation, pathfinding, and deterministic step engine. Include tests.

### Step 2: FastAPI API

Expose simulation create/state/step/instructions/interventions/logs endpoints. Include API tests.

### Step 3: Phaser frontend shell

Create Vite + Phaser app. Render tile map, objects, and agents from backend state.

### Step 4: Movement and overlays

Animate agent movement and render dialogue, goals, assistant interventions, feedback events, and trajectories.

### Step 5: Persona integration

Load enriched persona records into simulation agent profiles.

### Step 6: Assistant timing experiments

Add deterministic before/during/after policies and reward logging. Keep RL offline until the simulation data format is stable.

## Open Questions For Later

These do not block phase 1:

1. Final visual art style.
2. Whether maps should be authored in Tiled editor or generated from JSON.
3. Whether frontend and backend run as separate dev servers or through a single proxy.
4. Which LLM provider will later drive planning/reflection.
5. Whether reinforcement learning starts with contextual bandits, PPO, or offline preference modeling.
