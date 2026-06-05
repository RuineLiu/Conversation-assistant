from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from proactive_assistant.simulation.actions import (
    AssistantFeedback,
    AssistantInterventionAction,
    FeedbackToAssistantAction,
    InteractObjectAction,
    MoveToAction,
    NoOpAction,
    ObjectInteraction,
    SimulationAction,
    SpeakAction,
    UpdateGoalAction,
    parse_simulation_action,
)
from proactive_assistant.simulation.agents import Facing, PublicMood
from proactive_assistant.simulation.coordinates import Tile
from proactive_assistant.simulation.events import EventType, SimulationEvent
from proactive_assistant.simulation.objects import ObjectType, SimulationObject
from proactive_assistant.simulation.state import DialogueBubble, WorldState, create_default_world_state
from proactive_assistant.simulation.transcripts import TranscriptEvent


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: WorldState
    events: list[SimulationEvent]


class SimulationEngine:
    def __init__(self, state: WorldState | None = None) -> None:
        self.state = state or create_default_world_state()

    def step(
        self,
        actions: Sequence[SimulationAction | dict[str, Any]] | None = None,
    ) -> StepResult:
        state = self.state.model_copy(deep=True)
        state.tick += 1
        events: list[SimulationEvent] = []

        parsed_actions = [parse_simulation_action(action) for action in actions or []]
        for action in parsed_actions:
            self._apply_action(state, action, events)

        self._advance_agent_paths(state, events)
        state.recent_events = events
        state.event_log.extend(events)
        self.state = state
        return StepResult(state=state, events=events)

    def _event(
        self,
        events: list[SimulationEvent],
        state: WorldState,
        event_type: EventType,
        *,
        actor_id: str | None = None,
        target_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        events.append(
            SimulationEvent(
                event_id=f"event_{state.tick:04d}_{len(events) + 1:03d}",
                tick=state.tick,
                event_type=event_type,
                actor_id=actor_id,
                target_id=target_id,
                payload=payload or {},
            )
        )

    def _reject(
        self,
        state: WorldState,
        events: list[SimulationEvent],
        *,
        reason: str,
        action: SimulationAction,
        actor_id: str | None = None,
        target_id: str | None = None,
    ) -> None:
        self._event(
            events,
            state,
            EventType.ACTION_REJECTED,
            actor_id=actor_id,
            target_id=target_id,
            payload={"reason": reason, "action": action.model_dump(mode="json")},
        )

    def _apply_action(
        self,
        state: WorldState,
        action: SimulationAction,
        events: list[SimulationEvent],
    ) -> None:
        if isinstance(action, NoOpAction):
            self._event(events, state, EventType.NO_OP, payload={"reason": action.reason})
        elif isinstance(action, MoveToAction):
            self._apply_move(state, action, events)
        elif isinstance(action, SpeakAction):
            self._apply_speak(state, action, events)
        elif isinstance(action, InteractObjectAction):
            self._apply_interaction(state, action, events)
        elif isinstance(action, UpdateGoalAction):
            self._apply_goal_update(state, action, events)
        elif isinstance(action, AssistantInterventionAction):
            self._apply_intervention(state, action, events)
        elif isinstance(action, FeedbackToAssistantAction):
            self._apply_feedback(state, action, events)

    def _other_agent_tiles(self, state: WorldState, agent_id: str) -> set[Tile]:
        return {agent.tile for key, agent in state.agents.items() if key != agent_id}

    def _apply_move(
        self,
        state: WorldState,
        action: MoveToAction,
        events: list[SimulationEvent],
    ) -> None:
        agent = state.agents.get(action.agent_id)
        if agent is None:
            self._reject(state, events, reason="unknown agent", action=action, actor_id=action.agent_id)
            return

        path = state.map.find_path(
            agent.tile,
            action.target_tile,
            objects=state.objects.values(),
            extra_blocked=self._other_agent_tiles(state, action.agent_id),
        )
        if path is None:
            self._reject(
                state,
                events,
                reason="target is unreachable or non-walkable",
                action=action,
                actor_id=action.agent_id,
            )
            return

        agent.path = path[1:]
        agent.current_action = "move_to"
        self._event(
            events,
            state,
            EventType.MOVEMENT_PLANNED,
            actor_id=agent.agent_id,
            payload={
                "target_tile": action.target_tile,
                "path": path,
                "reason": action.reason,
            },
        )

    def _apply_speak(
        self,
        state: WorldState,
        action: SpeakAction,
        events: list[SimulationEvent],
    ) -> None:
        if action.agent_id not in state.agents:
            self._reject(state, events, reason="unknown agent", action=action, actor_id=action.agent_id)
            return

        bubble = DialogueBubble(
            agent_id=action.agent_id,
            utterance=action.utterance,
            audience=action.audience,
            started_tick=state.tick,
        )
        state.active_dialogue.append(bubble)
        transcript = TranscriptEvent(
            event_id=f"transcript_{state.tick:04d}_{len(state.transcript_events) + 1:03d}",
            tick=state.tick,
            speaker_agent_id=action.agent_id,
            text=action.utterance,
            topic="meeting",
            derived_intents=[],
        )
        state.transcript_events.append(transcript)
        self._event(
            events,
            state,
            EventType.DIALOGUE,
            actor_id=action.agent_id,
            payload=bubble.model_dump(mode="json"),
        )
        self._event(
            events,
            state,
            EventType.TRANSCRIPT,
            actor_id=action.agent_id,
            target_id=transcript.event_id,
            payload=transcript.model_dump(mode="json"),
        )

    def _apply_interaction(
        self,
        state: WorldState,
        action: InteractObjectAction,
        events: list[SimulationEvent],
    ) -> None:
        agent = state.agents.get(action.agent_id)
        item = state.objects.get(action.object_id)
        if agent is None:
            self._reject(state, events, reason="unknown agent", action=action, actor_id=action.agent_id)
            return
        if item is None:
            self._reject(state, events, reason="unknown object", action=action, actor_id=action.agent_id)
            return
        if not item.can_interact_from(agent.tile):
            self._reject(
                state,
                events,
                reason="agent is not on an interaction tile",
                action=action,
                actor_id=action.agent_id,
                target_id=action.object_id,
            )
            return

        self._mutate_object_state(item, action)
        agent.current_action = f"interact:{item.object_id}"
        self._event(
            events,
            state,
            EventType.OBJECT_INTERACTION,
            actor_id=agent.agent_id,
            target_id=item.object_id,
            payload={
                "interaction": action.interaction,
                "parameters": action.parameters,
                "object_state": item.state,
            },
        )

    def _mutate_object_state(self, item: SimulationObject, action: InteractObjectAction) -> None:
        if action.interaction == ObjectInteraction.TURN_ON:
            if item.type == ObjectType.LIGHT_SWITCH:
                item.state["lights"] = "on"
            else:
                item.state["power"] = "on"
        elif action.interaction == ObjectInteraction.TURN_OFF:
            if item.type == ObjectType.LIGHT_SWITCH:
                item.state["lights"] = "off"
            else:
                item.state["power"] = "off"
        elif action.interaction == ObjectInteraction.OPEN:
            item.state["open"] = True
        elif action.interaction == ObjectInteraction.CLOSE:
            item.state["open"] = False
        elif action.interaction == ObjectInteraction.WRITE:
            item.state["content"] = action.parameters.get("content", item.state.get("content", ""))
            item.state["last_written_by"] = action.agent_id
        elif action.interaction == ObjectInteraction.READ:
            reviewed_by = list(item.state.get("reviewed_by", []))
            if action.agent_id not in reviewed_by:
                reviewed_by.append(action.agent_id)
            item.state["reviewed_by"] = reviewed_by
        elif action.interaction == ObjectInteraction.USE:
            item.state["last_used_by"] = action.agent_id
        elif action.interaction == ObjectInteraction.PICK_UP:
            item.state["held_by"] = action.agent_id
        elif action.interaction == ObjectInteraction.PUT_DOWN:
            item.state["held_by"] = None
        elif action.interaction == ObjectInteraction.SIT:
            item.state["occupied_by"] = action.agent_id

    def _apply_goal_update(
        self,
        state: WorldState,
        action: UpdateGoalAction,
        events: list[SimulationEvent],
    ) -> None:
        agent = state.agents.get(action.agent_id)
        if agent is None:
            self._reject(state, events, reason="unknown agent", action=action, actor_id=action.agent_id)
            return
        old_goal = agent.current_goal
        agent.current_goal = action.goal
        self._event(
            events,
            state,
            EventType.GOAL_UPDATED,
            actor_id=agent.agent_id,
            payload={"old_goal": old_goal, "new_goal": action.goal},
        )

    def _apply_intervention(
        self,
        state: WorldState,
        action: AssistantInterventionAction,
        events: list[SimulationEvent],
    ) -> None:
        agent = state.agents.get(action.target_agent_id)
        if agent is None:
            self._reject(
                state,
                events,
                reason="unknown target agent",
                action=action,
                target_id=action.target_agent_id,
            )
            return
        state.active_interventions.append(action)
        agent.assistant_state.last_intervention_tick = state.tick
        agent.assistant_state.interruption_load = min(
            1.0, agent.assistant_state.interruption_load + 0.15
        )
        agent.public_mood = PublicMood.INTERRUPTED
        self._event(
            events,
            state,
            EventType.ASSISTANT_INTERVENTION,
            actor_id="assistant",
            target_id=agent.agent_id,
            payload=action.model_dump(mode="json"),
        )

    def _apply_feedback(
        self,
        state: WorldState,
        action: FeedbackToAssistantAction,
        events: list[SimulationEvent],
    ) -> None:
        agent = state.agents.get(action.agent_id)
        if agent is None:
            self._reject(state, events, reason="unknown agent", action=action, actor_id=action.agent_id)
            return
        matching = [
            intervention
            for intervention in state.active_interventions
            if intervention.intervention_id == action.intervention_id
        ]
        if not matching:
            self._reject(
                state,
                events,
                reason="unknown active intervention",
                action=action,
                actor_id=action.agent_id,
                target_id=action.intervention_id,
            )
            return

        if action.feedback in {
            AssistantFeedback.ACCEPT,
            AssistantFeedback.DISMISS,
            AssistantFeedback.VERBAL_REJECT,
            AssistantFeedback.TASK_DISRUPTION,
        }:
            state.active_interventions = [
                intervention
                for intervention in state.active_interventions
                if intervention.intervention_id != action.intervention_id
            ]
        agent.assistant_state.interruption_load = max(
            0.0,
            agent.assistant_state.interruption_load - 0.05
            if action.feedback == AssistantFeedback.ACCEPT
            else agent.assistant_state.interruption_load,
        )
        if action.feedback == AssistantFeedback.ACCEPT:
            agent.public_mood = PublicMood.FOCUSED

        self._event(
            events,
            state,
            EventType.ASSISTANT_FEEDBACK,
            actor_id=agent.agent_id,
            target_id=action.intervention_id,
            payload=action.model_dump(mode="json"),
        )

    def _advance_agent_paths(
        self,
        state: WorldState,
        events: list[SimulationEvent],
    ) -> None:
        for agent_id in sorted(state.agents):
            agent = state.agents[agent_id]
            if not agent.path:
                if agent.current_action == "move_to":
                    agent.current_action = None
                continue
            next_tile = agent.path[0]
            if not state.map.is_walkable(
                next_tile,
                objects=state.objects.values(),
                extra_blocked=self._other_agent_tiles(state, agent_id),
            ):
                self._event(
                    events,
                    state,
                    EventType.ACTION_REJECTED,
                    actor_id=agent.agent_id,
                    payload={
                        "reason": "reserved or blocked next path tile",
                        "next_tile": next_tile,
                    },
                )
                continue
            previous_tile = agent.tile
            agent.tile = next_tile
            agent.path = agent.path[1:]
            agent.facing = facing_from_delta(previous_tile, next_tile)
            if not agent.path and agent.current_action == "move_to":
                agent.current_action = None
            self._event(
                events,
                state,
                EventType.AGENT_MOVED,
                actor_id=agent.agent_id,
                payload={
                    "from_tile": previous_tile,
                    "to_tile": next_tile,
                    "remaining_path": agent.path,
                },
            )


def facing_from_delta(previous: Tile, current: Tile) -> Facing:
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    if dx > 0:
        return Facing.EAST
    if dx < 0:
        return Facing.WEST
    if dy > 0:
        return Facing.SOUTH
    return Facing.NORTH
