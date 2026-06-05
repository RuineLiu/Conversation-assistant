from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.simulation.actions import AssistantInterventionAction
from proactive_assistant.simulation.agents import AgentState, default_agents
from proactive_assistant.simulation.events import SimulationEvent
from proactive_assistant.simulation.map import ConferenceRoomMap, default_conference_room_map
from proactive_assistant.simulation.objects import (
    SimulationObject,
    default_conference_room_objects,
)
from proactive_assistant.simulation.transcripts import TranscriptEvent


class DialogueBubble(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    utterance: str
    audience: list[str] = Field(default_factory=list)
    started_tick: int = Field(ge=0)


class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str
    tick: int = Field(default=0, ge=0)
    scenario_id: str = "conference_room_meeting"
    map: ConferenceRoomMap
    agents: dict[str, AgentState]
    objects: dict[str, SimulationObject]
    active_dialogue: list[DialogueBubble] = Field(default_factory=list)
    active_interventions: list[AssistantInterventionAction] = Field(default_factory=list)
    transcript_events: list[TranscriptEvent] = Field(default_factory=list)
    recent_events: list[SimulationEvent] = Field(default_factory=list)
    event_log: list[SimulationEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def agents_must_start_on_walkable_tiles(self) -> "WorldState":
        occupied: set[tuple[int, int]] = set()
        for agent in self.agents.values():
            if agent.tile in occupied:
                raise ValueError(f"multiple agents occupy tile {agent.tile}")
            occupied.add(agent.tile)
            if not self.map.is_walkable(agent.tile, objects=self.objects.values()):
                raise ValueError(f"agent starts on non-walkable tile: {agent.agent_id}")
        return self


def create_default_world_state(simulation_id: str = "meeting_demo_001") -> WorldState:
    room_map = default_conference_room_map()
    objects = default_conference_room_objects()
    return WorldState(
        simulation_id=simulation_id,
        map=room_map,
        objects=objects,
        agents=default_agents(room_map),
        metadata={
            "phase": "phase_1_backend_kernel",
            "visualization": "2d_tile_sandbox",
        },
    )
