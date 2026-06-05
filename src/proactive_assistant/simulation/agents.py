from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.simulation.coordinates import Tile, validate_tile
from proactive_assistant.simulation.map import ConferenceRoomMap


class AgentRole(StrEnum):
    PRESENTER = "presenter"
    PARTICIPANT = "participant"
    FACILITATOR = "facilitator"


class Facing(StrEnum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"


class PublicMood(StrEnum):
    FOCUSED = "focused"
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    INTERRUPTED = "interrupted"


class AssistantState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wearing_device: bool = True
    last_intervention_tick: int | None = None
    interruption_load: float = Field(default=0.0, ge=0.0, le=1.0)


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    persona_id: str
    display_name: str
    role: AgentRole
    tile: Tile
    facing: Facing = Facing.SOUTH
    current_goal: str = ""
    current_action: str | None = None
    path: list[Tile] = Field(default_factory=list)
    public_mood: PublicMood = PublicMood.NEUTRAL
    assistant_state: AssistantState = Field(default_factory=AssistantState)

    @model_validator(mode="after")
    def coordinates_must_be_valid(self) -> "AgentState":
        validate_tile(self.tile)
        for tile in self.path:
            validate_tile(tile, field_name="path tile")
        return self


def default_agents(room_map: ConferenceRoomMap) -> dict[str, AgentState]:
    return {
        "agent_alex": AgentState(
            agent_id="agent_alex",
            persona_id="persona_alex",
            display_name="Alex",
            role=AgentRole.PRESENTER,
            tile=room_map.spawn_points["presenter"],
            facing=Facing.EAST,
            current_goal="prepare presentation",
            public_mood=PublicMood.FOCUSED,
        ),
        "agent_bao": AgentState(
            agent_id="agent_bao",
            persona_id="persona_bao",
            display_name="Bao",
            role=AgentRole.PARTICIPANT,
            tile=room_map.spawn_points["participant_a"],
            facing=Facing.WEST,
            current_goal="review meeting documents",
            public_mood=PublicMood.NEUTRAL,
        ),
        "agent_chris": AgentState(
            agent_id="agent_chris",
            persona_id="persona_chris",
            display_name="Chris",
            role=AgentRole.FACILITATOR,
            tile=room_map.spawn_points["participant_b"],
            facing=Facing.WEST,
            current_goal="capture action items",
            public_mood=PublicMood.FOCUSED,
        ),
    }
