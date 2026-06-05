from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    ACTION_REJECTED = "action_rejected"
    MOVEMENT_PLANNED = "movement_planned"
    AGENT_MOVED = "agent_moved"
    DIALOGUE = "dialogue"
    OBJECT_INTERACTION = "object_interaction"
    GOAL_UPDATED = "goal_updated"
    ASSISTANT_INTERVENTION = "assistant_intervention"
    ASSISTANT_FEEDBACK = "assistant_feedback"
    TRANSCRIPT = "transcript"
    USER_INSTRUCTION = "user_instruction"
    NO_OP = "no_op"


class SimulationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    tick: int = Field(ge=0)
    event_type: EventType
    actor_id: str | None = None
    target_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
