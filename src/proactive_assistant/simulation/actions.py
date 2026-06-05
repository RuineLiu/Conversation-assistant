from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from proactive_assistant.simulation.coordinates import Tile, validate_tile


class ActionType(StrEnum):
    MOVE_TO = "move_to"
    SPEAK = "speak"
    INTERACT_OBJECT = "interact_object"
    UPDATE_GOAL = "update_goal"
    ASSISTANT_INTERVENTION = "assistant_intervention"
    FEEDBACK_TO_ASSISTANT = "feedback_to_assistant"
    NO_OP = "no_op"


class ObjectInteraction(StrEnum):
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    OPEN = "open"
    CLOSE = "close"
    WRITE = "write"
    READ = "read"
    USE = "use"
    PICK_UP = "pick_up"
    PUT_DOWN = "put_down"
    SIT = "sit"


class TimingPolicy(StrEnum):
    BEFORE_ACTIVITY = "before_activity"
    DURING_ACTIVITY = "during_activity"
    AFTER_ACTIVITY = "after_activity"
    MANUAL = "manual"


class AssistantChannel(StrEnum):
    SMART_GLASSES_OVERLAY = "smart_glasses_overlay"
    AUDIO_CUE = "audio_cue"
    SUBTLE_NOTIFICATION = "subtle_notification"


class AssistantFeedback(StrEnum):
    ACCEPT = "accept"
    IGNORE = "ignore"
    DISMISS = "dismiss"
    SNOOZE = "snooze"
    ASK_FOLLOWUP = "ask_followup"
    VERBAL_REJECT = "verbal_reject"
    TASK_DISRUPTION = "task_disruption"


class BaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoOpAction(BaseAction):
    type: Literal["no_op"] = "no_op"
    reason: str = ""


class MoveToAction(BaseAction):
    type: Literal["move_to"] = "move_to"
    agent_id: str
    target_tile: Tile
    reason: str = ""

    @model_validator(mode="after")
    def target_tile_must_be_valid(self) -> "MoveToAction":
        validate_tile(self.target_tile, field_name="target_tile")
        return self


class SpeakAction(BaseAction):
    type: Literal["speak"] = "speak"
    agent_id: str
    utterance: str = Field(min_length=1)
    audience: list[str] = Field(default_factory=list)


class InteractObjectAction(BaseAction):
    type: Literal["interact_object"] = "interact_object"
    agent_id: str
    object_id: str
    interaction: ObjectInteraction
    parameters: dict[str, Any] = Field(default_factory=dict)


class UpdateGoalAction(BaseAction):
    type: Literal["update_goal"] = "update_goal"
    agent_id: str
    goal: str = Field(min_length=1)


class AssistantInterventionAction(BaseAction):
    type: Literal["assistant_intervention"] = "assistant_intervention"
    intervention_id: str
    target_agent_id: str
    timing_policy: TimingPolicy
    channel: AssistantChannel = AssistantChannel.SMART_GLASSES_OVERLAY
    content: str = Field(min_length=1)
    detail_level: float = Field(ge=0.0, le=1.0)
    context_refs: list[str] = Field(default_factory=list)


class FeedbackToAssistantAction(BaseAction):
    type: Literal["feedback_to_assistant"] = "feedback_to_assistant"
    agent_id: str
    intervention_id: str
    feedback: AssistantFeedback
    task_disruption: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


SimulationAction = Annotated[
    NoOpAction
    | MoveToAction
    | SpeakAction
    | InteractObjectAction
    | UpdateGoalAction
    | AssistantInterventionAction
    | FeedbackToAssistantAction,
    Field(discriminator="type"),
]

_ACTION_ADAPTER: TypeAdapter[SimulationAction] = TypeAdapter(SimulationAction)


def parse_simulation_action(payload: SimulationAction | dict[str, Any]) -> SimulationAction:
    if isinstance(payload, BaseAction):
        return payload
    return _ACTION_ADAPTER.validate_python(payload)
