from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.schemas.scenario import ActivityPhase


class InterventionActionType(StrEnum):
    NO_ACTION = "NO_ACTION"
    PRE_PLAN = "PRE_PLAN"
    PRE_CHECKLIST = "PRE_CHECKLIST"
    IN_TASK_HINT = "IN_TASK_HINT"
    IN_TASK_HELP = "IN_TASK_HELP"
    ASK_PERMISSION = "ASK_PERMISSION"
    POST_SUMMARY = "POST_SUMMARY"
    POST_GAP_CHECK = "POST_GAP_CHECK"


class OutputModality(StrEnum):
    SILENT_CARD = "silent_card"
    SUBTLE_HAPTIC = "subtle_haptic"
    AUDIO_WHISPER = "audio_whisper"
    BLOCKING_PROMPT = "blocking_prompt"


class CandidateIntervention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: InterventionActionType
    phase: ActivityPhase
    content_level: int = Field(ge=0, le=4)
    display_text: str = ""
    reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_interrupt_cost: float = Field(ge=0.0, le=1.0)
    estimated_help_value: float = Field(ge=0.0, le=1.0)
    output_modality: OutputModality = OutputModality.SILENT_CARD
    safety_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_content_for_interventions(self) -> "CandidateIntervention":
        if self.action_type != InterventionActionType.NO_ACTION and not self.display_text:
            raise ValueError("display_text is required for intervention actions")
        return self


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    context_hash: str
    available_actions: list[InterventionActionType] = Field(min_length=1)
    chosen_action: InterventionActionType
    action_probability: float = Field(gt=0.0, le=1.0)
    content_level: int = Field(ge=0, le=4)
    display_text: str = ""
    policy_version: str

    @model_validator(mode="after")
    def chosen_action_must_be_available(self) -> "PolicyDecision":
        if self.chosen_action not in self.available_actions:
            raise ValueError("chosen_action must be present in available_actions")
        return self
