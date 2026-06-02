from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActivityPhase(StrEnum):
    PRE_ACTIVITY = "pre_activity"
    IN_ACTIVITY = "in_activity"
    POST_ACTIVITY = "post_activity"
    IDLE = "idle"
    UNCERTAIN = "uncertain"


class ActivityPhaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: ActivityPhase
    duration_min: float = Field(gt=0)
    observable_transcript: list[str] = Field(default_factory=list)
    latent_risks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_observable_signal(self) -> "ActivityPhaseSpec":
        if not self.observable_transcript and self.phase != ActivityPhase.IDLE:
            raise ValueError("non-idle phases require at least one observable transcript segment")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    category: str
    hidden_user_goal: str
    activity_phases: list[ActivityPhaseSpec] = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
