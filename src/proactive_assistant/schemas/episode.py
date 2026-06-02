from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.schemas.feedback import FeedbackEvent
from proactive_assistant.schemas.intervention import PolicyDecision
from proactive_assistant.schemas.transcript import TranscriptSegment


class EpisodeTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    persona_id: str
    scenario_id: str
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    decisions: list[PolicyDecision] = Field(default_factory=list)
    feedback_events: list[FeedbackEvent] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed: bool = False
    total_reward: float | None = None

    @model_validator(mode="after")
    def feedback_must_reference_decisions(self) -> "EpisodeTrajectory":
        decision_ids = {decision.decision_id for decision in self.decisions}
        dangling = [
            feedback.decision_id
            for feedback in self.feedback_events
            if feedback.decision_id not in decision_ids
        ]
        if dangling:
            raise ValueError(f"feedback references unknown decisions: {dangling}")
        return self
