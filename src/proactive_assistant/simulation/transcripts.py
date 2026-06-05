from pydantic import BaseModel, ConfigDict, Field


class TranscriptEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    tick: int = Field(ge=0)
    speaker_agent_id: str
    text: str = Field(min_length=1)
    topic: str = "general"
    derived_intents: list[str] = Field(default_factory=list)
