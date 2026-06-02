from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    session_id: str
    speaker: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    asr_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> "TranscriptSegment":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self
