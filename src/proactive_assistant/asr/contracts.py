from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpeechTranscriptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    text: str
    language: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    duration_ms: int | None = Field(default=None, ge=0)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
