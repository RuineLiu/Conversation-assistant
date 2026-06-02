from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FeedbackType(StrEnum):
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    TIMING_NEGATIVE = "timing_negative"
    CONTENT_NEGATIVE = "content_negative"
    IMPLICIT_ACCEPT = "implicit_accept"
    IMPLICIT_REJECT = "implicit_reject"
    NO_FEEDBACK = "no_feedback"
    JUDGE = "judge"


class FeedbackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    feedback_type: FeedbackType
    explicit_text: str | None = None
    accept: float = Field(default=0.0, ge=0.0, le=1.0)
    annoyance: float = Field(default=0.0, ge=0.0, le=1.0)
    helpfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    timing_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    content_fit: float = Field(default=0.0, ge=-1.0, le=1.0)
    task_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    flow_break: float = Field(default=0.0, ge=0.0, le=1.0)
    redundancy: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    cognitive_load: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    reward: float
