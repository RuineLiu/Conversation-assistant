from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionType(StrEnum):
    FACT = "fact"
    PERSON = "person"
    CONCEPT = "concept"
    DATA = "data"
    REASON = "reason"
    OTHER = "other"


class ActionItemStatus(StrEnum):
    OPEN = "open"
    ASSIGNED = "assigned"
    DONE = "done"


class MentionedRefType(StrEnum):
    PREVIOUS_MEETING = "previous_meeting"
    COMMITMENT = "commitment"
    PERSON = "person"
    DEADLINE = "deadline"
    DOCUMENT = "document"
    OTHER = "other"


class MeetingGapType(StrEnum):
    UNANSWERED_QUESTION = "unanswered_question"
    ACTION_MISSING_OWNER = "action_missing_owner"
    ACTION_MISSING_DEADLINE = "action_missing_deadline"
    ACTION_MISSING_NEXT_STEP = "action_missing_next_step"
    DECISION_MISSING_CONCLUSION = "decision_missing_conclusion"
    OPEN_RISK = "open_risk"
    END_SUMMARY_NEEDED = "end_summary_needed"


class MeetingGapPriority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class MeetingUtterance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance_id: str
    session_id: str
    speaker: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    topic: str | None = None
    asr_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> "MeetingUtterance":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class OpenQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    question_id: str
    text: str = Field(min_length=1)
    asker: str
    source_utterance_id: str
    ts_ms: int = Field(ge=0)
    question_type: QuestionType = QuestionType.OTHER
    answered: bool = False
    answer: str | None = None
    answer_utterance_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    action_item_id: str
    desc: str = Field(min_length=1)
    source_utterance_id: str
    source_ts_ms: int = Field(ge=0)
    owner: str | None = None
    deadline: str | None = None
    next_step: str | None = None
    status: ActionItemStatus = ActionItemStatus.OPEN
    evidence: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    topic: str = Field(min_length=1)
    source_utterance_id: str
    ts_ms: int = Field(ge=0)
    conclusion: str | None = None
    evidence: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Risk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_id: str
    desc: str = Field(min_length=1)
    source_utterance_id: str
    ts_ms: int = Field(ge=0)
    closed: bool = False
    evidence: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MentionedRef(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    mentioned_ref_id: str
    ref_type: MentionedRefType
    text: str = Field(min_length=1)
    source_utterance_id: str
    ts_ms: int = Field(ge=0)
    resolved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeetingGap(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    gap_id: str
    session_id: str
    gap_type: MeetingGapType
    priority: MeetingGapPriority
    object_type: str
    object_id: str
    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    source_utterance_ids: list[str] = Field(default_factory=list)
    first_seen_ms: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeetingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    org_id: str = "default_org"
    subject_user_id: str = "default_user"
    participants: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    scheduled_end_ms: int | None = Field(default=None, ge=0)
    utterances: list[MeetingUtterance] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    mentioned_refs: list[MentionedRef] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeetingStateUpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: MeetingState
    added_questions: list[OpenQuestion] = Field(default_factory=list)
    answered_questions: list[OpenQuestion] = Field(default_factory=list)
    added_action_items: list[ActionItem] = Field(default_factory=list)
    added_decisions: list[Decision] = Field(default_factory=list)
    updated_decisions: list[Decision] = Field(default_factory=list)
    added_risks: list[Risk] = Field(default_factory=list)
    added_refs: list[MentionedRef] = Field(default_factory=list)
    gaps: list[MeetingGap] = Field(default_factory=list)
