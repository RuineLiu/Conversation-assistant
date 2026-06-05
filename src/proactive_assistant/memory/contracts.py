from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.prompting import PrivacyLevel


class MemoryType(StrEnum):
    USER_PREFERENCE = "user_preference"
    NEGATIVE_PREFERENCE = "negative_preference"
    PRIVACY_PREFERENCE = "privacy_preference"
    MEETING_FACT = "meeting_fact"
    ACTION_ITEM = "action_item"
    DECISION = "decision"
    PERSON_OR_FACT = "person_or_fact"
    PROJECT_CONTEXT = "project_context"
    SUMMARY = "summary"


class MemoryScope(StrEnum):
    USER = "user"
    ORG = "org"
    SESSION = "session"
    GLOBAL = "global"


class MemorySource(StrEnum):
    MANUAL = "manual"
    FEEDBACK_CANDIDATE = "feedback_candidate"
    TRANSCRIPT = "transcript"
    MEETING_STATE = "meeting_state"
    IMPORTED = "imported"


class RetentionPolicy(StrEnum):
    SESSION_ONLY = "session_only"
    THIRTY_DAYS = "30d"
    NINETY_DAYS = "90d"
    PERSISTENT = "persistent"
    UNTIL_REVOKED = "until_revoked"


class MemoryWriteStatus(StrEnum):
    PENDING_CONFIRMATION = "pending_confirmation"
    ACTIVE = "active"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    memory_id: str = Field(min_length=1)
    memory_type: MemoryType
    scope: MemoryScope
    text: str = Field(min_length=1)
    org_id: str = Field(default="default_org", min_length=1)
    user_id: str = Field(default="default_user", min_length=1)
    session_id: str | None = None
    source: MemorySource
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    retention_policy: RetentionPolicy = RetentionPolicy.PERSISTENT
    write_status: MemoryWriteStatus = MemoryWriteStatus.ACTIVE
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def scoped_memories_require_matching_ids(self) -> "MemoryRecord":
        scope = MemoryScope(self.scope)
        if scope == MemoryScope.SESSION and not self.session_id:
            raise ValueError("session scoped memories require session_id")
        return self


class MemoryRecordUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    memory_type: MemoryType | None = None
    scope: MemoryScope | None = None
    text: str | None = Field(default=None, min_length=1)
    org_id: str | None = Field(default=None, min_length=1)
    user_id: str | None = Field(default=None, min_length=1)
    session_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    privacy_level: PrivacyLevel | None = None
    retention_policy: RetentionPolicy | None = None
    write_status: MemoryWriteStatus | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class MemoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    query_text: str = ""
    org_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    memory_types: list[MemoryType] = Field(default_factory=list)
    scopes: list[MemoryScope] = Field(default_factory=list)
    privacy_levels: list[PrivacyLevel] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    include_archived: bool = False
    include_pending: bool = False
    limit: int = Field(default=20, ge=1, le=200)


class MemorySearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: MemoryRecord
    score: float = Field(ge=0.0)
    matched_terms: list[str] = Field(default_factory=list)
    reason: str = ""


class MemoryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_context: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    results: list[MemorySearchResult] = Field(default_factory=list)
