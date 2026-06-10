from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.prompting import PRDSurface, PrivacyLevel, PromptCategory


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
    PROMOTED = "promoted"
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
    FORGOTTEN = "forgotten"


class MemoryUpsertStatus(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class MemoryUpdatePolicyDecision(StrEnum):
    AUTO_UPDATE = "auto_update"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"


class MemoryPendingUpdateStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"


class MemoryPromotionStatus(StrEnum):
    PROMOTED = "promoted"
    UNCHANGED = "unchanged"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"


class MemoryMergeAction(StrEnum):
    CREATE = "create"
    DUPLICATE = "duplicate"
    REINFORCEMENT = "reinforcement"
    UPDATE = "update"
    CONFLICT = "conflict"
    SUPERSEDE = "supersede"
    BLOCKED = "blocked"


class MemoryRetrievalIntent(StrEnum):
    LOOKUP_DEADLINE = "lookup_deadline"
    LOOKUP_OWNER = "lookup_owner"
    LOOKUP_STATUS = "lookup_status"
    LOOKUP_RATIONALE = "lookup_rationale"
    LOOKUP_TASK_LIST = "lookup_task_list"
    LOOKUP_SCHEDULE = "lookup_schedule"
    OPEN_RECALL = "open_recall"


class MemoryUsePolicy(StrEnum):
    PROMPT_CONTEXT = "prompt_context"
    POLICY_HINT = "policy_hint"
    DISPLAY_REF_ONLY = "display_ref_only"
    BLOCKED_BY_PRIVACY = "blocked_by_privacy"


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
    source: MemorySource | None = None
    source_ids: list[str] | None = None
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
    include_forgotten: bool = False
    include_pending: bool = False
    limit: int = Field(default=20, ge=1, le=200)
    prompt_category: PromptCategory | None = None
    activity_phase: str = "discussion"
    current_gap_types: list[str] = Field(default_factory=list)
    recent_transcript_text: str = ""
    active_entities: list[dict[str, Any]] = Field(default_factory=list)
    target_entity: str | None = None
    time_window_start: str | None = None
    time_window_end: str | None = None
    privacy_constraints: list[str] = Field(default_factory=list)
    prd_surface: PRDSurface | None = None
    reference_time: datetime | None = None
    shown_memory_ids: list[str] = Field(default_factory=list)
    use_semantic_retrieval: bool = False
    semantic_min_score: float = Field(default=0.25, ge=0.0, le=1.0)


class MemorySearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    memory: MemoryRecord
    score: float = Field(ge=0.0)
    matched_terms: list[str] = Field(default_factory=list)
    reason: str = ""
    rank_features: dict[str, float] = Field(default_factory=dict)
    use_policy: MemoryUsePolicy = MemoryUsePolicy.PROMPT_CONTEXT
    provenance: list[str] = Field(default_factory=list)
    intent: MemoryRetrievalIntent = MemoryRetrievalIntent.OPEN_RECALL
    target_entity: str | None = None


class MemoryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_context: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    results: list[MemorySearchResult] = Field(default_factory=list)
    # P2-5: distinguishes "the user has no memory yet" from "memory exists
    # but did not match this query". Allows the UI to suppress the
    # "I remember from before..." region entirely on first use rather
    # than render an empty placeholder.
    is_empty_cold_start: bool = False


class MemoryMergeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    action: MemoryMergeAction = MemoryMergeAction.CREATE
    existing_memory_id: str | None = None
    proposed_memory_id: str | None = None
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    changed_fields: list[str] = Field(default_factory=list)
    conflict_fields: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    resolution_status: str = "resolved"


class MemoryUpsertResult(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    status: MemoryUpsertStatus
    memory: MemoryRecord
    previous_memory: MemoryRecord | None = None
    proposed_memory: MemoryRecord | None = None
    changed_fields: list[str] = Field(default_factory=list)
    previous_digest: str | None = None
    new_digest: str
    version: int = Field(ge=1)
    update_policy: MemoryUpdatePolicyDecision = MemoryUpdatePolicyDecision.AUTO_UPDATE
    policy_reasons: list[str] = Field(default_factory=list)
    pending_update: "MemoryPendingUpdate | None" = None
    merge_decision: MemoryMergeDecision | None = None


class MemoryPendingUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    update_id: str = Field(min_length=1)
    memory_id: str = Field(min_length=1)
    proposed_memory: MemoryRecord
    previous_digest: str
    new_digest: str
    changed_fields: list[str] = Field(default_factory=list)
    update_policy: MemoryUpdatePolicyDecision = MemoryUpdatePolicyDecision.NEEDS_CONFIRMATION
    policy_reasons: list[str] = Field(default_factory=list)
    status: MemoryPendingUpdateStatus = MemoryPendingUpdateStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    resolved_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryForgetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: MemoryRecord
    invalidated_pending_updates: list[MemoryPendingUpdate] = Field(default_factory=list)
    reason: str = ""
    redacted: bool = True


class MemoryPromotionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    status: MemoryPromotionStatus
    source_memory: MemoryRecord
    promoted_memory: MemoryRecord | None = None
    proposed_memory: MemoryRecord | None = None
    target_scope: MemoryScope
    target_memory_type: MemoryType
    policy_reasons: list[str] = Field(default_factory=list)
    approved: bool = False
