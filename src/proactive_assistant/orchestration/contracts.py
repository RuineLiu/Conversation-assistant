from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.detection import PromptOpportunity
from proactive_assistant.prompting import PromptGenerationRequest, PromptGenerationResult
from proactive_assistant.sessions import SessionContextSnapshot


class PromptCandidateStatus(StrEnum):
    GENERATED = "generated"
    SUPPRESSED = "suppressed"
    GENERATION_FAILED = "generation_failed"


class PromptCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    candidate_id: str
    session_id: str
    opportunity: PromptOpportunity
    prompt_request: PromptGenerationRequest
    status: PromptCandidateStatus
    prompt_result: PromptGenerationResult | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def status_must_match_result(self) -> "PromptCandidate":
        status = _enum_value(self.status)
        if status == PromptCandidateStatus.GENERATED.value:
            if self.prompt_result is None:
                raise ValueError("generated prompt candidates must include prompt_result")
            if not self.prompt_result.should_prompt:
                raise ValueError("generated prompt candidates require should_prompt=true")
        if status == PromptCandidateStatus.SUPPRESSED.value:
            if self.prompt_result is None:
                raise ValueError("suppressed prompt candidates must include prompt_result")
            if self.prompt_result.should_prompt:
                raise ValueError("suppressed prompt candidates require should_prompt=false")
        if status == PromptCandidateStatus.GENERATION_FAILED.value and not self.reason:
            raise ValueError("failed prompt candidates must include a reason")
        return self


class PromptOrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    snapshot: SessionContextSnapshot
    opportunities: list[PromptOpportunity] = Field(default_factory=list)
    candidates: list[PromptCandidate] = Field(default_factory=list)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value
