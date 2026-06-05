from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptCategory(StrEnum):
    QUESTION_ANSWER = "question_answer"
    CONCEPT_EXPLANATION = "concept_explanation"
    PERSON_OR_FACT = "person_or_fact"
    SUGGESTION = "suggestion"
    SUMMARY_GAP_CHECK = "summary_gap_check"


class ContentGranularity(IntEnum):
    NO_ACTION = 0
    ICON_TITLE_ONLY = 1
    ONE_LINE_ANSWER = 2
    CONCISE_BULLETS = 3
    DETAILED_WITH_CONTEXT = 4


class PRDSurface(StrEnum):
    GLASSES_STARTING = "glasses_starting"
    GLASSES_POPUP = "glasses_popup"
    GLASSES_PERSISTENT = "glasses_persistent"
    APP_PROMPT_TAB = "app_prompt_tab"
    APP_SUMMARY_TAB = "app_summary_tab"
    APP_SUGGEST = "app_suggest"


class DisplayMode(StrEnum):
    AUTO = "auto"
    WRIST_TURN = "wrist_turn"
    HEAD_RAISE = "head_raise"


class DurationPolicy(StrEnum):
    AUTO = "auto"
    THREE_SECONDS = "3s"
    FIVE_SECONDS = "5s"
    EIGHT_SECONDS = "8s"


class PrivacyLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TranscriptWindowItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript_id: str
    speaker: str
    text: str = Field(min_length=1)
    timestamp_ms: int | None = Field(default=None, ge=0)
    topic: str | None = None


class ModelUsageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    raw_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached: bool = False


class PromptGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    session_id: str
    scenario_id: str = "meeting_business_v1"
    locale: str = "zh-CN"
    transcript_window: list[TranscriptWindowItem] = Field(min_length=1)
    session_context: dict[str, Any] = Field(default_factory=dict)
    persona_context: dict[str, Any] = Field(default_factory=dict)
    memory_context: list[str] = Field(default_factory=list)
    privacy_constraints: list[str] = Field(default_factory=list)
    prompt_category_candidate: PromptCategory | None = None
    target_content_granularity: ContentGranularity = ContentGranularity.ONE_LINE_ANSWER
    prd_surface: PRDSurface = PRDSurface.GLASSES_POPUP
    display_mode: DisplayMode = DisplayMode.AUTO
    duration_policy: DurationPolicy = DurationPolicy.AUTO


class PromptGenerationModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    should_prompt: bool
    prompt_category: PromptCategory | None = None
    content_granularity: ContentGranularity
    glasses_title: str = ""
    glasses_text: str = ""
    app_detail_text: str = ""
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    privacy_risk: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    safety_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def result_must_match_prompt_decision(self) -> "PromptGenerationResult":
        if not self.should_prompt:
            if self.content_granularity != ContentGranularity.NO_ACTION:
                raise ValueError("no-prompt results must use content_granularity=0")
            return self

        if self.prompt_category is None:
            raise ValueError("prompt_category is required when should_prompt=true")
        if self.content_granularity == ContentGranularity.NO_ACTION:
            raise ValueError("prompt results must use content_granularity greater than 0")
        if not self.glasses_title:
            raise ValueError("glasses_title is required when should_prompt=true")
        if self.content_granularity >= ContentGranularity.ONE_LINE_ANSWER and not self.glasses_text:
            raise ValueError("glasses_text is required for content_granularity>=2")
        if not self.source_refs:
            raise ValueError("source_refs are required when should_prompt=true")
        return self


class PromptGenerationResult(PromptGenerationModelOutput):
    model_usage: ModelUsageMetadata | None = None

    def with_usage(self, usage: ModelUsageMetadata) -> "PromptGenerationResult":
        return self.model_copy(update={"model_usage": usage})


def openai_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return an OpenAI structured-output schema with strict object settings.

    OpenAI strict structured outputs work best when object schemas forbid extra
    properties and every property is listed in `required`. Optional semantics are
    represented by nullable types in the schema, not by omitted keys.
    """

    schema = model.model_json_schema()
    _make_schema_strict(schema)
    return schema


def _make_schema_strict(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties.keys())
        for value in node.values():
            _make_schema_strict(value)
    elif isinstance(node, list):
        for item in node:
            _make_schema_strict(item)
