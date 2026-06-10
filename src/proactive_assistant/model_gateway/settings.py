from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelGatewaySettings(BaseSettings):
    """Runtime settings for model gateway calls.

    Values can be provided either with generic OpenAI names or PROACTIVE_* names.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "PROACTIVE_OPENAI_API_KEY"),
    )
    default_model: str = Field(
        default="gpt-5.2",
        validation_alias=AliasChoices("OPENAI_MODEL", "PROACTIVE_OPENAI_MODEL"),
    )
    fast_model: str = Field(
        default="gpt-5-mini",
        validation_alias=AliasChoices("OPENAI_FAST_MODEL", "PROACTIVE_OPENAI_FAST_MODEL"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices(
            "OPENAI_EMBEDDING_MODEL",
            "PROACTIVE_OPENAI_EMBEDDING_MODEL",
            "EMBEDDING_MODEL",
        ),
    )
    max_output_tokens: int = Field(
        default=1200,
        ge=1,
        validation_alias=AliasChoices(
            "OPENAI_MAX_OUTPUT_TOKENS",
            "PROACTIVE_OPENAI_MAX_OUTPUT_TOKENS",
        ),
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias=AliasChoices(
            "OPENAI_REQUEST_TIMEOUT_SECONDS",
            "PROACTIVE_OPENAI_REQUEST_TIMEOUT_SECONDS",
        ),
    )
    glasses_prompt_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        validation_alias=AliasChoices(
            "PROACTIVE_GLASSES_PROMPT_TIMEOUT_SECONDS",
            "GLASSES_PROMPT_TIMEOUT_SECONDS",
        ),
    )
    store_model_responses: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "OPENAI_STORE_RESPONSES",
            "PROACTIVE_OPENAI_STORE_RESPONSES",
        ),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_BASE_URL",
            "PROACTIVE_OPENAI_BASE_URL",
            "MODEL_API_BASE_URL",
            "LLM_BASE_URL",
        ),
    )
    model_api_style: str = Field(
        default="responses",
        validation_alias=AliasChoices(
            "OPENAI_API_STYLE",
            "PROACTIVE_OPENAI_API_STYLE",
            "MODEL_API_STYLE",
        ),
    )
    chat_response_format: str = Field(
        default="json_schema",
        validation_alias=AliasChoices(
            "OPENAI_CHAT_RESPONSE_FORMAT",
            "PROACTIVE_OPENAI_CHAT_RESPONSE_FORMAT",
        ),
    )
    chat_max_tokens_param: str = Field(
        default="max_tokens",
        validation_alias=AliasChoices(
            "OPENAI_CHAT_MAX_TOKENS_PARAM",
            "PROACTIVE_OPENAI_CHAT_MAX_TOKENS_PARAM",
        ),
    )
