from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureSpeechSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    speech_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_SPEECH_KEY",
            "SPEECH_KEY",
            "MICROSOFT_SPEECH_KEY",
            "AZURE_ASR_KEY",
            "PROACTIVE_AZURE_SPEECH_KEY",
        ),
    )
    speech_region: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_SPEECH_REGION",
            "SPEECH_REGION",
            "MICROSOFT_SPEECH_REGION",
            "AZURE_ASR_REGION",
            "PROACTIVE_AZURE_SPEECH_REGION",
        ),
    )
    speech_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_SPEECH_ENDPOINT",
            "SPEECH_ENDPOINT",
            "MICROSOFT_SPEECH_ENDPOINT",
            "AZURE_ASR_ENDPOINT",
            "PROACTIVE_AZURE_SPEECH_ENDPOINT",
        ),
    )
    speech_language: str = Field(
        default="zh-CN",
        validation_alias=AliasChoices(
            "AZURE_SPEECH_LANGUAGE",
            "SPEECH_LANGUAGE",
            "AZURE_ASR_LANGUAGE",
            "PROACTIVE_AZURE_SPEECH_LANGUAGE",
        ),
    )
    speech_content_type: str = Field(
        default="audio/wav; codecs=audio/pcm; samplerate=16000",
        validation_alias=AliasChoices(
            "AZURE_SPEECH_CONTENT_TYPE",
            "SPEECH_CONTENT_TYPE",
            "AZURE_ASR_CONTENT_TYPE",
            "PROACTIVE_AZURE_SPEECH_CONTENT_TYPE",
        ),
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices(
            "AZURE_SPEECH_TIMEOUT_SECONDS",
            "SPEECH_TIMEOUT_SECONDS",
            "PROACTIVE_AZURE_SPEECH_TIMEOUT_SECONDS",
        ),
    )

    @property
    def is_configured(self) -> bool:
        return self.speech_key is not None and bool(self.speech_region or self.speech_endpoint)
