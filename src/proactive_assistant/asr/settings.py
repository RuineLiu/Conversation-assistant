from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureSpeechSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
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


class AliyunSpeechSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "ALIYUN_ASR_API_KEY",
            "DASHSCOPE_API_KEY",
            "ALIBABA_CLOUD_API_KEY",
            "PROACTIVE_ALIYUN_ASR_API_KEY",
        ),
    )
    region: str = Field(
        default="beijing",
        validation_alias=AliasChoices(
            "ALIYUN_ASR_REGION",
            "DASHSCOPE_REGION",
            "PROACTIVE_ALIYUN_ASR_REGION",
        ),
    )
    endpoint: str = Field(
        default="wss://dashscope.aliyuncs.com/api-ws/v1/inference",
        validation_alias=AliasChoices(
            "ALIYUN_ASR_ENDPOINT",
            "DASHSCOPE_WEBSOCKET_ENDPOINT",
            "PROACTIVE_ALIYUN_ASR_ENDPOINT",
        ),
    )
    model: str = Field(
        default="paraformer-realtime-v2",
        validation_alias=AliasChoices(
            "ALIYUN_ASR_MODEL",
            "DASHSCOPE_ASR_MODEL",
            "PROACTIVE_ALIYUN_ASR_MODEL",
        ),
    )
    speech_language: str = Field(
        default="zh-CN",
        validation_alias=AliasChoices(
            "ALIYUN_ASR_LANGUAGE",
            "DASHSCOPE_ASR_LANGUAGE",
            "PROACTIVE_ALIYUN_ASR_LANGUAGE",
        ),
    )
    sample_rate: int = Field(
        default=16000,
        validation_alias=AliasChoices(
            "ALIYUN_ASR_SAMPLE_RATE",
            "DASHSCOPE_ASR_SAMPLE_RATE",
            "PROACTIVE_ALIYUN_ASR_SAMPLE_RATE",
        ),
    )
    audio_format: str = Field(
        default="pcm",
        validation_alias=AliasChoices(
            "ALIYUN_ASR_AUDIO_FORMAT",
            "DASHSCOPE_ASR_AUDIO_FORMAT",
            "PROACTIVE_ALIYUN_ASR_AUDIO_FORMAT",
        ),
    )

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None


class VolcengineSpeechSettings(BaseSettings):
    """豆包大模型流式语音识别(火山引擎 v3 sauc bigmodel)。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("VOLC_ASR_APP_KEY", "VOLC_ASR_APP_ID", "DOUBAO_ASR_APP_KEY"),
    )
    access_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("VOLC_ASR_ACCESS_KEY", "VOLC_ASR_API_KEY", "DOUBAO_ASR_API_KEY"),
    )
    resource_id: str = Field(
        default="volc.bigasr.sauc.duration",
        validation_alias=AliasChoices("VOLC_ASR_RESOURCE_ID", "DOUBAO_ASR_RESOURCE_ID"),
    )
    endpoint: str = Field(
        default="wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
        validation_alias=AliasChoices("VOLC_ASR_ENDPOINT", "DOUBAO_ASR_ENDPOINT"),
    )
    speech_language: str = Field(
        default="zh-CN",
        validation_alias=AliasChoices("VOLC_ASR_LANGUAGE", "DOUBAO_ASR_LANGUAGE"),
    )
    sample_rate: int = Field(
        default=16000,
        validation_alias=AliasChoices("VOLC_ASR_SAMPLE_RATE", "DOUBAO_ASR_SAMPLE_RATE"),
    )

    @property
    def is_configured(self) -> bool:
        return self.app_key is not None and self.access_key is not None
