"""Model gateway abstractions for proactive assistant model calls."""

from proactive_assistant.model_gateway.clients import (
    FakeModelClient,
    ModelClient,
    ModelGatewayError,
    ModelOutputValidationError,
    ModelRequest,
    ModelResponse,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from proactive_assistant.model_gateway.embeddings import (
    EmbeddingClient,
    EmbeddingResponse,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
)
from proactive_assistant.model_gateway.settings import ModelGatewaySettings

__all__ = [
    "EmbeddingClient",
    "EmbeddingResponse",
    "FakeModelClient",
    "FakeEmbeddingClient",
    "ModelClient",
    "ModelGatewayError",
    "ModelGatewaySettings",
    "ModelOutputValidationError",
    "ModelRequest",
    "ModelResponse",
    "OpenAIChatCompletionsClient",
    "OpenAIEmbeddingClient",
    "OpenAIResponsesClient",
]
