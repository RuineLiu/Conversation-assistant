"""Product-facing flow service for proactive assistant sessions."""

from proactive_assistant.product.contracts import (
    ProductFeedbackResult,
    ProductPromptPayload,
    ProductTranscriptStepResult,
)
from proactive_assistant.product.api import create_app, create_default_product_service
from proactive_assistant.product.service import ProductAssistantService, prompt_payload_from_decision

__all__ = [
    "ProductAssistantService",
    "ProductFeedbackResult",
    "ProductPromptPayload",
    "ProductTranscriptStepResult",
    "create_app",
    "create_default_product_service",
    "prompt_payload_from_decision",
]
