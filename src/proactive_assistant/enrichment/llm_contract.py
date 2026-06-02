from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from proactive_assistant.schemas.llm_enrichment import (
    LLM_ENRICHMENT_SCHEMA_VERSION,
    LLMEnrichmentResponse,
)


def parse_llm_enrichment_response(payload: Mapping[str, Any]) -> LLMEnrichmentResponse:
    """Validate untrusted LLM JSON output against the enrichment contract."""

    response = LLMEnrichmentResponse.model_validate(payload)
    if response.schema_version != LLM_ENRICHMENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported LLM enrichment schema_version: "
            f"{response.schema_version}"
        )
    return response
