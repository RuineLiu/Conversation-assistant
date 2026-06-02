from typing import Protocol

from proactive_assistant.schemas.enrichment import PersonaEnrichmentResult
from proactive_assistant.schemas.persona import Persona


class PersonaEnricher(Protocol):
    """Interface for deterministic, LLM, or simulator-based persona enrichment."""

    def enrich(self, persona: Persona) -> PersonaEnrichmentResult:
        """Return an enriched persona and auditable field updates."""
