"""Persona enrichment modules."""

from proactive_assistant.enrichment.llm_contract import parse_llm_enrichment_response
from proactive_assistant.enrichment.merge import merge_llm_enrichment
from proactive_assistant.enrichment.persona_enricher import PersonaEnricher
from proactive_assistant.enrichment.rule_based import RuleBasedPersonaEnricher

__all__ = [
    "PersonaEnricher",
    "RuleBasedPersonaEnricher",
    "merge_llm_enrichment",
    "parse_llm_enrichment_response",
]
