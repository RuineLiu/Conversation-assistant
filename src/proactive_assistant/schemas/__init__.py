"""Validated data contracts for proactive assistant simulations."""

from proactive_assistant.schemas.episode import EpisodeTrajectory
from proactive_assistant.schemas.enrichment import (
    EnrichmentEvidence,
    PersonaEnrichmentResult,
    PersonaFieldEnrichment,
)
from proactive_assistant.schemas.feedback import FeedbackEvent, FeedbackType
from proactive_assistant.schemas.intervention import (
    CandidateIntervention,
    InterventionActionType,
    OutputModality,
    PolicyDecision,
)
from proactive_assistant.schemas.llm_enrichment import (
    LLMEnrichmentField,
    LLMEnrichmentProposal,
    LLMEnrichmentResponse,
)
from proactive_assistant.schemas.persona import Persona
from proactive_assistant.schemas.scenario import ActivityPhase, ActivityPhaseSpec, Scenario
from proactive_assistant.schemas.transcript import TranscriptSegment

__all__ = [
    "ActivityPhase",
    "ActivityPhaseSpec",
    "CandidateIntervention",
    "EnrichmentEvidence",
    "EpisodeTrajectory",
    "FeedbackEvent",
    "FeedbackType",
    "InterventionActionType",
    "LLMEnrichmentField",
    "LLMEnrichmentProposal",
    "LLMEnrichmentResponse",
    "OutputModality",
    "Persona",
    "PersonaEnrichmentResult",
    "PersonaFieldEnrichment",
    "PolicyDecision",
    "Scenario",
    "TranscriptSegment",
]
