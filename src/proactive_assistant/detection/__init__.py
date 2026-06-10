"""Realtime prompt opportunity detection for transcript windows.

Note: ``PersonalVocabularyService`` lives in ``proactive_assistant.detection.vocabulary``
and is intentionally not re-exported here because it depends on the memory
package, which is part of a pre-existing import cycle through
``runtime.contracts → orchestration.service → detection``. Import it
directly when needed::

    from proactive_assistant.detection.vocabulary import PersonalVocabularyService
"""

from proactive_assistant.detection.contracts import (
    CandidateTimingAction,
    DetectionRuleMatch,
    PromptOpportunity,
    PromptOpportunityResult,
    PromptPriority,
)
from proactive_assistant.detection.meeting_state_adapter import opportunities_from_meeting_gaps
from proactive_assistant.detection.opportunity_detector import (
    OPPORTUNITY_SYSTEM_INSTRUCTIONS,
    OpportunityCandidate,
    OpportunityCategory,
    OpportunityDetectionRequest,
    OpportunityDetectionResult,
    OpportunityDetector,
    OpportunityGapType,
    OpportunityModelCandidate,
    OpportunityModelOutput,
    OpportunityPriority,
)
from proactive_assistant.detection.service import PromptOpportunityDetector
from proactive_assistant.detection.unknown_term_detector import (
    UNKNOWN_TERM_SYSTEM_INSTRUCTIONS,
    UnknownTermCandidate,
    UnknownTermDetectionRequest,
    UnknownTermDetectionResult,
    UnknownTermDetector,
    UnknownTermModelCandidate,
    UnknownTermModelOutput,
    UnknownTermType,
)

__all__ = [
    "CandidateTimingAction",
    "DetectionRuleMatch",
    "OPPORTUNITY_SYSTEM_INSTRUCTIONS",
    "OpportunityCandidate",
    "OpportunityCategory",
    "OpportunityDetectionRequest",
    "OpportunityDetectionResult",
    "OpportunityDetector",
    "OpportunityGapType",
    "OpportunityModelCandidate",
    "OpportunityModelOutput",
    "OpportunityPriority",
    "PromptOpportunity",
    "PromptOpportunityDetector",
    "PromptOpportunityResult",
    "PromptPriority",
    "UNKNOWN_TERM_SYSTEM_INSTRUCTIONS",
    "UnknownTermCandidate",
    "UnknownTermDetectionRequest",
    "UnknownTermDetectionResult",
    "UnknownTermDetector",
    "UnknownTermModelCandidate",
    "UnknownTermModelOutput",
    "UnknownTermType",
    "opportunities_from_meeting_gaps",
]
