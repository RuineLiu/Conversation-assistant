"""Realtime prompt opportunity detection for transcript windows."""

from proactive_assistant.detection.contracts import (
    CandidateTimingAction,
    DetectionRuleMatch,
    PromptOpportunity,
    PromptOpportunityResult,
    PromptPriority,
)
from proactive_assistant.detection.meeting_state_adapter import opportunities_from_meeting_gaps
from proactive_assistant.detection.service import PromptOpportunityDetector

__all__ = [
    "CandidateTimingAction",
    "DetectionRuleMatch",
    "PromptOpportunity",
    "PromptOpportunityDetector",
    "PromptOpportunityResult",
    "PromptPriority",
    "opportunities_from_meeting_gaps",
]
