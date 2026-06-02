from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from proactive_assistant.schemas import (
    ActivityPhase,
    ActivityPhaseSpec,
    CandidateIntervention,
    EpisodeTrajectory,
    FeedbackEvent,
    FeedbackType,
    InterventionActionType,
    Persona,
    PolicyDecision,
    Scenario,
    TranscriptSegment,
)
from proactive_assistant.schemas.persona import FieldSource, PreferenceScore


def score(value: float) -> PreferenceScore:
    return PreferenceScore(
        value=value,
        source=FieldSource.MANUAL,
        confidence=0.9,
    )


def sample_persona() -> Persona:
    return Persona(
        source_persona_id="persona_0001",
        internal_persona_id="P0001",
        demographic_stub={
            "age_band": "25-34",
            "occupation_type": "knowledge_worker",
            "daily_routine_density": "high",
        },
        big_five={
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.3,
            "agreeableness": 0.6,
            "neuroticism": 0.4,
        },
        proactive_preferences={
            "pre_activity_tolerance": score(0.8),
            "in_activity_tolerance": score(0.2),
            "post_activity_tolerance": score(0.6),
            "detail_preference": score(0.4),
            "permission_first": score(0.7),
            "privacy_sensitivity": score(0.8),
            "notification_budget_per_hour": 2,
        },
        context_sensitivity={
            "meeting_interrupt_cost": score(0.9),
            "solo_task_interrupt_cost": score(0.3),
            "social_interrupt_cost": score(0.8),
            "deadline_help_value": score(0.7),
        },
        feedback_style={
            "explicit_feedback_rate": score(0.4),
            "negative_feedback_threshold": score(0.6),
            "politeness_bias": score(0.7),
        },
    )


def test_persona_round_trips_to_json() -> None:
    persona = sample_persona()

    restored = Persona.model_validate_json(persona.model_dump_json())

    assert restored.internal_persona_id == "P0001"
    assert restored.proactive_preferences.notification_budget_per_hour == 2


def test_scenario_requires_observable_transcript_for_active_phase() -> None:
    with pytest.raises(ValidationError):
        ActivityPhaseSpec(
            phase=ActivityPhase.PRE_ACTIVITY,
            duration_min=5,
            observable_transcript=[],
        )


def test_transcript_segment_requires_increasing_time() -> None:
    with pytest.raises(ValidationError):
        TranscriptSegment(
            segment_id="seg_001",
            session_id="sess_001",
            speaker="user",
            start_ms=1000,
            end_ms=1000,
            text="准备开会。",
            asr_confidence=0.9,
        )


def test_candidate_intervention_requires_display_text_for_action() -> None:
    with pytest.raises(ValidationError):
        CandidateIntervention(
            action_type=InterventionActionType.IN_TASK_HINT,
            phase=ActivityPhase.IN_ACTIVITY,
            content_level=1,
            confidence=0.8,
            estimated_interrupt_cost=0.2,
            estimated_help_value=0.7,
        )


def test_episode_trajectory_validates_feedback_references() -> None:
    scenario = Scenario(
        scenario_id="meeting_project_sync_001",
        category="meeting",
        hidden_user_goal="确认责任人与截止时间。",
        activity_phases=[
            ActivityPhaseSpec(
                phase=ActivityPhase.PRE_ACTIVITY,
                duration_min=5,
                observable_transcript=["十分钟后开项目会。"],
                latent_risks=["agenda_missing"],
            )
        ],
        success_criteria=["确认 owner", "确认 deadline"],
    )
    transcript = TranscriptSegment(
        segment_id="seg_001",
        session_id="sess_001",
        speaker="user",
        start_ms=0,
        end_ms=2500,
        text="十分钟后开项目会。",
        asr_confidence=0.92,
    )
    decision = PolicyDecision(
        decision_id="dec_001",
        context_hash="ctx_001",
        available_actions=[
            InterventionActionType.NO_ACTION,
            InterventionActionType.PRE_CHECKLIST,
        ],
        chosen_action=InterventionActionType.PRE_CHECKLIST,
        action_probability=0.42,
        content_level=2,
        display_text="会前确认：目标、owner、deadline。",
        policy_version="timing_cb_v0",
    )
    feedback = FeedbackEvent(
        decision_id="dec_001",
        feedback_type=FeedbackType.IMPLICIT_ACCEPT,
        accept=1.0,
        helpfulness=0.8,
        timing_fit=0.9,
        content_fit=0.7,
        reward=3.64,
    )

    episode = EpisodeTrajectory(
        episode_id="episode_001",
        persona_id=sample_persona().internal_persona_id,
        scenario_id=scenario.scenario_id,
        transcript_segments=[transcript],
        decisions=[decision],
        feedback_events=[feedback],
        completed=True,
        total_reward=3.64,
    )

    assert episode.feedback_events[0].decision_id == "dec_001"


def test_episode_rejects_dangling_feedback() -> None:
    with pytest.raises(ValidationError):
        EpisodeTrajectory(
            episode_id="episode_001",
            persona_id="P0001",
            scenario_id="scenario_001",
            feedback_events=[
                FeedbackEvent(
                    decision_id="missing_decision",
                    feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
                    annoyance=1.0,
                    reward=-2.5,
                )
            ],
        )


def test_reward_weight_config_loads() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "reward_weights.yaml"

    weights = yaml.safe_load(config_path.read_text())

    assert weights["accept"] > 0
    assert weights["annoyance"] < 0
    assert "latency_penalty" in weights
