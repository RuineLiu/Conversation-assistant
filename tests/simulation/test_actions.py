import pytest
from pydantic import ValidationError

from proactive_assistant.simulation import (
    AssistantFeedback,
    MoveToAction,
    parse_simulation_action,
)


def test_parse_simulation_action_uses_discriminator() -> None:
    action = parse_simulation_action(
        {
            "type": "feedback_to_assistant",
            "agent_id": "agent_alex",
            "intervention_id": "int_001",
            "feedback": "accept",
            "task_disruption": 0.0,
        }
    )

    assert action.feedback == AssistantFeedback.ACCEPT


def test_move_action_rejects_negative_tile() -> None:
    with pytest.raises(ValidationError):
        MoveToAction(agent_id="agent_alex", target_tile=(-1, 3))
