from proactive_assistant.simulation import (
    AssistantChannel,
    AssistantFeedback,
    AssistantInterventionAction,
    EventType,
    FeedbackToAssistantAction,
    InteractObjectAction,
    MoveToAction,
    ObjectInteraction,
    SimulationEngine,
    SpeakAction,
    TimingPolicy,
    UpdateGoalAction,
    create_default_world_state,
)


def move_until_arrival(engine: SimulationEngine, agent_id: str, target: tuple[int, int]) -> None:
    for _ in range(40):
        if engine.state.agents[agent_id].tile == target:
            return
        engine.step()
    raise AssertionError(f"{agent_id} did not reach {target}")


def test_move_action_sets_path_and_advances_one_tile() -> None:
    engine = SimulationEngine()
    start_tile = engine.state.agents["agent_alex"].tile

    result = engine.step([MoveToAction(agent_id="agent_alex", target_tile=(11, 6))])

    assert result.state.agents["agent_alex"].tile != start_tile
    assert any(event.event_type == EventType.MOVEMENT_PLANNED for event in result.events)
    assert any(event.event_type == EventType.AGENT_MOVED for event in result.events)


def test_move_action_rejects_blocked_target() -> None:
    engine = SimulationEngine()

    result = engine.step([MoveToAction(agent_id="agent_alex", target_tile=(8, 8))])

    assert result.events[-1].event_type == EventType.ACTION_REJECTED
    assert result.events[-1].payload["reason"] == "target is unreachable or non-walkable"


def test_object_interaction_requires_interaction_tile() -> None:
    engine = SimulationEngine()

    result = engine.step(
        [
            InteractObjectAction(
                agent_id="agent_alex",
                object_id="screen_1",
                interaction=ObjectInteraction.TURN_ON,
            )
        ]
    )

    assert result.events[-1].event_type == EventType.ACTION_REJECTED


def test_object_interaction_updates_state_from_valid_tile() -> None:
    state = create_default_world_state()
    state.agents["agent_alex"].tile = (2, 6)
    engine = SimulationEngine(state)

    result = engine.step(
        [
            InteractObjectAction(
                agent_id="agent_alex",
                object_id="screen_1",
                interaction=ObjectInteraction.TURN_ON,
            )
        ]
    )

    assert result.state.objects["screen_1"].state["power"] == "on"
    assert result.events[-1].event_type == EventType.OBJECT_INTERACTION


def test_speak_adds_dialogue_and_transcript_event() -> None:
    engine = SimulationEngine()

    result = engine.step(
        [
            SpeakAction(
                agent_id="agent_bao",
                utterance="Can we confirm the action items?",
                audience=["agent_alex"],
            )
        ]
    )

    assert result.state.active_dialogue[-1].utterance == "Can we confirm the action items?"
    assert result.state.transcript_events[-1].speaker_agent_id == "agent_bao"
    assert [event.event_type for event in result.events] == [
        EventType.DIALOGUE,
        EventType.TRANSCRIPT,
    ]


def test_goal_update_changes_public_goal() -> None:
    engine = SimulationEngine()

    result = engine.step(
        [UpdateGoalAction(agent_id="agent_chris", goal="write action items on whiteboard")]
    )

    assert result.state.agents["agent_chris"].current_goal == "write action items on whiteboard"
    assert result.events[-1].event_type == EventType.GOAL_UPDATED


def test_assistant_intervention_and_feedback_update_state() -> None:
    engine = SimulationEngine()
    intervention = AssistantInterventionAction(
        intervention_id="int_001",
        target_agent_id="agent_alex",
        timing_policy=TimingPolicy.BEFORE_ACTIVITY,
        channel=AssistantChannel.SMART_GLASSES_OVERLAY,
        content="You planned to turn on the screen before the meeting starts.",
        detail_level=0.4,
        context_refs=["object:screen_1"],
    )

    first = engine.step([intervention])

    assert first.state.active_interventions[0].intervention_id == "int_001"
    assert first.state.agents["agent_alex"].assistant_state.last_intervention_tick == 1

    second = engine.step(
        [
            FeedbackToAssistantAction(
                agent_id="agent_alex",
                intervention_id="int_001",
                feedback=AssistantFeedback.ACCEPT,
                reason="useful",
            )
        ]
    )

    assert second.state.active_interventions == []
    assert second.events[-1].event_type == EventType.ASSISTANT_FEEDBACK
