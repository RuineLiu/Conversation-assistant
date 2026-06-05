from fastapi.testclient import TestClient

from proactive_assistant.simulation.api import create_app


def create_client() -> TestClient:
    return TestClient(create_app())


def test_health_endpoint() -> None:
    client = create_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "simulation-api"}


def test_create_and_read_simulation_state() -> None:
    client = create_client()

    created = client.post("/simulations", json={"simulation_id": "sim_api_001"})

    assert created.status_code == 201
    payload = created.json()
    assert payload["simulation_id"] == "sim_api_001"
    assert payload["state"]["tick"] == 0
    assert "agent_alex" in payload["state"]["agents"]

    state = client.get("/simulations/sim_api_001/state")
    assert state.status_code == 200
    assert state.json()["simulation_id"] == "sim_api_001"


def test_create_duplicate_simulation_rejected() -> None:
    client = create_client()

    assert client.post("/simulations", json={"simulation_id": "sim_api_001"}).status_code == 201
    duplicate = client.post("/simulations", json={"simulation_id": "sim_api_001"})

    assert duplicate.status_code == 409


def test_unknown_simulation_returns_404() -> None:
    client = create_client()

    response = client.get("/simulations/missing/state")

    assert response.status_code == 404


def test_step_endpoint_applies_structured_actions() -> None:
    client = create_client()
    client.post("/simulations", json={"simulation_id": "sim_api_001"})

    response = client.post(
        "/simulations/sim_api_001/step",
        json={
            "ticks": 1,
            "actions": [
                {
                    "type": "move_to",
                    "agent_id": "agent_alex",
                    "target_tile": [11, 6],
                    "reason": "approach laptop",
                },
                {
                    "type": "speak",
                    "agent_id": "agent_bao",
                    "utterance": "Can we confirm the action items?",
                    "audience": ["agent_alex"],
                },
            ],
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"]["tick"] == 1
    assert payload["events"][0]["event_type"] == "movement_planned"
    assert any(event["event_type"] == "transcript" for event in payload["events"])


def test_step_endpoint_can_advance_multiple_ticks() -> None:
    client = create_client()
    client.post("/simulations", json={"simulation_id": "sim_api_001"})

    response = client.post(
        "/simulations/sim_api_001/step",
        json={
            "ticks": 3,
            "actions": [
                {
                    "type": "move_to",
                    "agent_id": "agent_alex",
                    "target_tile": [11, 6],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["tick"] == 3
    assert len(payload["events"]) >= 3


def test_instruction_endpoint_records_pending_instruction() -> None:
    client = create_client()
    client.post("/simulations", json={"simulation_id": "sim_api_001"})

    response = client.post(
        "/simulations/sim_api_001/instructions",
        json={
            "instruction": "Ask Alex to start the presentation.",
            "target_agent_id": "agent_alex",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["event"]["event_type"] == "user_instruction"
    assert payload["state"]["metadata"]["pending_instructions"][0]["target_agent_id"] == "agent_alex"


def test_assistant_intervention_endpoint_injects_intervention() -> None:
    client = create_client()
    client.post("/simulations", json={"simulation_id": "sim_api_001"})

    response = client.post(
        "/simulations/sim_api_001/assistant/interventions",
        json={
            "type": "assistant_intervention",
            "intervention_id": "int_api_001",
            "target_agent_id": "agent_alex",
            "timing_policy": "before_activity",
            "channel": "smart_glasses_overlay",
            "content": "You planned to turn on the screen before presenting.",
            "detail_level": 0.4,
            "context_refs": ["object:screen_1"],
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"]["active_interventions"][0]["intervention_id"] == "int_api_001"
    assert payload["events"][0]["event_type"] == "assistant_intervention"


def test_logs_endpoint_returns_limited_events() -> None:
    client = create_client()
    client.post("/simulations", json={"simulation_id": "sim_api_001"})
    client.post(
        "/simulations/sim_api_001/instructions",
        json={"instruction": "Start the meeting.", "target_agent_id": "agent_alex"},
    )
    client.post(
        "/simulations/sim_api_001/step",
        json={
            "actions": [
                {
                    "type": "speak",
                    "agent_id": "agent_alex",
                    "utterance": "Let's begin.",
                }
            ]
        },
    )

    response = client.get("/simulations/sim_api_001/logs?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["simulation_id"] == "sim_api_001"
    assert len(payload["events"]) == 2
