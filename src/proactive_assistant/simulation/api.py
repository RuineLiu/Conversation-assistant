from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.simulation.actions import (
    AssistantInterventionAction,
    SimulationAction,
)
from proactive_assistant.simulation.engine import SimulationEngine
from proactive_assistant.simulation.events import EventType, SimulationEvent
from proactive_assistant.simulation.state import WorldState, create_default_world_state


class CreateSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str | None = None


class SimulationCreatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str
    state: WorldState


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticks: int = Field(default=1, ge=1, le=100)
    actions: list[SimulationAction] = Field(default_factory=list)


class StepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: WorldState
    events: list[SimulationEvent]


class InstructionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    target_agent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstructionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: WorldState
    event: SimulationEvent


class LogsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str
    events: list[SimulationEvent]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str


class SimulationRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, SimulationEngine] = {}

    def create(self, simulation_id: str | None = None) -> SimulationEngine:
        resolved_id = simulation_id or f"sim_{uuid4().hex[:12]}"
        if resolved_id in self._engines:
            raise ValueError(f"simulation already exists: {resolved_id}")
        engine = SimulationEngine(create_default_world_state(simulation_id=resolved_id))
        engine.state.metadata.setdefault("pending_instructions", [])
        self._engines[resolved_id] = engine
        return engine

    def get(self, simulation_id: str) -> SimulationEngine:
        try:
            return self._engines[simulation_id]
        except KeyError as exc:
            raise KeyError(f"simulation not found: {simulation_id}") from exc


def create_app() -> FastAPI:
    registry = SimulationRegistry()
    app = FastAPI(
        title="Proactive Assistant Simulation API",
        version="0.1.0",
    )
    app.state.simulation_registry = registry

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="simulation-api")

    @app.post("/simulations", response_model=SimulationCreatedResponse, status_code=201)
    def create_simulation(request: CreateSimulationRequest | None = None) -> SimulationCreatedResponse:
        try:
            engine = registry.create(request.simulation_id if request else None)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SimulationCreatedResponse(
            simulation_id=engine.state.simulation_id,
            state=engine.state,
        )

    @app.get("/simulations/{simulation_id}/state", response_model=WorldState)
    def get_state(simulation_id: str) -> WorldState:
        return _get_engine_or_404(registry, simulation_id).state

    @app.post("/simulations/{simulation_id}/step", response_model=StepResponse)
    def step_simulation(simulation_id: str, request: StepRequest | None = None) -> StepResponse:
        engine = _get_engine_or_404(registry, simulation_id)
        resolved_request = request or StepRequest()
        all_events: list[SimulationEvent] = []
        for tick_index in range(resolved_request.ticks):
            tick_actions = resolved_request.actions if tick_index == 0 else []
            result = engine.step(tick_actions)
            all_events.extend(result.events)
        return StepResponse(state=engine.state, events=all_events)

    @app.post("/simulations/{simulation_id}/instructions", response_model=InstructionResponse)
    def send_instruction(
        simulation_id: str,
        request: InstructionRequest,
    ) -> InstructionResponse:
        engine = _get_engine_or_404(registry, simulation_id)
        state = engine.state.model_copy(deep=True)
        pending = list(state.metadata.get("pending_instructions", []))
        instruction_payload = request.model_dump(mode="json")
        pending.append(instruction_payload)
        state.metadata["pending_instructions"] = pending
        event = SimulationEvent(
            event_id=f"instruction_{state.tick:04d}_{len(state.event_log) + 1:03d}",
            tick=state.tick,
            event_type=EventType.USER_INSTRUCTION,
            actor_id="user",
            target_id=request.target_agent_id,
            payload=instruction_payload,
        )
        state.recent_events = [event]
        state.event_log.append(event)
        engine.state = state
        return InstructionResponse(state=state, event=event)

    @app.post(
        "/simulations/{simulation_id}/assistant/interventions",
        response_model=StepResponse,
    )
    def inject_assistant_intervention(
        simulation_id: str,
        request: AssistantInterventionAction,
    ) -> StepResponse:
        engine = _get_engine_or_404(registry, simulation_id)
        result = engine.step([request])
        return StepResponse(state=result.state, events=result.events)

    @app.get("/simulations/{simulation_id}/logs", response_model=LogsResponse)
    def get_logs(
        simulation_id: str,
        limit: int | None = Query(default=None, ge=1, le=1000),
    ) -> LogsResponse:
        engine = _get_engine_or_404(registry, simulation_id)
        events = engine.state.event_log[-limit:] if limit is not None else engine.state.event_log
        return LogsResponse(simulation_id=simulation_id, events=events)

    return app


def _get_engine_or_404(registry: SimulationRegistry, simulation_id: str) -> SimulationEngine:
    try:
        return registry.get(simulation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


app = create_app()
