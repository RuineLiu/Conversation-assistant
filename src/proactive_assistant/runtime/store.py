from proactive_assistant.runtime.contracts import (
    MemoryCandidate,
    PromptDecisionRecord,
    RewardObservation,
    RuntimeFeedbackEvent,
)


class RuntimeStoreError(RuntimeError):
    """Base error for runtime store operations."""


class DecisionRecordNotFoundError(RuntimeStoreError):
    """Raised when feedback or reward references an unknown decision."""


class DecisionRecordAlreadyExistsError(RuntimeStoreError):
    """Raised when a decision id is inserted twice."""


class RuntimeFeedbackEventAlreadyExistsError(RuntimeStoreError):
    """Raised when a feedback event id is inserted twice."""


class InMemoryRuntimeStore:
    """Append-oriented in-memory ledger for early product wiring and tests."""

    def __init__(self) -> None:
        self._decisions: dict[str, PromptDecisionRecord] = {}
        self._decision_order: list[str] = []
        self._feedback_events: dict[str, RuntimeFeedbackEvent] = {}
        self._feedback_order: list[str] = []
        self._reward_observations: dict[str, RewardObservation] = {}
        self._reward_order: list[str] = []
        self._memory_candidates: dict[str, MemoryCandidate] = {}
        self._memory_order: list[str] = []

    def add_decision(self, decision: PromptDecisionRecord) -> PromptDecisionRecord:
        if decision.decision_id in self._decisions:
            raise DecisionRecordAlreadyExistsError(f"decision already exists: {decision.decision_id}")
        self._decisions[decision.decision_id] = decision
        self._decision_order.append(decision.decision_id)
        return decision

    def get_decision(self, decision_id: str) -> PromptDecisionRecord:
        try:
            return self._decisions[decision_id]
        except KeyError as exc:
            raise DecisionRecordNotFoundError(f"unknown decision: {decision_id}") from exc

    def list_decisions(self, *, session_id: str | None = None) -> list[PromptDecisionRecord]:
        decisions = [self._decisions[decision_id] for decision_id in self._decision_order]
        if session_id is None:
            return decisions
        return [decision for decision in decisions if decision.session_id == session_id]

    def add_feedback_event(self, event: RuntimeFeedbackEvent) -> RuntimeFeedbackEvent:
        decision = self.get_decision(event.decision_id)
        if event.session_id != decision.session_id:
            raise RuntimeStoreError("feedback event session_id must match decision session_id")
        if event.event_id in self._feedback_events:
            raise RuntimeFeedbackEventAlreadyExistsError(f"feedback event already exists: {event.event_id}")
        self._feedback_events[event.event_id] = event
        self._feedback_order.append(event.event_id)
        return event

    def list_feedback_events(
        self,
        *,
        decision_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RuntimeFeedbackEvent]:
        events = [self._feedback_events[event_id] for event_id in self._feedback_order]
        if decision_id is not None:
            events = [event for event in events if event.decision_id == decision_id]
        if session_id is not None:
            events = [event for event in events if event.session_id == session_id]
        return events

    def add_reward_observation(self, observation: RewardObservation) -> RewardObservation:
        decision = self.get_decision(observation.decision_id)
        if observation.session_id != decision.session_id:
            raise RuntimeStoreError("reward observation session_id must match decision session_id")
        self._reward_observations[observation.observation_id] = observation
        self._reward_order.append(observation.observation_id)
        return observation

    def list_reward_observations(
        self,
        *,
        decision_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RewardObservation]:
        observations = [self._reward_observations[observation_id] for observation_id in self._reward_order]
        if decision_id is not None:
            observations = [observation for observation in observations if observation.decision_id == decision_id]
        if session_id is not None:
            observations = [observation for observation in observations if observation.session_id == session_id]
        return observations

    def add_memory_candidate(self, candidate: MemoryCandidate) -> MemoryCandidate:
        decision = self.get_decision(candidate.decision_id)
        if candidate.session_id != decision.session_id:
            raise RuntimeStoreError("memory candidate session_id must match decision session_id")
        self._memory_candidates[candidate.memory_candidate_id] = candidate
        self._memory_order.append(candidate.memory_candidate_id)
        return candidate

    def list_memory_candidates(
        self,
        *,
        decision_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryCandidate]:
        candidates = [self._memory_candidates[candidate_id] for candidate_id in self._memory_order]
        if decision_id is not None:
            candidates = [candidate for candidate in candidates if candidate.decision_id == decision_id]
        if session_id is not None:
            candidates = [candidate for candidate in candidates if candidate.session_id == session_id]
        return candidates
