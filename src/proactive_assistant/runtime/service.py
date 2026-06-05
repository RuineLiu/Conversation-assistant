from datetime import UTC, datetime
from hashlib import sha1
from typing import Any
from uuid import uuid4

from proactive_assistant.orchestration import PromptCandidate, PromptOrchestrationResult
from proactive_assistant.prompting import ContentGranularity, PrivacyLevel
from proactive_assistant.runtime.contracts import (
    FeedbackPolarity,
    FeedbackSignalSource,
    FeedbackSignalType,
    MemoryCandidate,
    MemoryCandidateType,
    MemoryWritePolicy,
    PromptDecisionDisplayStatus,
    PromptDecisionRecord,
    RewardComponents,
    RewardObservation,
    RuntimeFeedbackEvent,
    display_status_for_candidate,
)
from proactive_assistant.runtime.store import InMemoryRuntimeStore, RuntimeRepository


DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "accept": 1.0,
    "helpfulness": 0.8,
    "task_progress": 0.7,
    "timing_fit": 0.5,
    "content_fit": 0.5,
    "annoyance": -1.0,
    "flow_break": -0.8,
    "redundancy": -0.7,
    "privacy_risk": -1.2,
    "latency_penalty": -0.3,
}


class PromptRuntimeService:
    """Runtime ledger for prompt decisions, feedback, reward, and memory candidates."""

    def __init__(
        self,
        store: RuntimeRepository | None = None,
        *,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        self.store = store or InMemoryRuntimeStore()
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS | (reward_weights or {}))

    def log_orchestration_result(
        self,
        result: PromptOrchestrationResult,
        *,
        policy_version: str = "prompt_orchestration_v0",
    ) -> list[PromptDecisionRecord]:
        return [self.log_candidate(candidate, policy_version=policy_version) for candidate in result.candidates]

    def log_candidate(
        self,
        candidate: PromptCandidate,
        *,
        decision_id: str | None = None,
        policy_version: str = "prompt_orchestration_v0",
        shown_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromptDecisionRecord:
        display_status = display_status_for_candidate(candidate)
        now = datetime.now(UTC)
        prompt_result = candidate.prompt_result
        decision = PromptDecisionRecord(
            decision_id=decision_id or _decision_id(candidate),
            session_id=candidate.session_id,
            candidate_id=candidate.candidate_id,
            opportunity_id=candidate.opportunity.opportunity_id,
            candidate=candidate,
            display_status=display_status,
            prompt_category=_prompt_category(candidate),
            content_granularity=_actual_content_granularity(candidate),
            prd_surface=candidate.prompt_request.prd_surface,
            display_mode=candidate.prompt_request.display_mode,
            duration_policy=candidate.prompt_request.duration_policy,
            privacy_level=_privacy_level(candidate),
            privacy_risk=_privacy_risk(candidate),
            policy_version=policy_version,
            created_at=now,
            shown_at=shown_at or (now if display_status == PromptDecisionDisplayStatus.SHOWN else None),
            metadata=metadata or {},
        )
        if prompt_result is not None and prompt_result.model_usage is not None:
            decision = decision.model_copy(
                update={
                    "metadata": {
                        **decision.metadata,
                        "model_usage": prompt_result.model_usage.model_dump(mode="json"),
                    }
                }
            )
        return self.store.add_decision(decision)

    def record_feedback(
        self,
        decision_id: str,
        signal_type: FeedbackSignalType | str,
        *,
        event_id: str | None = None,
        source: FeedbackSignalSource | str | None = None,
        polarity: FeedbackPolarity | str | None = None,
        intensity: float | None = None,
        text: str = "",
        dwell_ms: int | None = None,
        latency_ms: int | None = None,
        helpfulness: float | None = None,
        timing_fit: float | None = None,
        content_fit: float | None = None,
        task_progress_delta: float | None = None,
        flow_break_score: float | None = None,
        redundancy_score: float | None = None,
        privacy_risk_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeFeedbackEvent:
        decision = self.store.get_decision(decision_id)
        resolved_signal = FeedbackSignalType(signal_type)
        resolved_intensity = intensity if intensity is not None else _default_intensity(resolved_signal, dwell_ms, latency_ms)
        resolved_source = FeedbackSignalSource(source) if source is not None else _default_source(resolved_signal)
        resolved_polarity = FeedbackPolarity(polarity) if polarity is not None else _default_polarity(resolved_signal)
        defaults = _default_metrics(resolved_signal, resolved_intensity, dwell_ms, latency_ms)
        event = RuntimeFeedbackEvent(
            event_id=event_id or f"fb_{uuid4().hex[:12]}",
            decision_id=decision_id,
            session_id=decision.session_id,
            signal_type=resolved_signal,
            source=resolved_source,
            polarity=resolved_polarity,
            intensity=resolved_intensity,
            text=text,
            dwell_ms=dwell_ms,
            latency_ms=latency_ms,
            helpfulness=helpfulness if helpfulness is not None else defaults["helpfulness"],
            timing_fit=timing_fit if timing_fit is not None else defaults["timing_fit"],
            content_fit=content_fit if content_fit is not None else defaults["content_fit"],
            task_progress_delta=(
                task_progress_delta if task_progress_delta is not None else defaults["task_progress_delta"]
            ),
            flow_break_score=flow_break_score if flow_break_score is not None else defaults["flow_break_score"],
            redundancy_score=redundancy_score if redundancy_score is not None else defaults["redundancy_score"],
            privacy_risk_score=privacy_risk_score if privacy_risk_score is not None else defaults["privacy_risk_score"],
            metadata=metadata or {},
        )
        return self.store.add_feedback_event(event)

    def compute_reward_observation(self, decision_id: str) -> RewardObservation:
        decision = self.store.get_decision(decision_id)
        events = self.store.list_feedback_events(decision_id=decision_id)
        components = _aggregate_reward_components(decision, events)
        explicit_score = _direct_score(events, FeedbackSignalSource.EXPLICIT)
        implicit_score = _direct_score(events, FeedbackSignalSource.IMPLICIT)
        task_outcome_score = _direct_score(events, FeedbackSignalSource.TASK_OUTCOME)
        interruption_cost = _round(
            components.annoyance
            + abs(self.reward_weights["flow_break"]) * components.flow_break
            + abs(self.reward_weights["redundancy"]) * components.redundancy
        )
        privacy_penalty = _round(abs(self.reward_weights["privacy_risk"]) * components.privacy_risk)
        latency_penalty = _round(abs(self.reward_weights["latency_penalty"]) * components.latency_penalty)
        final_reward = _round(sum(self.reward_weights[name] * getattr(components, name) for name in self.reward_weights))
        observation = RewardObservation(
            observation_id=f"rew_{uuid4().hex[:12]}",
            decision_id=decision.decision_id,
            session_id=decision.session_id,
            event_ids=[event.event_id for event in events],
            components=components,
            explicit_score=explicit_score,
            implicit_score=implicit_score,
            task_outcome_score=task_outcome_score,
            interruption_cost=interruption_cost,
            privacy_penalty=privacy_penalty,
            latency_penalty=latency_penalty,
            final_reward=final_reward,
            confidence=_reward_confidence(events),
            metadata={"reward_formula": "prompt_runtime_v0", "weights": self.reward_weights},
        )
        return self.store.add_reward_observation(observation)

    def propose_memory_candidates(
        self,
        decision_id: str,
        *,
        persist: bool = True,
    ) -> list[MemoryCandidate]:
        decision = self.store.get_decision(decision_id)
        events = self.store.list_feedback_events(decision_id=decision_id)
        candidates: list[MemoryCandidate] = []
        for index, event in enumerate(events):
            candidate = _memory_candidate_for_event(decision, event, index)
            if candidate is not None:
                candidates.append(candidate)
        if persist:
            return [self.store.add_memory_candidate(candidate) for candidate in candidates]
        return candidates


def _decision_id(candidate: PromptCandidate) -> str:
    digest = candidate.candidate_id.removeprefix("cand_")
    return f"dec_{digest}"


def _prompt_category(candidate: PromptCandidate) -> str | None:
    if candidate.prompt_result is not None and candidate.prompt_result.prompt_category is not None:
        return str(_enum_value(candidate.prompt_result.prompt_category))
    if candidate.prompt_request.prompt_category_candidate is not None:
        return str(_enum_value(candidate.prompt_request.prompt_category_candidate))
    return None


def _actual_content_granularity(candidate: PromptCandidate) -> ContentGranularity:
    if candidate.prompt_result is None:
        return ContentGranularity.NO_ACTION
    return ContentGranularity(int(candidate.prompt_result.content_granularity))


def _privacy_level(candidate: PromptCandidate) -> PrivacyLevel:
    if candidate.prompt_result is not None:
        return PrivacyLevel(str(_enum_value(candidate.prompt_result.privacy_level)))
    return PrivacyLevel(str(_enum_value(candidate.opportunity.privacy_level)))


def _privacy_risk(candidate: PromptCandidate) -> float:
    risks = [candidate.opportunity.privacy_risk]
    if candidate.prompt_result is not None:
        risks.append(candidate.prompt_result.privacy_risk)
    return max(risks)


def _default_source(signal: FeedbackSignalType) -> FeedbackSignalSource:
    if signal in {
        FeedbackSignalType.IGNORE,
        FeedbackSignalType.SECOND_LOOK,
        FeedbackSignalType.DWELL,
        FeedbackSignalType.FLOW_BREAK,
        FeedbackSignalType.REDUNDANT_TRIGGER,
    }:
        return FeedbackSignalSource.IMPLICIT
    if signal in {FeedbackSignalType.TASK_PROGRESS, FeedbackSignalType.TASK_REGRESSION}:
        return FeedbackSignalSource.TASK_OUTCOME
    if signal == FeedbackSignalType.LATENCY_OBSERVED:
        return FeedbackSignalSource.SYSTEM
    return FeedbackSignalSource.EXPLICIT


def _default_polarity(signal: FeedbackSignalType) -> FeedbackPolarity:
    if signal in {
        FeedbackSignalType.ACCEPT,
        FeedbackSignalType.ASK_FOLLOWUP,
        FeedbackSignalType.OPEN_DETAIL,
        FeedbackSignalType.SECOND_LOOK,
        FeedbackSignalType.DWELL,
        FeedbackSignalType.SAVE,
        FeedbackSignalType.MARK_HELPFUL,
        FeedbackSignalType.VERBAL_POSITIVE,
        FeedbackSignalType.TASK_PROGRESS,
    }:
        return FeedbackPolarity.POSITIVE
    if signal == FeedbackSignalType.IGNORE:
        return FeedbackPolarity.NEUTRAL
    return FeedbackPolarity.NEGATIVE


def _default_intensity(signal: FeedbackSignalType, dwell_ms: int | None, latency_ms: int | None) -> float:
    if signal == FeedbackSignalType.IGNORE:
        return 0.2
    if signal == FeedbackSignalType.SNOOZE:
        return 0.5
    if signal == FeedbackSignalType.DISMISS:
        return 0.8
    if signal == FeedbackSignalType.OPEN_DETAIL:
        return 0.7
    if signal == FeedbackSignalType.SECOND_LOOK:
        return 0.6
    if signal == FeedbackSignalType.DWELL:
        return min(1.0, (dwell_ms or 0) / 8000)
    if signal == FeedbackSignalType.LATENCY_OBSERVED:
        return min(1.0, (latency_ms or 0) / 5000)
    if signal in {FeedbackSignalType.SWITCH_TO_MANUAL, FeedbackSignalType.COMPLAIN_INTERRUPTIVE}:
        return 0.9
    if signal == FeedbackSignalType.REDUNDANT_TRIGGER:
        return 0.7
    return 1.0


def _default_metrics(
    signal: FeedbackSignalType,
    intensity: float,
    dwell_ms: int | None,
    latency_ms: int | None,
) -> dict[str, float]:
    metrics = {
        "helpfulness": 0.0,
        "timing_fit": 0.0,
        "content_fit": 0.0,
        "task_progress_delta": 0.0,
        "flow_break_score": 0.0,
        "redundancy_score": 0.0,
        "privacy_risk_score": 0.0,
    }
    if signal in {FeedbackSignalType.ACCEPT, FeedbackSignalType.VERBAL_POSITIVE}:
        metrics.update(helpfulness=0.7 * intensity, timing_fit=0.5, content_fit=0.5)
    elif signal == FeedbackSignalType.ASK_FOLLOWUP:
        metrics.update(helpfulness=0.9 * intensity, timing_fit=0.4, content_fit=0.6, task_progress_delta=0.3)
    elif signal == FeedbackSignalType.OPEN_DETAIL:
        metrics.update(helpfulness=0.6 * intensity, timing_fit=0.3, content_fit=0.3)
    elif signal == FeedbackSignalType.SECOND_LOOK:
        metrics.update(helpfulness=0.5 * intensity, content_fit=0.3)
    elif signal == FeedbackSignalType.DWELL:
        metrics.update(helpfulness=min(0.6, (dwell_ms or 0) / 12000), content_fit=0.2 if dwell_ms else 0.0)
    elif signal in {FeedbackSignalType.SAVE, FeedbackSignalType.MARK_HELPFUL}:
        metrics.update(helpfulness=1.0 * intensity, content_fit=0.8, task_progress_delta=0.4)
    elif signal == FeedbackSignalType.DISMISS:
        metrics.update(timing_fit=-0.4, content_fit=-0.2)
    elif signal == FeedbackSignalType.SNOOZE:
        metrics.update(timing_fit=-0.5)
    elif signal == FeedbackSignalType.MARK_NOT_HELPFUL:
        metrics.update(content_fit=-0.8)
    elif signal == FeedbackSignalType.VERBAL_REJECT:
        metrics.update(timing_fit=-0.7, content_fit=-0.5)
    elif signal == FeedbackSignalType.PRIVACY_REJECT:
        metrics.update(content_fit=-1.0, privacy_risk_score=1.0)
    elif signal in {FeedbackSignalType.DISABLE_AUTO_PROMPT, FeedbackSignalType.SWITCH_TO_MANUAL}:
        metrics.update(timing_fit=-0.8, flow_break_score=0.5)
    elif signal == FeedbackSignalType.COMPLAIN_INTERRUPTIVE:
        metrics.update(timing_fit=-0.9, flow_break_score=0.8)
    elif signal == FeedbackSignalType.FLOW_BREAK:
        metrics.update(timing_fit=-0.8, flow_break_score=intensity)
    elif signal == FeedbackSignalType.REDUNDANT_TRIGGER:
        metrics.update(content_fit=-0.5, redundancy_score=intensity)
    elif signal == FeedbackSignalType.TASK_PROGRESS:
        metrics.update(task_progress_delta=intensity)
    elif signal == FeedbackSignalType.TASK_REGRESSION:
        metrics.update(task_progress_delta=-intensity)
    elif signal == FeedbackSignalType.LATENCY_OBSERVED:
        metrics.update(timing_fit=-min(1.0, (latency_ms or 0) / 5000))
    return metrics


def _aggregate_reward_components(
    decision: PromptDecisionRecord,
    events: list[RuntimeFeedbackEvent],
) -> RewardComponents:
    accept_signals = {
        FeedbackSignalType.ACCEPT,
        FeedbackSignalType.ASK_FOLLOWUP,
        FeedbackSignalType.OPEN_DETAIL,
        FeedbackSignalType.SAVE,
        FeedbackSignalType.MARK_HELPFUL,
        FeedbackSignalType.VERBAL_POSITIVE,
    }
    annoyance_signals = {
        FeedbackSignalType.DISMISS,
        FeedbackSignalType.SNOOZE,
        FeedbackSignalType.MARK_NOT_HELPFUL,
        FeedbackSignalType.VERBAL_REJECT,
        FeedbackSignalType.DISABLE_AUTO_PROMPT,
        FeedbackSignalType.SWITCH_TO_MANUAL,
        FeedbackSignalType.COMPLAIN_INTERRUPTIVE,
    }
    timing_values = [event.timing_fit for event in events if event.timing_fit != 0.0]
    content_values = [event.content_fit for event in events if event.content_fit != 0.0]
    return RewardComponents(
        accept=_max_intensity(events, accept_signals),
        helpfulness=_max_value(event.helpfulness for event in events),
        task_progress=_clip(sum(event.task_progress_delta for event in events), -1.0, 1.0),
        timing_fit=_avg(timing_values),
        content_fit=_avg(content_values),
        annoyance=_max_intensity(events, annoyance_signals),
        flow_break=_max_value(event.flow_break_score for event in events),
        redundancy=_max_value(event.redundancy_score for event in events),
        privacy_risk=max(decision.privacy_risk, _max_value(event.privacy_risk_score for event in events)),
        latency_penalty=_max_value(_latency_penalty(event) for event in events),
    )


def _direct_score(events: list[RuntimeFeedbackEvent], source: FeedbackSignalSource) -> float:
    score = 0.0
    for event in events:
        if event.source != source:
            continue
        if event.polarity == FeedbackPolarity.POSITIVE:
            score += event.intensity
        elif event.polarity == FeedbackPolarity.NEGATIVE:
            score -= event.intensity
    return _round(_clip(score, -3.0, 3.0))


def _reward_confidence(events: list[RuntimeFeedbackEvent]) -> float:
    if not events:
        return 0.15
    sources = {event.source for event in events}
    confidence = 0.2
    if FeedbackSignalSource.EXPLICIT in sources:
        confidence += 0.35
    if FeedbackSignalSource.IMPLICIT in sources:
        confidence += 0.2
    if FeedbackSignalSource.TASK_OUTCOME in sources:
        confidence += 0.25
    if len(events) >= 3:
        confidence += 0.1
    return _round(min(confidence, 1.0))


def _memory_candidate_for_event(
    decision: PromptDecisionRecord,
    event: RuntimeFeedbackEvent,
    index: int,
) -> MemoryCandidate | None:
    signal = FeedbackSignalType(event.signal_type)
    category = decision.prompt_category or "unknown"
    surface = str(_enum_value(decision.prd_surface))
    granularity = int(decision.content_granularity)
    event_ids = [event.event_id]
    if signal == FeedbackSignalType.PRIVACY_REJECT:
        return _memory_candidate(
            decision,
            event_ids,
            index,
            MemoryCandidateType.PRIVACY_PREFERENCE,
            f"Avoid showing detailed automatic prompts for {category} on {surface} when privacy risk is high.",
            confidence=0.95,
            write_policy=MemoryWritePolicy.ELIGIBLE,
            privacy_level=PrivacyLevel.MEDIUM,
            reason="explicit privacy rejection",
        )
    if signal == FeedbackSignalType.DISABLE_AUTO_PROMPT:
        return _memory_candidate(
            decision,
            event_ids,
            index,
            MemoryCandidateType.NEGATIVE_PREFERENCE,
            f"User prefers no automatic prompts for {category} in this meeting context.",
            confidence=0.9,
            write_policy=MemoryWritePolicy.ELIGIBLE,
            reason="explicit opt-out from automatic prompting",
        )
    if signal == FeedbackSignalType.SWITCH_TO_MANUAL:
        return _memory_candidate(
            decision,
            event_ids,
            index,
            MemoryCandidateType.USER_PREFERENCE,
            f"User prefers manual prompt triggering for {category} during meetings.",
            confidence=0.85,
            write_policy=MemoryWritePolicy.ELIGIBLE,
            reason="explicit switch to manual mode",
        )
    if signal in {FeedbackSignalType.COMPLAIN_INTERRUPTIVE, FeedbackSignalType.VERBAL_REJECT, FeedbackSignalType.SNOOZE}:
        return _memory_candidate(
            decision,
            event_ids,
            index,
            MemoryCandidateType.NEGATIVE_PREFERENCE,
            f"User may prefer fewer or later {category} prompts during meetings.",
            confidence=0.6,
            write_policy=MemoryWritePolicy.NEEDS_CONFIRMATION,
            reason="negative interruption feedback",
        )
    if signal in {FeedbackSignalType.ACCEPT, FeedbackSignalType.MARK_HELPFUL, FeedbackSignalType.SAVE}:
        if category == "summary_gap_check" and decision.candidate.prompt_result is not None:
            text = decision.candidate.prompt_result.glasses_text or decision.candidate.prompt_result.app_detail_text
            return _memory_candidate(
                decision,
                event_ids,
                index,
                MemoryCandidateType.ACTION_ITEM,
                text,
                confidence=0.65,
                write_policy=_content_memory_write_policy(decision),
                privacy_level=decision.privacy_level,
                reason="accepted or saved gap-check prompt",
            )
        return _memory_candidate(
            decision,
            event_ids,
            index,
            MemoryCandidateType.USER_PREFERENCE,
            f"User may find granularity-{granularity} {category} prompts useful on {surface}.",
            confidence=0.55,
            write_policy=MemoryWritePolicy.NEEDS_CONFIRMATION,
            reason="positive prompt feedback",
        )
    return None


def _memory_candidate(
    decision: PromptDecisionRecord,
    source_event_ids: list[str],
    index: int,
    candidate_type: MemoryCandidateType,
    text: str,
    *,
    confidence: float,
    write_policy: MemoryWritePolicy,
    privacy_level: PrivacyLevel | None = None,
    reason: str,
) -> MemoryCandidate:
    candidate_id = _memory_candidate_id(decision.decision_id, source_event_ids, str(candidate_type), index)
    prompt_result = decision.candidate.prompt_result
    opportunity = decision.candidate.opportunity
    return MemoryCandidate(
        memory_candidate_id=candidate_id,
        decision_id=decision.decision_id,
        session_id=decision.session_id,
        source_event_ids=source_event_ids,
        candidate_type=candidate_type,
        text=text,
        confidence=confidence,
        write_policy=write_policy,
        privacy_level=privacy_level or decision.privacy_level,
        reason=reason,
        metadata={
            "candidate_id": decision.candidate_id,
            "opportunity_id": decision.opportunity_id,
            "prompt_category": decision.prompt_category,
            "content_granularity": int(decision.content_granularity),
            "prd_surface": _enum_value(decision.prd_surface),
            "display_mode": _enum_value(decision.display_mode),
            "duration_policy": _enum_value(decision.duration_policy),
            "privacy_level": _enum_value(decision.privacy_level),
            "privacy_risk": decision.privacy_risk,
            "source_refs": list(prompt_result.source_refs) if prompt_result is not None else [],
            "source_capture_ref": prompt_result.source_refs[0] if prompt_result is not None and prompt_result.source_refs else "",
            "captured_text": opportunity.captured_text,
            "trigger_segment_ids": list(opportunity.trigger_segment_ids),
            "activity_phase": _enum_value(opportunity.activity_phase),
            "candidate_timing_action": _enum_value(opportunity.candidate_timing_action),
            "opportunity_priority": _enum_value(opportunity.priority),
            "opportunity_reason": opportunity.reason,
            "safety_flags": sorted(set(opportunity.safety_flags + (prompt_result.safety_flags if prompt_result is not None else []))),
        },
    )


def _content_memory_write_policy(decision: PromptDecisionRecord) -> MemoryWritePolicy:
    if decision.privacy_risk >= 0.7 or _enum_value(decision.privacy_level) == PrivacyLevel.HIGH.value:
        return MemoryWritePolicy.BLOCKED
    return MemoryWritePolicy.NEEDS_CONFIRMATION


def _memory_candidate_id(decision_id: str, event_ids: list[str], candidate_type: str, index: int) -> str:
    digest = sha1(f"{decision_id}:{','.join(event_ids)}:{candidate_type}:{index}".encode("utf-8")).hexdigest()[:12]
    return f"memcand_{digest}"


def _max_intensity(events: list[RuntimeFeedbackEvent], signals: set[FeedbackSignalType]) -> float:
    return _max_value(event.intensity for event in events if FeedbackSignalType(event.signal_type) in signals)


def _max_value(values: Any) -> float:
    return max([float(value) for value in values], default=0.0)


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / len(values))


def _latency_penalty(event: RuntimeFeedbackEvent) -> float:
    if event.signal_type != FeedbackSignalType.LATENCY_OBSERVED or event.latency_ms is None:
        return 0.0
    return min(1.0, event.latency_ms / 5000)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float) -> float:
    return round(value, 4)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value
