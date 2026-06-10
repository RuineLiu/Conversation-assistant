from datetime import UTC, datetime
from hashlib import sha1
from typing import Any
from uuid import uuid4

from proactive_assistant.orchestration import PromptCandidate, PromptOrchestrationResult
from proactive_assistant.prompting import ContentGranularity, PrivacyLevel
from proactive_assistant.runtime.contracts import (
    FeedbackInputChannel,
    FeedbackPolarity,
    FeedbackSignalSource,
    FeedbackSignalType,
    FeedbackTarget,
    MemoryCandidate,
    MemoryCandidateType,
    MemoryWritePolicy,
    PolicyActionSnapshot,
    PolicyActionBreakdown,
    PolicyBaselineAction,
    PolicyBaselineComparisonReport,
    PolicyBaselineDecision,
    PolicyBaselineEvaluation,
    PolicyBaselineMetrics,
    PolicyBaselineName,
    PolicyBaselineRelation,
    PolicyEpisode,
    PolicyEvaluationReport,
    PolicyMetricSummary,
    PolicyNextStateRef,
    PolicyStateRef,
    PolicyStep,
    PolicyTrainingExample,
    PolicyTrainingExport,
    PolicyTrainingLabel,
    ProactiveDisplayStrategy,
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
    "granularity_fit": 0.4,
    "display_fit": 0.4,
    "annoyance": -1.0,
    "flow_break": -0.8,
    "redundancy": -0.7,
    "privacy_risk": -1.2,
    "latency_penalty": -0.3,
    "missed_opportunity": -1.1,
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
        input_channel: FeedbackInputChannel | str | None = None,
        target: FeedbackTarget | str | None = None,
        polarity: FeedbackPolarity | str | None = None,
        intensity: float | None = None,
        display_strategy: ProactiveDisplayStrategy | str | None = None,
        text: str = "",
        dwell_ms: int | None = None,
        latency_ms: int | None = None,
        helpfulness: float | None = None,
        timing_fit: float | None = None,
        content_fit: float | None = None,
        granularity_fit: float | None = None,
        display_fit: float | None = None,
        task_progress_delta: float | None = None,
        flow_break_score: float | None = None,
        redundancy_score: float | None = None,
        privacy_risk_score: float | None = None,
        missed_opportunity_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeFeedbackEvent:
        decision = self.store.get_decision(decision_id)
        resolved_signal = FeedbackSignalType(signal_type)
        resolved_intensity = intensity if intensity is not None else _default_intensity(resolved_signal, dwell_ms, latency_ms)
        resolved_source = FeedbackSignalSource(source) if source is not None else _default_source(resolved_signal)
        resolved_input_channel = (
            FeedbackInputChannel(input_channel) if input_channel is not None else _default_input_channel(resolved_signal)
        )
        resolved_target = FeedbackTarget(target) if target is not None else _default_target(resolved_signal)
        resolved_polarity = FeedbackPolarity(polarity) if polarity is not None else _default_polarity(resolved_signal)
        defaults = _default_metrics(resolved_signal, resolved_intensity, dwell_ms, latency_ms)
        event = RuntimeFeedbackEvent(
            event_id=event_id or f"fb_{uuid4().hex[:12]}",
            decision_id=decision_id,
            session_id=decision.session_id,
            signal_type=resolved_signal,
            source=resolved_source,
            input_channel=resolved_input_channel,
            target=resolved_target,
            polarity=resolved_polarity,
            intensity=resolved_intensity,
            display_strategy=ProactiveDisplayStrategy(display_strategy) if display_strategy is not None else None,
            text=text,
            dwell_ms=dwell_ms,
            latency_ms=latency_ms,
            helpfulness=helpfulness if helpfulness is not None else defaults["helpfulness"],
            timing_fit=timing_fit if timing_fit is not None else defaults["timing_fit"],
            content_fit=content_fit if content_fit is not None else defaults["content_fit"],
            granularity_fit=granularity_fit if granularity_fit is not None else defaults["granularity_fit"],
            display_fit=display_fit if display_fit is not None else defaults["display_fit"],
            task_progress_delta=(
                task_progress_delta if task_progress_delta is not None else defaults["task_progress_delta"]
            ),
            flow_break_score=flow_break_score if flow_break_score is not None else defaults["flow_break_score"],
            redundancy_score=redundancy_score if redundancy_score is not None else defaults["redundancy_score"],
            privacy_risk_score=privacy_risk_score if privacy_risk_score is not None else defaults["privacy_risk_score"],
            missed_opportunity_score=(
                missed_opportunity_score
                if missed_opportunity_score is not None
                else defaults["missed_opportunity_score"]
            ),
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

    def build_policy_episode(self, session_id: str) -> PolicyEpisode:
        decisions = self.store.list_decisions(session_id=session_id)
        feedback_by_decision = _feedback_by_decision(self.store.list_feedback_events(session_id=session_id))
        reward_by_decision = _latest_reward_by_decision(self.store.list_reward_observations(session_id=session_id))
        steps: list[PolicyStep] = []
        for index, decision in enumerate(decisions):
            next_decision = decisions[index + 1] if index + 1 < len(decisions) else None
            reward = reward_by_decision.get(decision.decision_id)
            step = PolicyStep(
                step_id=f"step_{index:04d}_{decision.decision_id}",
                index=index,
                decision_id=decision.decision_id,
                state=_policy_state_ref(decision),
                action=_policy_action_snapshot(decision),
                feedback_events=feedback_by_decision.get(decision.decision_id, []),
                reward_observation=reward,
                next_state_ref=_next_state_ref(next_decision),
                final_reward=reward.final_reward if reward is not None else None,
                metadata={
                    "candidate_status": _enum_value(decision.candidate.status),
                    "opportunity_priority": _enum_value(decision.candidate.opportunity.priority),
                    "opportunity_reason": decision.candidate.opportunity.reason,
                },
            )
            steps.append(step)
        total_reward = _round(sum(step.final_reward or 0.0 for step in steps))
        policy_versions = {decision.policy_version for decision in decisions}
        return PolicyEpisode(
            episode_id=f"episode_{session_id}",
            session_id=session_id,
            policy_version=next(iter(policy_versions)) if len(policy_versions) == 1 else None,
            steps=steps,
            total_reward=total_reward,
            feedback_event_count=sum(len(step.feedback_events) for step in steps),
            rewarded_step_count=sum(1 for step in steps if step.reward_observation is not None),
            metadata={
                "builder": "runtime_policy_episode_v0",
                "decision_count": len(decisions),
                "policy_versions": sorted(policy_versions),
            },
        )

    def evaluate_policy_episode(self, session_id: str) -> PolicyEvaluationReport:
        episode = self.build_policy_episode(session_id)
        return _evaluate_policy_episode(episode, reward_weights=self.reward_weights)

    def evaluate_policy_baselines(self, session_id: str) -> PolicyBaselineComparisonReport:
        episode = self.build_policy_episode(session_id)
        return _evaluate_policy_baselines(episode)

    def export_policy_training_examples(self, session_id: str) -> PolicyTrainingExport:
        episode = self.build_policy_episode(session_id)
        evaluation = _evaluate_policy_episode(episode, reward_weights=self.reward_weights)
        baselines = _evaluate_policy_baselines(episode)
        return _policy_training_export(episode, evaluation, baselines)


def _decision_id(candidate: PromptCandidate) -> str:
    digest = candidate.candidate_id.removeprefix("cand_")
    return f"dec_{digest}"


def _policy_state_ref(decision: PromptDecisionRecord) -> PolicyStateRef:
    opportunity = decision.candidate.opportunity
    prompt_request = decision.candidate.prompt_request
    return PolicyStateRef(
        session_id=decision.session_id,
        scenario_id=prompt_request.scenario_id,
        transcript_segment_ids=list(opportunity.trigger_segment_ids),
        memory_refs=list(prompt_request.session_context.get("memory_refs", [])),
        retrieved_memory_refs=list(decision.metadata.get("retrieved_memory_refs", [])),
        opportunity_id=decision.opportunity_id,
        captured_text=opportunity.captured_text,
        activity_phase=str(_enum_value(opportunity.activity_phase)),
        prompt_category=decision.prompt_category,
        timing_action=str(_enum_value(opportunity.candidate_timing_action)),
        privacy_level=decision.privacy_level,
        privacy_risk=decision.privacy_risk,
        metadata={
            "detector_confidence": opportunity.confidence,
            "rule_matches": [match.model_dump(mode="json") for match in opportunity.rule_matches],
            "safety_flags": list(opportunity.safety_flags),
            "memory_query_text": decision.metadata.get("memory_query_text", ""),
            "memory_query_prompt_category": decision.metadata.get("memory_query_prompt_category", ""),
            "memory_query_activity_phase": decision.metadata.get("memory_query_activity_phase", ""),
            "retrieved_memory_result_count": decision.metadata.get("retrieved_memory_result_count", 0),
        },
    )


def _policy_action_snapshot(decision: PromptDecisionRecord) -> PolicyActionSnapshot:
    prompt_result = decision.candidate.prompt_result
    opportunity = decision.candidate.opportunity
    return PolicyActionSnapshot(
        decision_id=decision.decision_id,
        candidate_id=decision.candidate_id,
        display_status=decision.display_status,
        should_prompt=_enum_value(decision.display_status) == PromptDecisionDisplayStatus.SHOWN.value,
        timing_action=str(_enum_value(opportunity.candidate_timing_action)),
        prompt_category=decision.prompt_category,
        content_granularity=decision.content_granularity,
        prd_surface=decision.prd_surface,
        display_mode=decision.display_mode,
        display_strategy=_display_strategy_for_decision(decision),
        duration_policy=decision.duration_policy,
        policy_version=decision.policy_version,
        confidence=prompt_result.confidence if prompt_result is not None else 0.0,
        created_at=decision.created_at,
        shown_at=decision.shown_at,
        metadata={
            "candidate_id": decision.candidate_id,
            "opportunity_id": decision.opportunity_id,
            "source_refs": list(prompt_result.source_refs) if prompt_result is not None else [],
            "glasses_title": prompt_result.glasses_title if prompt_result is not None else "",
            "glasses_text": prompt_result.glasses_text if prompt_result is not None else "",
            "app_detail_text": prompt_result.app_detail_text if prompt_result is not None else "",
            "safety_flags": list(prompt_result.safety_flags) if prompt_result is not None else [],
        },
    )


def _display_strategy_for_decision(decision: PromptDecisionRecord) -> ProactiveDisplayStrategy:
    if _enum_value(decision.display_status) != PromptDecisionDisplayStatus.SHOWN.value:
        return ProactiveDisplayStrategy.MANUAL_RESPONSE
    if _enum_value(decision.display_mode) == "auto":
        return ProactiveDisplayStrategy.AUTO_POPUP
    return ProactiveDisplayStrategy.SUBTLE_AVAILABLE


def _next_state_ref(next_decision: PromptDecisionRecord | None) -> PolicyNextStateRef:
    if next_decision is None:
        return PolicyNextStateRef()
    return PolicyNextStateRef(
        next_decision_id=next_decision.decision_id,
        next_transcript_segment_ids=list(next_decision.candidate.opportunity.trigger_segment_ids),
    )


def _feedback_by_decision(events: list[RuntimeFeedbackEvent]) -> dict[str, list[RuntimeFeedbackEvent]]:
    grouped: dict[str, list[RuntimeFeedbackEvent]] = {}
    for event in events:
        grouped.setdefault(event.decision_id, []).append(event)
    return grouped


def _latest_reward_by_decision(observations: list[RewardObservation]) -> dict[str, RewardObservation]:
    latest: dict[str, RewardObservation] = {}
    for observation in observations:
        latest[observation.decision_id] = observation
    return latest


def _evaluate_policy_baselines(episode: PolicyEpisode) -> PolicyBaselineComparisonReport:
    baselines = [
        _baseline_evaluation(episode, PolicyBaselineName.CONSERVATIVE),
        _baseline_evaluation(episode, PolicyBaselineName.BALANCED),
        _baseline_evaluation(episode, PolicyBaselineName.AGGRESSIVE),
    ]
    return PolicyBaselineComparisonReport(
        report_id=f"baseline_{episode.episode_id}",
        episode_id=episode.episode_id,
        session_id=episode.session_id,
        baselines=baselines,
        metadata={
            "evaluator": "runtime_policy_baseline_v0",
            "counterfactual_reward": False,
            "baseline_names": [baseline.baseline_name for baseline in baselines],
        },
    )


def _policy_training_export(
    episode: PolicyEpisode,
    evaluation: PolicyEvaluationReport,
    baselines: PolicyBaselineComparisonReport,
) -> PolicyTrainingExport:
    baseline_decisions = _baseline_decisions_by_decision_id(baselines)
    examples = [
        _policy_training_example(
            episode=episode,
            step=step,
            baseline_decisions=baseline_decisions.get(step.decision_id, []),
        )
        for step in episode.steps
    ]
    return PolicyTrainingExport(
        export_id=f"policy_export_{episode.episode_id}",
        session_id=episode.session_id,
        episode_id=episode.episode_id,
        example_count=len(examples),
        examples=examples,
        evaluation_summary=evaluation.summary,
        metadata={
            "export_version": "policy_training_export_v0",
            "episode_builder": episode.metadata.get("builder", ""),
            "evaluation_report_id": evaluation.report_id,
            "baseline_report_id": baselines.report_id,
            "counterfactual_reward": False,
            "record_format": "jsonl_ready",
            "label_source": "feedback_reward_v1",
        },
    )


def _baseline_decisions_by_decision_id(
    baselines: PolicyBaselineComparisonReport,
) -> dict[str, list[PolicyBaselineDecision]]:
    grouped: dict[str, list[PolicyBaselineDecision]] = {}
    for baseline in baselines.baselines:
        for decision in baseline.decisions:
            grouped.setdefault(decision.decision_id, []).append(decision)
    return grouped


def _policy_training_example(
    *,
    episode: PolicyEpisode,
    step: PolicyStep,
    baseline_decisions: list[PolicyBaselineDecision],
) -> PolicyTrainingExample:
    return PolicyTrainingExample(
        example_id=f"example_{step.step_id}",
        session_id=episode.session_id,
        episode_id=episode.episode_id,
        step_id=step.step_id,
        decision_id=step.decision_id,
        state=step.state,
        action=step.action,
        feedback_events=step.feedback_events,
        reward_observation=step.reward_observation,
        baseline_decisions=baseline_decisions,
        label=_policy_training_label(step),
        next_state_ref=step.next_state_ref,
        metadata={
            "export_version": "policy_training_export_v0",
            "step_index": step.index,
            "policy_version": episode.policy_version,
            "candidate_status": step.metadata.get("candidate_status", ""),
            "opportunity_priority": step.metadata.get("opportunity_priority", ""),
            "opportunity_reason": step.metadata.get("opportunity_reason", ""),
            "baseline_count": len(baseline_decisions),
            "has_counterfactual_reward": False,
        },
    )


def _policy_training_label(step: PolicyStep) -> PolicyTrainingLabel:
    return PolicyTrainingLabel(
        final_reward=step.final_reward,
        has_feedback=bool(step.feedback_events),
        accepted=_step_has_accept_feedback(step),
        negative_feedback=_step_has_negative_feedback(step),
        missed_opportunity=_step_has_missed_opportunity(step),
        interruption=_step_has_interruption(step),
        privacy_rejected=_step_has_privacy_rejection(step),
        granularity_mismatch=_step_has_granularity_mismatch(step),
        display_mismatch=_step_has_display_mismatch(step),
    )


def _baseline_evaluation(episode: PolicyEpisode, baseline_name: PolicyBaselineName) -> PolicyBaselineEvaluation:
    decisions = [_baseline_decision(step, baseline_name) for step in episode.steps]
    return PolicyBaselineEvaluation(
        baseline_name=baseline_name,
        episode_id=episode.episode_id,
        session_id=episode.session_id,
        metrics=_baseline_metrics(decisions),
        decisions=decisions,
    )


def _baseline_decision(step: PolicyStep, baseline_name: PolicyBaselineName) -> PolicyBaselineDecision:
    action = _baseline_action(step, baseline_name)
    original_should_prompt = step.action.should_prompt
    original_score = _action_score(
        original_should_prompt,
        step.action.content_granularity,
        step.action.display_strategy,
    )
    baseline_score = _action_score(action.should_prompt, action.content_granularity, action.display_strategy)
    action_delta = _round(abs(baseline_score - original_score))
    relation = PolicyBaselineRelation.SAME
    if baseline_score < original_score:
        relation = PolicyBaselineRelation.MORE_CONSERVATIVE
    elif baseline_score > original_score:
        relation = PolicyBaselineRelation.MORE_AGGRESSIVE
    matches_original = (
        action.should_prompt == original_should_prompt
        and int(action.content_granularity) == int(step.action.content_granularity)
        and _enum_value(action.display_strategy) == _enum_value(step.action.display_strategy)
    )
    return PolicyBaselineDecision(
        decision_id=step.decision_id,
        baseline_name=baseline_name,
        original_should_prompt=original_should_prompt,
        baseline_action=action,
        matches_original=matches_original,
        relation_to_original=relation,
        action_delta=action_delta,
        missed_opportunity_covered=_step_has_missed_opportunity(step) and action.should_prompt,
        metadata={
            "original_content_granularity": int(step.action.content_granularity),
            "original_display_strategy": _enum_value(step.action.display_strategy),
            "priority": _step_priority(step),
            "privacy_risk": step.state.privacy_risk,
            "detector_confidence": _step_detector_confidence(step),
            "original_missed_opportunity": _step_has_missed_opportunity(step),
        },
    )


def _baseline_action(step: PolicyStep, baseline_name: PolicyBaselineName) -> PolicyBaselineAction:
    if baseline_name == PolicyBaselineName.CONSERVATIVE:
        return _conservative_baseline_action(step)
    if baseline_name == PolicyBaselineName.BALANCED:
        return _balanced_baseline_action(step)
    if baseline_name == PolicyBaselineName.AGGRESSIVE:
        return _aggressive_baseline_action(step)
    raise ValueError(f"unsupported baseline: {baseline_name}")


def _conservative_baseline_action(step: PolicyStep) -> PolicyBaselineAction:
    if _privacy_blocked(step, threshold=0.65):
        return _suppressed_baseline_action("privacy risk exceeds conservative threshold", privacy_blocked=True)
    should_prompt = (
        _step_priority(step) == "P0"
        and _step_prompt_category(step) == "summary_gap_check"
        and _step_detector_confidence(step) >= 0.7
    )
    if not should_prompt:
        return _suppressed_baseline_action("not high-confidence P0 summary gap-check")
    return PolicyBaselineAction(
        should_prompt=True,
        content_granularity=ContentGranularity.ONE_LINE_ANSWER,
        display_strategy=ProactiveDisplayStrategy.SUBTLE_AVAILABLE,
        reason="conservative P0 summary gap-check prompt",
    )


def _balanced_baseline_action(step: PolicyStep) -> PolicyBaselineAction:
    if _privacy_blocked(step, threshold=0.75):
        return _suppressed_baseline_action("privacy risk exceeds balanced threshold", privacy_blocked=True)
    if _step_timing_action(step) == "no_action" or _step_detector_confidence(step) < 0.45:
        return _suppressed_baseline_action("low confidence or no-action timing")
    granularity = ContentGranularity(min(max(int(step.action.content_granularity), 1), 3))
    display_strategy = (
        ProactiveDisplayStrategy.AUTO_POPUP
        if _step_priority(step) == "P0" and step.state.privacy_risk < 0.5
        else ProactiveDisplayStrategy.SUBTLE_AVAILABLE
    )
    return PolicyBaselineAction(
        should_prompt=True,
        content_granularity=granularity,
        display_strategy=display_strategy,
        reason="balanced confidence/privacy prompt",
    )


def _aggressive_baseline_action(step: PolicyStep) -> PolicyBaselineAction:
    if _privacy_blocked(step, threshold=0.9):
        return _suppressed_baseline_action("privacy risk exceeds aggressive threshold", privacy_blocked=True)
    if _step_timing_action(step) == "no_action" or _step_detector_confidence(step) < 0.25:
        return _suppressed_baseline_action("very low confidence or no-action timing")
    granularity = ContentGranularity(min(max(int(step.action.content_granularity), 2), 4))
    display_strategy = (
        ProactiveDisplayStrategy.AUTO_POPUP
        if step.state.privacy_risk < 0.7
        else ProactiveDisplayStrategy.SUBTLE_AVAILABLE
    )
    return PolicyBaselineAction(
        should_prompt=True,
        content_granularity=granularity,
        display_strategy=display_strategy,
        reason="aggressive opportunity coverage prompt",
    )


def _suppressed_baseline_action(reason: str, *, privacy_blocked: bool = False) -> PolicyBaselineAction:
    return PolicyBaselineAction(
        should_prompt=False,
        content_granularity=ContentGranularity.NO_ACTION,
        display_strategy=ProactiveDisplayStrategy.MANUAL_RESPONSE,
        privacy_blocked=privacy_blocked,
        reason=reason,
    )


def _baseline_metrics(decisions: list[PolicyBaselineDecision]) -> PolicyBaselineMetrics:
    decision_count = len(decisions)
    missed_opportunity_decisions = [
        decision for decision in decisions if decision.missed_opportunity_covered or _is_original_missed_opportunity(decision)
    ]
    return PolicyBaselineMetrics(
        decision_count=decision_count,
        agreement_rate=_safe_rate(sum(1 for decision in decisions if decision.matches_original), decision_count),
        more_conservative_rate=_safe_rate(
            sum(1 for decision in decisions if decision.relation_to_original == PolicyBaselineRelation.MORE_CONSERVATIVE),
            decision_count,
        ),
        more_aggressive_rate=_safe_rate(
            sum(1 for decision in decisions if decision.relation_to_original == PolicyBaselineRelation.MORE_AGGRESSIVE),
            decision_count,
        ),
        would_prompt_rate=_safe_rate(sum(1 for decision in decisions if decision.baseline_action.should_prompt), decision_count),
        would_suppress_rate=_safe_rate(
            sum(1 for decision in decisions if not decision.baseline_action.should_prompt),
            decision_count,
        ),
        privacy_block_rate=_safe_rate(
            sum(1 for decision in decisions if decision.baseline_action.privacy_blocked),
            decision_count,
        ),
        missed_opportunity_coverage_rate=_safe_rate(
            sum(1 for decision in missed_opportunity_decisions if decision.missed_opportunity_covered),
            len(missed_opportunity_decisions),
        ),
        average_action_delta=_safe_rate(sum(decision.action_delta for decision in decisions), decision_count),
    )


def _is_original_missed_opportunity(decision: PolicyBaselineDecision) -> bool:
    return bool(decision.metadata.get("original_missed_opportunity", False))


def _action_score(
    should_prompt: bool,
    content_granularity: ContentGranularity,
    display_strategy: ProactiveDisplayStrategy,
) -> float:
    if not should_prompt:
        return 0.0
    display_score = {
        ProactiveDisplayStrategy.MANUAL_RESPONSE.value: 0.0,
        ProactiveDisplayStrategy.SUBTLE_AVAILABLE.value: 0.5,
        ProactiveDisplayStrategy.AUTO_POPUP.value: 1.0,
    }[str(_enum_value(display_strategy))]
    return _round(1.0 + (int(content_granularity) / 4.0) + display_score)


def _privacy_blocked(step: PolicyStep, *, threshold: float) -> bool:
    return step.state.privacy_risk >= threshold or _enum_value(step.state.privacy_level) == PrivacyLevel.HIGH.value


def _step_priority(step: PolicyStep) -> str:
    return str(step.metadata.get("opportunity_priority", "unknown"))


def _step_detector_confidence(step: PolicyStep) -> float:
    value = step.state.metadata.get("detector_confidence", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _evaluate_policy_episode(
    episode: PolicyEpisode,
    *,
    reward_weights: dict[str, float],
) -> PolicyEvaluationReport:
    return PolicyEvaluationReport(
        report_id=f"eval_{episode.episode_id}",
        episode_id=episode.episode_id,
        session_id=episode.session_id,
        policy_version=episode.policy_version,
        summary=_metric_summary(episode.steps),
        by_prompt_category=_breakdown(episode.steps, "prompt_category", _step_prompt_category),
        by_content_granularity=_breakdown(episode.steps, "content_granularity", _step_content_granularity),
        by_timing_action=_breakdown(episode.steps, "timing_action", _step_timing_action),
        by_display_strategy=_breakdown(episode.steps, "display_strategy", _step_display_strategy),
        by_display_status=_breakdown(episode.steps, "display_status", _step_display_status),
        metadata={
            "evaluator": "runtime_policy_evaluation_v0",
            "reward_formula": "prompt_runtime_v0",
            "reward_weights": reward_weights,
        },
    )


def _metric_summary(steps: list[PolicyStep]) -> PolicyMetricSummary:
    step_count = len(steps)
    rewarded_steps = [step for step in steps if step.reward_observation is not None]
    feedback_event_count = sum(len(step.feedback_events) for step in steps)
    total_reward = _round(sum(step.final_reward or 0.0 for step in steps))
    return PolicyMetricSummary(
        step_count=step_count,
        rewarded_step_count=len(rewarded_steps),
        feedback_event_count=feedback_event_count,
        total_reward=total_reward,
        average_reward_per_step=_safe_rate(total_reward, step_count),
        average_reward_per_rewarded_step=_safe_rate(total_reward, len(rewarded_steps)),
        accept_rate=_step_rate(steps, _step_has_accept_feedback),
        negative_feedback_rate=_step_rate(steps, _step_has_negative_feedback),
        missed_opportunity_rate=_step_rate(steps, _step_has_missed_opportunity),
        interruption_rate=_step_rate(steps, _step_has_interruption),
        privacy_rejection_rate=_step_rate(steps, _step_has_privacy_rejection),
        granularity_mismatch_rate=_step_rate(steps, _step_has_granularity_mismatch),
        display_mismatch_rate=_step_rate(steps, _step_has_display_mismatch),
        rewarded_step_coverage=_safe_rate(len(rewarded_steps), step_count),
    )


def _breakdown(
    steps: list[PolicyStep],
    dimension: str,
    key_fn: Any,
) -> list[PolicyActionBreakdown]:
    grouped: dict[str, list[PolicyStep]] = {}
    for step in steps:
        grouped.setdefault(str(key_fn(step)), []).append(step)
    return [
        PolicyActionBreakdown(dimension=dimension, value=value, metrics=_metric_summary(group_steps))
        for value, group_steps in sorted(grouped.items())
    ]


def _step_prompt_category(step: PolicyStep) -> str:
    return step.action.prompt_category or "unknown"


def _step_content_granularity(step: PolicyStep) -> str:
    return str(int(step.action.content_granularity))


def _step_timing_action(step: PolicyStep) -> str:
    return step.action.timing_action or "unknown"


def _step_display_strategy(step: PolicyStep) -> str:
    return str(_enum_value(step.action.display_strategy))


def _step_display_status(step: PolicyStep) -> str:
    return str(_enum_value(step.action.display_status))


def _step_rate(steps: list[PolicyStep], predicate: Any) -> float:
    return _safe_rate(sum(1 for step in steps if predicate(step)), len(steps))


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _round(numerator / denominator)


def _step_has_accept_feedback(step: PolicyStep) -> bool:
    accept_signals = {
        FeedbackSignalType.ACCEPT.value,
        FeedbackSignalType.NOD_ACCEPT.value,
        FeedbackSignalType.DEFAULT_ACCEPT.value,
        FeedbackSignalType.ASK_FOLLOWUP.value,
        FeedbackSignalType.OPEN_DETAIL.value,
        FeedbackSignalType.SAVE.value,
        FeedbackSignalType.MARK_HELPFUL.value,
        FeedbackSignalType.VERBAL_POSITIVE.value,
    }
    return any(_enum_value(event.signal_type) in accept_signals for event in step.feedback_events)


def _step_has_negative_feedback(step: PolicyStep) -> bool:
    return any(_enum_value(event.polarity) == FeedbackPolarity.NEGATIVE.value for event in step.feedback_events)


def _step_has_missed_opportunity(step: PolicyStep) -> bool:
    if step.reward_observation is not None and step.reward_observation.components.missed_opportunity > 0.0:
        return True
    return any(_enum_value(event.signal_type) == FeedbackSignalType.MANUAL_REQUEST.value for event in step.feedback_events)


def _step_has_interruption(step: PolicyStep) -> bool:
    interruption_signals = {
        FeedbackSignalType.DISMISS.value,
        FeedbackSignalType.SNOOZE.value,
        FeedbackSignalType.DISABLE_AUTO_PROMPT.value,
        FeedbackSignalType.SWITCH_TO_MANUAL.value,
        FeedbackSignalType.COMPLAIN_INTERRUPTIVE.value,
        FeedbackSignalType.FLOW_BREAK.value,
    }
    if step.reward_observation is not None and step.reward_observation.components.flow_break > 0.0:
        return True
    return any(_enum_value(event.signal_type) in interruption_signals for event in step.feedback_events)


def _step_has_privacy_rejection(step: PolicyStep) -> bool:
    return any(_enum_value(event.signal_type) == FeedbackSignalType.PRIVACY_REJECT.value for event in step.feedback_events)


def _step_has_granularity_mismatch(step: PolicyStep) -> bool:
    if step.reward_observation is not None and step.reward_observation.components.granularity_fit < 0.0:
        return True
    return any(
        _enum_value(event.target) == FeedbackTarget.GRANULARITY.value
        and _enum_value(event.polarity) == FeedbackPolarity.NEGATIVE.value
        for event in step.feedback_events
    )


def _step_has_display_mismatch(step: PolicyStep) -> bool:
    if step.reward_observation is not None and step.reward_observation.components.display_fit < 0.0:
        return True
    return any(
        _enum_value(event.target) == FeedbackTarget.DISPLAY.value
        and _enum_value(event.polarity) == FeedbackPolarity.NEGATIVE.value
        for event in step.feedback_events
    )


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
        FeedbackSignalType.DEFAULT_ACCEPT,
        FeedbackSignalType.FLOW_BREAK,
        FeedbackSignalType.REDUNDANT_TRIGGER,
    }:
        return FeedbackSignalSource.IMPLICIT
    if signal in {FeedbackSignalType.TASK_PROGRESS, FeedbackSignalType.TASK_REGRESSION}:
        return FeedbackSignalSource.TASK_OUTCOME
    if signal == FeedbackSignalType.LATENCY_OBSERVED:
        return FeedbackSignalSource.SYSTEM
    return FeedbackSignalSource.EXPLICIT


def _default_input_channel(signal: FeedbackSignalType) -> FeedbackInputChannel:
    if signal in {FeedbackSignalType.NOD_ACCEPT, FeedbackSignalType.HEAD_SHAKE_REJECT}:
        return FeedbackInputChannel.GESTURE
    if signal == FeedbackSignalType.MANUAL_REQUEST:
        return FeedbackInputChannel.BUTTON
    if signal in {FeedbackSignalType.VERBAL_POSITIVE, FeedbackSignalType.VERBAL_REJECT}:
        return FeedbackInputChannel.VOICE
    if signal in {FeedbackSignalType.SECOND_LOOK, FeedbackSignalType.IGNORE}:
        return FeedbackInputChannel.GAZE
    if signal == FeedbackSignalType.DWELL:
        return FeedbackInputChannel.DWELL_TIME
    if signal in {FeedbackSignalType.TASK_PROGRESS, FeedbackSignalType.TASK_REGRESSION}:
        return FeedbackInputChannel.CONVERSATION_OUTCOME
    if signal == FeedbackSignalType.LATENCY_OBSERVED:
        return FeedbackInputChannel.SYSTEM
    return FeedbackInputChannel.UNKNOWN


def _default_target(signal: FeedbackSignalType) -> FeedbackTarget:
    if signal in {
        FeedbackSignalType.MANUAL_REQUEST,
        FeedbackSignalType.SNOOZE,
        FeedbackSignalType.COMPLAIN_INTERRUPTIVE,
        FeedbackSignalType.FLOW_BREAK,
    }:
        return FeedbackTarget.TIMING
    if signal in {FeedbackSignalType.ASK_FOLLOWUP, FeedbackSignalType.OPEN_DETAIL}:
        return FeedbackTarget.GRANULARITY
    if signal in {FeedbackSignalType.MARK_HELPFUL, FeedbackSignalType.MARK_NOT_HELPFUL, FeedbackSignalType.SAVE}:
        return FeedbackTarget.CONTENT
    if signal in {FeedbackSignalType.DISABLE_AUTO_PROMPT, FeedbackSignalType.SWITCH_TO_MANUAL}:
        return FeedbackTarget.DISPLAY
    if signal == FeedbackSignalType.REDUNDANT_TRIGGER:
        return FeedbackTarget.FREQUENCY
    if signal == FeedbackSignalType.PRIVACY_REJECT:
        return FeedbackTarget.PRIVACY
    if signal in {FeedbackSignalType.TASK_PROGRESS, FeedbackSignalType.TASK_REGRESSION}:
        return FeedbackTarget.TASK_OUTCOME
    if signal == FeedbackSignalType.LATENCY_OBSERVED:
        return FeedbackTarget.LATENCY
    return FeedbackTarget.OVERALL


def _default_polarity(signal: FeedbackSignalType) -> FeedbackPolarity:
    if signal in {
        FeedbackSignalType.ACCEPT,
        FeedbackSignalType.NOD_ACCEPT,
        FeedbackSignalType.DEFAULT_ACCEPT,
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
    if signal == FeedbackSignalType.DEFAULT_ACCEPT:
        return 0.4
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
    if signal == FeedbackSignalType.HEAD_SHAKE_REJECT:
        return 1.0
    if signal == FeedbackSignalType.MANUAL_REQUEST:
        return 1.0
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
        "granularity_fit": 0.0,
        "display_fit": 0.0,
        "task_progress_delta": 0.0,
        "flow_break_score": 0.0,
        "redundancy_score": 0.0,
        "privacy_risk_score": 0.0,
        "missed_opportunity_score": 0.0,
    }
    if signal in {
        FeedbackSignalType.ACCEPT,
        FeedbackSignalType.NOD_ACCEPT,
        FeedbackSignalType.DEFAULT_ACCEPT,
        FeedbackSignalType.VERBAL_POSITIVE,
    }:
        metrics.update(helpfulness=0.7 * intensity, timing_fit=0.5, content_fit=0.5)
    elif signal == FeedbackSignalType.ASK_FOLLOWUP:
        metrics.update(
            helpfulness=0.9 * intensity,
            timing_fit=0.4,
            content_fit=0.6,
            granularity_fit=-0.4,
            task_progress_delta=0.3,
        )
    elif signal == FeedbackSignalType.OPEN_DETAIL:
        metrics.update(helpfulness=0.6 * intensity, timing_fit=0.3, content_fit=0.3, granularity_fit=-0.3)
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
    elif signal == FeedbackSignalType.HEAD_SHAKE_REJECT:
        metrics.update(timing_fit=-0.8, content_fit=-0.5, display_fit=-0.3)
    elif signal == FeedbackSignalType.MANUAL_REQUEST:
        metrics.update(
            timing_fit=-1.0,
            content_fit=-0.4,
            display_fit=-0.4,
            missed_opportunity_score=intensity,
        )
    elif signal == FeedbackSignalType.PRIVACY_REJECT:
        metrics.update(content_fit=-1.0, privacy_risk_score=1.0)
    elif signal in {FeedbackSignalType.DISABLE_AUTO_PROMPT, FeedbackSignalType.SWITCH_TO_MANUAL}:
        metrics.update(timing_fit=-0.8, display_fit=-0.8, flow_break_score=0.5)
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
        FeedbackSignalType.NOD_ACCEPT,
        FeedbackSignalType.DEFAULT_ACCEPT,
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
        FeedbackSignalType.HEAD_SHAKE_REJECT,
        FeedbackSignalType.DISABLE_AUTO_PROMPT,
        FeedbackSignalType.SWITCH_TO_MANUAL,
        FeedbackSignalType.COMPLAIN_INTERRUPTIVE,
    }
    timing_values = [event.timing_fit for event in events if event.timing_fit != 0.0]
    content_values = [event.content_fit for event in events if event.content_fit != 0.0]
    granularity_values = [event.granularity_fit for event in events if event.granularity_fit != 0.0]
    display_values = [event.display_fit for event in events if event.display_fit != 0.0]
    return RewardComponents(
        accept=_max_intensity(events, accept_signals),
        helpfulness=_max_value(event.helpfulness for event in events),
        task_progress=_clip(sum(event.task_progress_delta for event in events), -1.0, 1.0),
        timing_fit=_avg(timing_values),
        content_fit=_avg(content_values),
        granularity_fit=_avg(granularity_values),
        display_fit=_avg(display_values),
        annoyance=_max_intensity(events, annoyance_signals),
        flow_break=_max_value(event.flow_break_score for event in events),
        redundancy=_max_value(event.redundancy_score for event in events),
        privacy_risk=max(decision.privacy_risk, _max_value(event.privacy_risk_score for event in events)),
        latency_penalty=_max_value(_latency_penalty(event) for event in events),
        missed_opportunity=_max_value(event.missed_opportunity_score for event in events),
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
