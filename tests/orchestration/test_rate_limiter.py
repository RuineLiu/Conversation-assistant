from proactive_assistant.detection import (
    CandidateTimingAction,
    DetectionRuleMatch,
    PromptOpportunity,
    PromptPriority,
)
from proactive_assistant.orchestration import (
    InMemoryRateLimitHistory,
    RateLimitAction,
    RateLimitConfig,
    RateLimitReason,
    RateLimiter,
    rate_limit_metadata,
)
from proactive_assistant.prompting import (
    ContentGranularity,
    PRDSurface,
    PrivacyLevel,
    PromptCategory,
)
from proactive_assistant.schemas.scenario import ActivityPhase


class FakeClock:
    def __init__(self, t: int = 0) -> None:
        self._t = t

    def now_ms(self) -> int:
        return self._t

    def advance(self, ms: int) -> None:
        self._t += ms


def make_opportunity(
    *,
    opportunity_id: str = "opp_001",
    session_id: str = "session_001",
    triggers: tuple[str, ...] = ("seg_0",),
    category: PromptCategory = PromptCategory.QUESTION_ANSWER,
    phase: ActivityPhase = ActivityPhase.IN_ACTIVITY,
    priority: PromptPriority = PromptPriority.P1,
    timing: CandidateTimingAction = CandidateTimingAction.DURING_ACTIVITY,
    granularity: ContentGranularity = ContentGranularity.ONE_LINE_ANSWER,
    privacy_level: PrivacyLevel = PrivacyLevel.LOW,
    privacy_risk: float = 0.1,
) -> PromptOpportunity:
    return PromptOpportunity(
        opportunity_id=opportunity_id,
        session_id=session_id,
        trigger_segment_ids=list(triggers),
        captured_text="负责人是谁？",
        prompt_category=category,
        activity_phase=phase,
        candidate_timing_action=timing,
        suggested_content_granularity=granularity,
        priority=priority,
        confidence=0.7,
        privacy_level=privacy_level,
        privacy_risk=privacy_risk,
        reason="test opportunity",
        rule_matches=[
            DetectionRuleMatch(rule_name="test", matched_terms=["?"], confidence_delta=0.0, reason="t")
        ],
    )


def test_allow_for_app_surface_with_passthrough_reason() -> None:
    limiter = RateLimiter(clock=FakeClock(0))
    decision = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(),
        surface=PRDSurface.APP_PROMPT_TAB,
        phase=ActivityPhase.IN_ACTIVITY,
    )

    assert decision.action == RateLimitAction.ALLOW
    assert RateLimitReason.NOT_GLASSES_SURFACE in decision.reasons


def test_allow_first_glasses_show() -> None:
    limiter = RateLimiter(clock=FakeClock(0))
    decision = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )

    assert decision.action == RateLimitAction.ALLOW
    assert RateLimitReason.PASSTHROUGH in decision.reasons


def test_defer_when_within_surface_cooldown() -> None:
    clock = FakeClock(0)
    limiter = RateLimiter(config=RateLimitConfig(surface_cooldown_ms=20_000), clock=clock)
    opportunity = make_opportunity()

    # First call shows, second within cooldown defers.
    first = limiter.decide(
        session_id="session_001",
        opportunity=opportunity,
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )
    assert first.action == RateLimitAction.ALLOW
    limiter.record_show(
        session_id="session_001",
        opportunity=opportunity,
        surface=PRDSurface.GLASSES_POPUP,
    )

    clock.advance(5_000)
    second = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(opportunity_id="opp_002", triggers=("seg_1",)),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )

    assert second.action == RateLimitAction.DEFER_TO_APP
    assert RateLimitReason.SURFACE_COOLDOWN in second.reasons
    assert second.cooldown_remaining_ms is not None
    assert 14_000 < second.cooldown_remaining_ms <= 15_000
    assert second.new_surface == PRDSurface.APP_PROMPT_TAB


def test_allow_after_cooldown_elapses() -> None:
    clock = FakeClock(0)
    limiter = RateLimiter(config=RateLimitConfig(surface_cooldown_ms=10_000), clock=clock)
    opportunity = make_opportunity()
    limiter.record_show(
        session_id="session_001",
        opportunity=opportunity,
        surface=PRDSurface.GLASSES_POPUP,
    )

    clock.advance(10_001)
    decision = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(opportunity_id="opp_002", triggers=("seg_1",)),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )

    assert decision.action == RateLimitAction.ALLOW


def test_dedup_window_defers_same_entity_repeat() -> None:
    clock = FakeClock(0)
    limiter = RateLimiter(
        config=RateLimitConfig(surface_cooldown_ms=0, dedup_window_ms=30_000),
        clock=clock,
    )
    opportunity = make_opportunity()
    limiter.record_show(
        session_id="session_001",
        opportunity=opportunity,
        surface=PRDSurface.GLASSES_POPUP,
    )

    clock.advance(10_000)
    # Same triggers and category -> same dedup key
    decision = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(opportunity_id="opp_repeat"),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )

    assert decision.action == RateLimitAction.DEFER_TO_APP
    assert RateLimitReason.DEDUP_WINDOW in decision.reasons


def test_burst_suppression_when_no_cooldown_active() -> None:
    clock = FakeClock(0)
    limiter = RateLimiter(
        config=RateLimitConfig(
            surface_cooldown_ms=0,
            dedup_window_ms=0,
            burst_window_ms=5_000,
            burst_max_per_window=1,
        ),
        clock=clock,
    )
    opportunity_a = make_opportunity(triggers=("seg_a",))
    limiter.record_show(
        session_id="session_001",
        opportunity=opportunity_a,
        surface=PRDSurface.GLASSES_POPUP,
    )

    clock.advance(2_000)
    opportunity_b = make_opportunity(opportunity_id="opp_b", triggers=("seg_b",))
    decision = limiter.decide(
        session_id="session_001",
        opportunity=opportunity_b,
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )

    assert decision.action == RateLimitAction.DEFER_TO_APP
    assert RateLimitReason.BURST_SUPPRESSED in decision.reasons


def test_pre_activity_only_allows_p0_on_glasses() -> None:
    limiter = RateLimiter(clock=FakeClock(0))
    p1 = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(priority=PromptPriority.P1, phase=ActivityPhase.PRE_ACTIVITY),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.PRE_ACTIVITY,
    )
    p0 = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(
            opportunity_id="opp_p0",
            triggers=("seg_p0",),
            priority=PromptPriority.P0,
            phase=ActivityPhase.PRE_ACTIVITY,
        ),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.PRE_ACTIVITY,
    )

    assert p1.action == RateLimitAction.DEFER_TO_APP
    assert RateLimitReason.PRE_ACTIVITY_NON_P0 in p1.reasons
    assert p0.action == RateLimitAction.ALLOW


def test_post_activity_always_defers_glasses() -> None:
    limiter = RateLimiter(clock=FakeClock(0))
    decision = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(priority=PromptPriority.P0, phase=ActivityPhase.POST_ACTIVITY),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.POST_ACTIVITY,
    )

    assert decision.action == RateLimitAction.DEFER_TO_APP
    assert RateLimitReason.POST_ACTIVITY_GLASSES in decision.reasons


def test_enforcer_fallback_signal_defers_immediately() -> None:
    limiter = RateLimiter(clock=FakeClock(0))
    decision = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
        enforcer_fallback_to_app=True,
    )

    assert decision.action == RateLimitAction.DEFER_TO_APP
    assert RateLimitReason.ENFORCER_FALLBACK in decision.reasons


def test_drop_when_no_app_fallback_configured() -> None:
    clock = FakeClock(0)
    limiter = RateLimiter(
        config=RateLimitConfig(
            surface_cooldown_ms=20_000,
            drop_when_no_app_fallback=True,
        ),
        clock=clock,
    )
    opportunity = make_opportunity()
    limiter.record_show(
        session_id="session_001",
        opportunity=opportunity,
        surface=PRDSurface.GLASSES_POPUP,
    )

    decision = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(opportunity_id="opp_002", triggers=("seg_1",)),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )

    assert decision.action == RateLimitAction.DROP
    assert decision.new_surface is None


def test_record_show_only_tracks_glasses_surfaces() -> None:
    history = InMemoryRateLimitHistory()
    limiter = RateLimiter(history=history, clock=FakeClock(0))
    opportunity = make_opportunity()

    # App surface record_show should be a no-op
    limiter.record_show(
        session_id="session_001",
        opportunity=opportunity,
        surface=PRDSurface.APP_PROMPT_TAB,
    )

    assert history.last_shown_on_surface(session_id="session_001", surface="glasses_popup") is None
    assert history.last_shown_on_surface(session_id="session_001", surface="app_prompt_tab") is None


def test_history_isolation_across_sessions() -> None:
    clock = FakeClock(0)
    limiter = RateLimiter(
        config=RateLimitConfig(surface_cooldown_ms=20_000),
        clock=clock,
    )
    limiter.record_show(
        session_id="session_a",
        opportunity=make_opportunity(session_id="session_a"),
        surface=PRDSurface.GLASSES_POPUP,
    )

    decision = limiter.decide(
        session_id="session_b",
        opportunity=make_opportunity(session_id="session_b", opportunity_id="opp_b"),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )

    assert decision.action == RateLimitAction.ALLOW


def test_rate_limit_metadata_serializes_decision_for_ledger() -> None:
    clock = FakeClock(0)
    limiter = RateLimiter(config=RateLimitConfig(surface_cooldown_ms=20_000), clock=clock)
    opportunity = make_opportunity()
    limiter.record_show(
        session_id="session_001",
        opportunity=opportunity,
        surface=PRDSurface.GLASSES_POPUP,
    )

    clock.advance(1_000)
    decision = limiter.decide(
        session_id="session_001",
        opportunity=make_opportunity(opportunity_id="opp_002", triggers=("seg_1",)),
        surface=PRDSurface.GLASSES_POPUP,
        phase=ActivityPhase.IN_ACTIVITY,
    )
    payload = rate_limit_metadata(decision)

    assert payload["rate_limit_action"] == RateLimitAction.DEFER_TO_APP.value
    assert RateLimitReason.SURFACE_COOLDOWN.value in payload["rate_limit_reasons"]
    assert payload["rate_limit_new_surface"] == PRDSurface.APP_PROMPT_TAB.value
    assert payload["rate_limit_cooldown_remaining_ms"] == 19_000


def test_in_memory_history_dedup_key_lookup() -> None:
    history = InMemoryRateLimitHistory()
    history.record_show(
        session_id="session_001",
        surface="glasses_popup",
        dedup_key="glasses_popup:question_answer:seg_0",
        priority="P1",
        shown_at_ms=1_000,
    )

    assert history.shown_for_dedup_key(
        session_id="session_001",
        dedup_key="glasses_popup:question_answer:seg_0",
        since_ms=0,
    )
    assert not history.shown_for_dedup_key(
        session_id="session_001",
        dedup_key="glasses_popup:question_answer:seg_other",
        since_ms=0,
    )
    assert not history.shown_for_dedup_key(
        session_id="session_001",
        dedup_key="glasses_popup:question_answer:seg_0",
        since_ms=2_000,
    )
