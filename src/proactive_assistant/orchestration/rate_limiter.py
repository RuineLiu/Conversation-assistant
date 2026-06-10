"""Per-surface and per-session rate limiting for prompt candidates.

The rate limiter is a deterministic post-processor that decides, for each
``PromptCandidate``, whether the candidate should display on its requested
surface, be rerouted to a less intrusive surface, or be dropped entirely.

It operates on a session-scoped recent-history window. By default the
history lives in-process (``InMemoryRateLimitHistory``); production callers
can plug an alternative backend by supplying their own ``RateLimitHistory``.

The limiter does not call any model and never edits prompt content (length
enforcement is the enforcer's job). It is also non-destructive at decide
time: ``record_show()`` is called by the orchestrator only after the
upstream actually accepts the candidate, so test rollouts and dry-run flows
do not pollute the history.
"""

from __future__ import annotations

import time
from collections import deque
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.detection import PromptOpportunity, PromptPriority
from proactive_assistant.prompting import PRDSurface
from proactive_assistant.schemas.scenario import ActivityPhase


GLASSES_SURFACE_VALUES: frozenset[str] = frozenset(
    {
        PRDSurface.GLASSES_POPUP.value,
        PRDSurface.GLASSES_STARTING.value,
        PRDSurface.GLASSES_PERSISTENT.value,
    }
)


class RateLimitAction(StrEnum):
    """The decision the limiter wants the orchestrator to apply."""

    ALLOW = "allow"
    DEFER_TO_APP = "defer_to_app"
    DROP = "drop"


class RateLimitReason(StrEnum):
    """Why the limiter chose its action. Multiple reasons may apply."""

    PASSTHROUGH = "passthrough"
    NOT_GLASSES_SURFACE = "not_glasses_surface"
    SURFACE_COOLDOWN = "surface_cooldown"
    BURST_SUPPRESSED = "burst_suppressed"
    DEDUP_WINDOW = "dedup_window"
    PRE_ACTIVITY_NON_P0 = "pre_activity_non_p0"
    POST_ACTIVITY_GLASSES = "post_activity_glasses"
    ENFORCER_FALLBACK = "enforcer_fallback_to_app"


class RateLimitConfig(BaseModel):
    """Thresholds and behavior toggles for the rate limiter.

    All times are in milliseconds. Defaults are tuned for the demo:
    glasses surfaces enforce a 20s minimum interval, a 5s burst window
    that allows at most one show, and a 30s same-entity dedup window.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    surface_cooldown_ms: int = Field(default=20_000, ge=0)
    burst_window_ms: int = Field(default=5_000, ge=0)
    burst_max_per_window: int = Field(default=1, ge=1)
    dedup_window_ms: int = Field(default=30_000, ge=0)
    pre_activity_allow_priority: PromptPriority = PromptPriority.P0
    drop_when_no_app_fallback: bool = False


class RateLimitDecision(BaseModel):
    """Outcome of running the limiter on a single candidate."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    action: RateLimitAction
    reasons: list[RateLimitReason] = Field(default_factory=list)
    new_surface: PRDSurface | None = None
    cooldown_remaining_ms: int | None = None
    notes: dict[str, Any] = Field(default_factory=dict)


class _ShownRecord:
    """In-memory record of one shown glasses-surface prompt."""

    __slots__ = ("surface", "dedup_key", "shown_at_ms", "priority")

    def __init__(self, *, surface: str, dedup_key: str, shown_at_ms: int, priority: str) -> None:
        self.surface = surface
        self.dedup_key = dedup_key
        self.shown_at_ms = shown_at_ms
        self.priority = priority


@runtime_checkable
class RateLimitHistory(Protocol):
    """Recent-shows lookup contract used by ``RateLimiter``.

    Implementations should be cheap for the hot path: each ``decide()`` call
    issues up to three queries (last on surface, count in burst window,
    dedup membership).
    """

    def last_shown_on_surface(self, *, session_id: str, surface: str) -> int | None: ...

    def shown_in_window(self, *, session_id: str, surface: str, since_ms: int) -> int: ...

    def shown_for_dedup_key(self, *, session_id: str, dedup_key: str, since_ms: int) -> bool: ...

    def record_show(
        self,
        *,
        session_id: str,
        surface: str,
        dedup_key: str,
        priority: str,
        shown_at_ms: int,
    ) -> None: ...


class InMemoryRateLimitHistory:
    """In-process recent-shows store keyed on session id.

    Bounded deque per session; suitable for single-process demos and tests.
    Replace with a runtime-ledger-backed store for multi-worker production.
    """

    def __init__(self, *, max_records_per_session: int = 200) -> None:
        self._records: dict[str, deque[_ShownRecord]] = {}
        self._max = max_records_per_session

    def last_shown_on_surface(self, *, session_id: str, surface: str) -> int | None:
        records = self._records.get(session_id)
        if not records:
            return None
        for record in reversed(records):
            if record.surface == surface:
                return record.shown_at_ms
        return None

    def shown_in_window(self, *, session_id: str, surface: str, since_ms: int) -> int:
        records = self._records.get(session_id)
        if not records:
            return 0
        return sum(
            1
            for record in records
            if record.surface == surface and record.shown_at_ms >= since_ms
        )

    def shown_for_dedup_key(self, *, session_id: str, dedup_key: str, since_ms: int) -> bool:
        records = self._records.get(session_id)
        if not records:
            return False
        return any(
            record.dedup_key == dedup_key and record.shown_at_ms >= since_ms
            for record in records
        )

    def record_show(
        self,
        *,
        session_id: str,
        surface: str,
        dedup_key: str,
        priority: str,
        shown_at_ms: int,
    ) -> None:
        bucket = self._records.setdefault(session_id, deque(maxlen=self._max))
        bucket.append(
            _ShownRecord(
                surface=surface,
                dedup_key=dedup_key,
                shown_at_ms=shown_at_ms,
                priority=priority,
            )
        )


@runtime_checkable
class Clock(Protocol):
    def now_ms(self) -> int: ...


class WallClock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class RateLimiter:
    """Decide whether a prompt candidate should display, defer, or drop.

    Usage::

        limiter = RateLimiter()
        decision = limiter.decide(
            session_id=session_id,
            opportunity=opportunity,
            surface=prompt_request.prd_surface,
            phase=opportunity.activity_phase,
            enforcer_fallback_to_app=outcome.fallback_to_app_surface,
        )
        # apply decision to candidate, then if effectively shown on glasses:
        limiter.record_show(
            session_id=session_id,
            opportunity=opportunity,
            surface=effective_surface,
        )
    """

    def __init__(
        self,
        *,
        config: RateLimitConfig | None = None,
        history: RateLimitHistory | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or RateLimitConfig()
        self._history = history if history is not None else InMemoryRateLimitHistory()
        self._clock = clock if clock is not None else WallClock()

    @property
    def config(self) -> RateLimitConfig:
        return self._config

    @property
    def history(self) -> RateLimitHistory:
        return self._history

    def now_ms(self) -> int:
        return self._clock.now_ms()

    def decide(
        self,
        *,
        session_id: str,
        opportunity: PromptOpportunity,
        surface: PRDSurface | str,
        phase: ActivityPhase | str,
        enforcer_fallback_to_app: bool = False,
        now_ms: int | None = None,
    ) -> RateLimitDecision:
        resolved_now = now_ms if now_ms is not None else self.now_ms()
        surface_value = _surface_value(surface)
        priority_value = _priority_value(opportunity.priority)
        phase_value = _phase_value(phase)

        # App surfaces pass through. The limiter only constrains glasses.
        if surface_value not in GLASSES_SURFACE_VALUES:
            return RateLimitDecision(
                action=RateLimitAction.ALLOW,
                reasons=[RateLimitReason.NOT_GLASSES_SURFACE],
            )

        # Enforcer already concluded the glasses copy was unusable.
        if enforcer_fallback_to_app:
            return self._build_defer(
                reasons=[RateLimitReason.ENFORCER_FALLBACK],
            )

        # Phase rules.
        allow_priority = (
            self._config.pre_activity_allow_priority
            if isinstance(self._config.pre_activity_allow_priority, str)
            else self._config.pre_activity_allow_priority.value
        )
        if phase_value == ActivityPhase.PRE_ACTIVITY.value and priority_value != allow_priority:
            return self._build_defer(reasons=[RateLimitReason.PRE_ACTIVITY_NON_P0])
        if phase_value == ActivityPhase.POST_ACTIVITY.value:
            return self._build_defer(reasons=[RateLimitReason.POST_ACTIVITY_GLASSES])

        # Same-entity dedup.
        dedup_key = _dedup_key(opportunity, surface_value)
        if self._config.dedup_window_ms > 0 and self._history.shown_for_dedup_key(
            session_id=session_id,
            dedup_key=dedup_key,
            since_ms=resolved_now - self._config.dedup_window_ms,
        ):
            return self._build_defer(reasons=[RateLimitReason.DEDUP_WINDOW])

        # Per-surface cooldown.
        last_shown = self._history.last_shown_on_surface(
            session_id=session_id, surface=surface_value
        )
        if last_shown is not None and self._config.surface_cooldown_ms > 0:
            elapsed = resolved_now - last_shown
            if elapsed < self._config.surface_cooldown_ms:
                remaining = self._config.surface_cooldown_ms - elapsed
                return self._build_defer(
                    reasons=[RateLimitReason.SURFACE_COOLDOWN],
                    cooldown_remaining_ms=remaining,
                )

        # Burst suppression (extra guard inside the cooldown window).
        if self._config.burst_window_ms > 0:
            burst_since = resolved_now - self._config.burst_window_ms
            burst_count = self._history.shown_in_window(
                session_id=session_id,
                surface=surface_value,
                since_ms=burst_since,
            )
            if burst_count >= self._config.burst_max_per_window:
                return self._build_defer(reasons=[RateLimitReason.BURST_SUPPRESSED])

        return RateLimitDecision(
            action=RateLimitAction.ALLOW,
            reasons=[RateLimitReason.PASSTHROUGH],
        )

    def record_show(
        self,
        *,
        session_id: str,
        opportunity: PromptOpportunity,
        surface: PRDSurface | str,
        now_ms: int | None = None,
    ) -> None:
        """Record that a candidate is being shown on a glasses surface.

        App-surface shows are ignored: the limiter only tracks glasses so
        the history stays small and the hot path stays cheap.
        """
        surface_value = _surface_value(surface)
        if surface_value not in GLASSES_SURFACE_VALUES:
            return
        resolved_now = now_ms if now_ms is not None else self.now_ms()
        self._history.record_show(
            session_id=session_id,
            surface=surface_value,
            dedup_key=_dedup_key(opportunity, surface_value),
            priority=_priority_value(opportunity.priority),
            shown_at_ms=resolved_now,
        )

    def _build_defer(
        self,
        *,
        reasons: list[RateLimitReason],
        cooldown_remaining_ms: int | None = None,
    ) -> RateLimitDecision:
        if self._config.drop_when_no_app_fallback:
            return RateLimitDecision(
                action=RateLimitAction.DROP,
                reasons=reasons,
                cooldown_remaining_ms=cooldown_remaining_ms,
            )
        return RateLimitDecision(
            action=RateLimitAction.DEFER_TO_APP,
            reasons=reasons,
            new_surface=PRDSurface.APP_PROMPT_TAB,
            cooldown_remaining_ms=cooldown_remaining_ms,
        )


def _surface_value(surface: PRDSurface | str) -> str:
    return surface.value if hasattr(surface, "value") else str(surface)


def _priority_value(priority: PromptPriority | str) -> str:
    return priority.value if hasattr(priority, "value") else str(priority)


def _phase_value(phase: ActivityPhase | str) -> str:
    return phase.value if hasattr(phase, "value") else str(phase)


def _dedup_key(opportunity: PromptOpportunity, surface_value: str) -> str:
    triggers = "|".join(sorted(opportunity.trigger_segment_ids))
    category = (
        opportunity.prompt_category.value
        if hasattr(opportunity.prompt_category, "value")
        else str(opportunity.prompt_category)
    )
    return f"{surface_value}:{category}:{triggers}"


def rate_limit_metadata(decision: RateLimitDecision) -> dict[str, Any]:
    """Compact metadata payload suitable for ``PromptCandidate.metadata``."""

    return {
        "rate_limit_action": _value(decision.action),
        "rate_limit_reasons": [_value(reason) for reason in decision.reasons],
        "rate_limit_new_surface": (
            None if decision.new_surface is None else _value(decision.new_surface)
        ),
        "rate_limit_cooldown_remaining_ms": decision.cooldown_remaining_ms,
        "rate_limit_notes": dict(decision.notes),
    }


def _value(value: Any) -> Any:
    if value is None:
        return None
    return value if isinstance(value, str) else value.value
