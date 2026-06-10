"""Hard length and surface enforcement for glasses-bound prompt results.

The enforcer is a deterministic post-processor for ``PromptGenerationResult``.
It applies locale-aware unit counting (CJK characters vs English words),
truncates over-budget ``glasses_title`` / ``glasses_text`` at punctuation
boundaries when possible, downgrades ``content_granularity`` when text cannot
be made to fit, and preserves the original verbose copy in ``app_detail_text``
so the companion app surface can still show full content.

The enforcer never calls a model. It is meant to run inside the orchestrator
after prompt generation (LLM or rule-based) and before the candidate is logged.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.prompting.contracts import (
    ContentGranularity,
    PRDSurface,
    PromptGenerationResult,
)


_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION = "，。！？；：、,.!?;:\"'“”‘’()（）[]【】"
_SENTENCE_END = "。！？.!?"
_CLAUSE_END = "，；,;、"


GLASSES_SURFACES: frozenset[str] = frozenset(
    {
        PRDSurface.GLASSES_POPUP.value,
        PRDSurface.GLASSES_STARTING.value,
        PRDSurface.GLASSES_PERSISTENT.value,
    }
)


class LengthMode(StrEnum):
    """How to count length units for a given locale."""

    CJK_CHARS = "cjk_chars"
    EN_WORDS = "en_words"


class EnforcementAction(StrEnum):
    """What the enforcer did to a prompt result."""

    NO_OP = "no_op"
    NOT_GLASSES_SURFACE = "not_glasses_surface"
    NOT_PROMPTING = "not_prompting"
    TITLE_TRUNCATED = "title_truncated"
    TEXT_TRUNCATED = "text_truncated"
    GRANULARITY_DOWNGRADED = "granularity_downgraded"
    DETAIL_FALLBACK_FILLED = "detail_fallback_filled"


class GlassesLengthLimits(BaseModel):
    """Hard caps for glasses-surface prompt text.

    Defaults match the demo spec: glasses popup shows ≤30 Chinese chars or
    ≤20 English words; title stays under 12 chars / 6 words.
    """

    model_config = ConfigDict(extra="forbid")

    title_cjk_chars: int = Field(default=12, ge=1)
    title_en_words: int = Field(default=6, ge=1)
    text_cjk_chars: int = Field(default=30, ge=1)
    text_en_words: int = Field(default=20, ge=1)
    prefer_cjk_when_mixed: bool = True


class EnforcementMetrics(BaseModel):
    """Length accounting before/after enforcement, for audit and telemetry."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    locale: str
    mode: LengthMode
    title_limit: int = Field(ge=1)
    text_limit: int = Field(ge=1)
    original_title_units: int = Field(ge=0)
    final_title_units: int = Field(ge=0)
    original_text_units: int = Field(ge=0)
    final_text_units: int = Field(ge=0)


class EnforcementOutcome(BaseModel):
    """Result of running the enforcer over a ``PromptGenerationResult``."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    result: PromptGenerationResult
    actions: list[EnforcementAction] = Field(default_factory=list)
    metrics: EnforcementMetrics
    enforced: bool = False
    fallback_to_app_surface: bool = False


class PromptResultEnforcer:
    """Apply hard length caps to glasses-bound prompt results.

    Usage::

        enforcer = PromptResultEnforcer()
        outcome = enforcer.enforce(result, prd_surface="glasses_popup", locale="zh-CN")
        final_result = outcome.result   # ready to display
        actions = outcome.actions       # for telemetry / decision metadata
    """

    def __init__(self, limits: GlassesLengthLimits | None = None) -> None:
        self._limits = limits or GlassesLengthLimits()

    @property
    def limits(self) -> GlassesLengthLimits:
        return self._limits

    def enforce(
        self,
        result: PromptGenerationResult,
        *,
        prd_surface: PRDSurface | str,
        locale: str = "zh-CN",
    ) -> EnforcementOutcome:
        title = result.glasses_title
        text = result.glasses_text
        mode = self._resolve_mode(locale, title + " " + text)
        title_limit = self._title_limit(mode)
        text_limit = self._text_limit(mode)
        original_title_units = count_units(title, mode)
        original_text_units = count_units(text, mode)

        # Early returns: nothing to enforce.
        if not result.should_prompt:
            return _passthrough_outcome(
                result,
                action=EnforcementAction.NOT_PROMPTING,
                locale=locale,
                mode=mode,
                title_limit=title_limit,
                text_limit=text_limit,
                original_title_units=original_title_units,
                original_text_units=original_text_units,
            )
        if not _is_glasses_surface(prd_surface):
            return _passthrough_outcome(
                result,
                action=EnforcementAction.NOT_GLASSES_SURFACE,
                locale=locale,
                mode=mode,
                title_limit=title_limit,
                text_limit=text_limit,
                original_title_units=original_title_units,
                original_text_units=original_text_units,
            )

        actions: list[EnforcementAction] = []
        new_title = title
        new_text = text

        # Title cap. Title must remain non-empty (contracts require it when
        # should_prompt=true), so on degenerate edge cases we keep at least
        # one visible character.
        if original_title_units > title_limit:
            candidate_title = truncate_to_limit(title, title_limit, mode)
            if not _has_visible_content(candidate_title):
                candidate_title = _safe_minimum_title(title)
            new_title = candidate_title
            actions.append(EnforcementAction.TITLE_TRUNCATED)

        # Text cap.
        if original_text_units > text_limit:
            new_text = truncate_to_limit(text, text_limit, mode)
            actions.append(EnforcementAction.TEXT_TRUNCATED)

        # Granularity downgrade: if text became empty/unusable and current
        # granularity requires text, drop to icon+title only.
        new_granularity = ContentGranularity(result.content_granularity)
        if new_granularity >= ContentGranularity.ONE_LINE_ANSWER and not _has_visible_content(new_text):
            new_granularity = ContentGranularity.ICON_TITLE_ONLY
            new_text = ""
            actions.append(EnforcementAction.GRANULARITY_DOWNGRADED)

        # Preserve original verbose glasses text into app_detail_text when we
        # truncated it, so the companion app surface still has full content.
        new_detail = result.app_detail_text
        if EnforcementAction.TEXT_TRUNCATED in actions and not new_detail.strip():
            new_detail = text
            actions.append(EnforcementAction.DETAIL_FALLBACK_FILLED)

        # Hint downstream (rate limiter / orchestrator) that if even the
        # downgraded icon-only card is undesirable, the prompt should be
        # rerouted to an app surface. We don't reroute here; we signal.
        fallback_to_app = (
            EnforcementAction.GRANULARITY_DOWNGRADED in actions
            and new_granularity == ContentGranularity.ICON_TITLE_ONLY
            and not _has_visible_content(new_title)
        )

        new_safety_flags = list(result.safety_flags)
        for action in actions:
            tag = f"enforcer:{action.value}"
            if tag not in new_safety_flags:
                new_safety_flags.append(tag)

        if not actions:
            return EnforcementOutcome(
                result=result,
                actions=[EnforcementAction.NO_OP],
                metrics=EnforcementMetrics(
                    locale=locale,
                    mode=mode,
                    title_limit=title_limit,
                    text_limit=text_limit,
                    original_title_units=original_title_units,
                    final_title_units=original_title_units,
                    original_text_units=original_text_units,
                    final_text_units=original_text_units,
                ),
                enforced=False,
                fallback_to_app_surface=False,
            )

        # Use model_validate to re-run validators and surface contract drift
        # immediately (e.g. if any future change leaves title empty).
        updated_payload = result.model_dump(mode="python")
        updated_payload.update(
            {
                "glasses_title": new_title,
                "glasses_text": new_text,
                "content_granularity": new_granularity,
                "app_detail_text": new_detail,
                "safety_flags": new_safety_flags,
            }
        )
        updated = PromptGenerationResult.model_validate(updated_payload)
        return EnforcementOutcome(
            result=updated,
            actions=actions,
            metrics=EnforcementMetrics(
                locale=locale,
                mode=mode,
                title_limit=title_limit,
                text_limit=text_limit,
                original_title_units=original_title_units,
                final_title_units=count_units(new_title, mode),
                original_text_units=original_text_units,
                final_text_units=count_units(new_text, mode),
            ),
            enforced=True,
            fallback_to_app_surface=fallback_to_app,
        )

    def _resolve_mode(self, locale: str, sample_text: str) -> LengthMode:
        locale_lower = locale.lower()
        if locale_lower.startswith("en"):
            if self._limits.prefer_cjk_when_mixed and _CJK_RE.search(sample_text):
                return LengthMode.CJK_CHARS
            return LengthMode.EN_WORDS
        return LengthMode.CJK_CHARS

    def _title_limit(self, mode: LengthMode) -> int:
        return (
            self._limits.title_cjk_chars
            if mode == LengthMode.CJK_CHARS
            else self._limits.title_en_words
        )

    def _text_limit(self, mode: LengthMode) -> int:
        return (
            self._limits.text_cjk_chars
            if mode == LengthMode.CJK_CHARS
            else self._limits.text_en_words
        )


def count_units(text: str, mode: LengthMode | str) -> int:
    """Count length units according to ``mode``.

    CJK_CHARS counts CJK characters and ASCII alphanumerics individually, and
    excludes whitespace and punctuation. EN_WORDS counts whitespace-delimited
    tokens.
    """

    if not text:
        return 0
    resolved = LengthMode(mode)
    if resolved == LengthMode.EN_WORDS:
        return len([token for token in _WHITESPACE_RE.split(text.strip()) if token])
    count = 0
    for ch in text:
        if ch.isspace() or ch in _PUNCTUATION:
            continue
        count += 1
    return count


def truncate_to_limit(text: str, limit: int, mode: LengthMode | str) -> str:
    """Truncate ``text`` to fit within ``limit`` units in ``mode``.

    Cascade:
      1. Try a cut at the last sentence-ending punctuation within the limit.
      2. Try a cut at the last clause-ending punctuation within the limit.
      3. Hard cut with an ellipsis.

    Returns the original text unchanged if it already fits.
    """

    resolved = LengthMode(mode)
    if not text or count_units(text, resolved) <= limit:
        return text

    sentence_cut = _truncate_at_boundary(text, limit, resolved, _SENTENCE_END)
    if _has_visible_content(sentence_cut):
        return sentence_cut
    clause_cut = _truncate_at_boundary(text, limit, resolved, _CLAUSE_END)
    if _has_visible_content(clause_cut):
        return clause_cut
    return _hard_truncate(text, limit, resolved)


def _truncate_at_boundary(text: str, limit: int, mode: LengthMode, boundary_chars: str) -> str:
    """Return the longest prefix of ``text`` that ends at one of ``boundary_chars``
    and counts at most ``limit`` units, or '' when no such prefix exists."""

    if mode == LengthMode.EN_WORDS:
        tokens = [token for token in _WHITESPACE_RE.split(text.strip()) if token]
        last_boundary_idx = -1
        for idx, token in enumerate(tokens):
            if idx + 1 > limit:
                break
            if token and token[-1] in boundary_chars:
                last_boundary_idx = idx
        if last_boundary_idx < 0:
            return ""
        return " ".join(tokens[: last_boundary_idx + 1])

    # CJK_CHARS mode: walk characters, count visible units, remember the last
    # position of a boundary char while still under the unit budget.
    visible_count = 0
    last_boundary_pos = -1
    for pos, ch in enumerate(text):
        if ch.isspace() or ch in _PUNCTUATION:
            if ch in boundary_chars and visible_count > 0:
                last_boundary_pos = pos
            continue
        if visible_count + 1 > limit:
            break
        visible_count += 1
        if ch in boundary_chars:
            last_boundary_pos = pos
    if last_boundary_pos < 0:
        return ""
    return text[: last_boundary_pos + 1].rstrip()


def _hard_truncate(text: str, limit: int, mode: LengthMode) -> str:
    """Drop characters/words until under ``limit``, appending an ellipsis."""

    if mode == LengthMode.EN_WORDS:
        tokens = [token for token in _WHITESPACE_RE.split(text.strip()) if token]
        if limit <= 1:
            return tokens[0][:1] + "…" if tokens else "…"
        kept = tokens[: max(0, limit - 1)]
        if not kept:
            return "…"
        return " ".join(kept) + "…"

    target = max(1, limit - 1)
    kept_chars: list[str] = []
    visible_count = 0
    for ch in text:
        if ch.isspace() or ch in _PUNCTUATION:
            if visible_count == 0:
                continue
            kept_chars.append(ch)
            continue
        if visible_count + 1 > target:
            break
        kept_chars.append(ch)
        visible_count += 1
    out = "".join(kept_chars).rstrip()
    return out + "…" if out else "…"


def _has_visible_content(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        if ch.isspace() or ch in _PUNCTUATION or ch == "…":
            continue
        return True
    return False


def _safe_minimum_title(original: str) -> str:
    """Fallback title for the rare case where truncation drops every visible char.

    Returns the first visible run from ``original`` (up to 4 chars), or a
    static placeholder if even that is impossible. The placeholder satisfies
    the contract that ``glasses_title`` must be non-empty when
    ``should_prompt=true``.
    """

    visible: list[str] = []
    for ch in original:
        if ch.isspace() or ch in _PUNCTUATION:
            continue
        visible.append(ch)
        if len(visible) >= 4:
            break
    if visible:
        return "".join(visible)
    return "提示"


def _is_glasses_surface(prd_surface: PRDSurface | str) -> bool:
    value = prd_surface.value if hasattr(prd_surface, "value") else str(prd_surface)
    return value in GLASSES_SURFACES


def _passthrough_outcome(
    result: PromptGenerationResult,
    *,
    action: EnforcementAction,
    locale: str,
    mode: LengthMode,
    title_limit: int,
    text_limit: int,
    original_title_units: int,
    original_text_units: int,
) -> EnforcementOutcome:
    return EnforcementOutcome(
        result=result,
        actions=[action],
        metrics=EnforcementMetrics(
            locale=locale,
            mode=mode,
            title_limit=title_limit,
            text_limit=text_limit,
            original_title_units=original_title_units,
            final_title_units=original_title_units,
            original_text_units=original_text_units,
            final_text_units=original_text_units,
        ),
        enforced=False,
        fallback_to_app_surface=False,
    )


def enforcement_metadata(outcome: EnforcementOutcome) -> dict[str, Any]:
    """Compact serializable view of an enforcement outcome.

    Suitable for embedding in ``PromptCandidate.metadata`` so the runtime
    ledger and post-hoc analytics can audit what the enforcer did.
    """

    return {
        "enforcement_actions": [
            action if isinstance(action, str) else action.value for action in outcome.actions
        ],
        "enforcement_enforced": outcome.enforced,
        "enforcement_fallback_to_app_surface": outcome.fallback_to_app_surface,
        "enforcement_metrics": outcome.metrics.model_dump(mode="json"),
    }
