from proactive_assistant.prompting import (
    ContentGranularity,
    EnforcementAction,
    GlassesLengthLimits,
    LengthMode,
    PRDSurface,
    PrivacyLevel,
    PromptCategory,
    PromptGenerationResult,
    PromptResultEnforcer,
    count_units,
    enforcement_metadata,
    truncate_to_limit,
)


def make_result(
    *,
    title: str = "负责人待确认",
    text: str = "这个风险还没有明确 owner 和截止时间。",
    detail: str = "",
    category: PromptCategory = PromptCategory.SUMMARY_GAP_CHECK,
    granularity: ContentGranularity = ContentGranularity.ONE_LINE_ANSWER,
    should_prompt: bool = True,
) -> PromptGenerationResult:
    return PromptGenerationResult(
        should_prompt=should_prompt,
        prompt_category=category if should_prompt else None,
        content_granularity=granularity,
        glasses_title=title,
        glasses_text=text,
        app_detail_text=detail,
        source_refs=["transcript:seg_0"] if should_prompt else [],
        confidence=0.82,
        privacy_level=PrivacyLevel.LOW,
        privacy_risk=0.1,
        rationale="test",
        safety_flags=[],
    )


def test_count_units_cjk_chars_excludes_whitespace_and_punctuation() -> None:
    assert count_units("负责人待确认", LengthMode.CJK_CHARS) == 6
    # 张三负责下周五截止 = 9 visible chars; punctuation/space excluded
    assert count_units("张三 负责，下周五 截止。", LengthMode.CJK_CHARS) == 9
    assert count_units("ALL CAPS API", LengthMode.CJK_CHARS) == 10


def test_count_units_en_words_splits_on_whitespace() -> None:
    assert count_units("Quarterly review pending owner.", LengthMode.EN_WORDS) == 4
    assert count_units("  many   spaces  here  ", LengthMode.EN_WORDS) == 3
    assert count_units("", LengthMode.EN_WORDS) == 0


def test_truncate_to_limit_keeps_text_when_within_budget() -> None:
    text = "下周五前确认负责人。"
    assert truncate_to_limit(text, 30, LengthMode.CJK_CHARS) == text


def test_truncate_to_limit_prefers_sentence_boundary_for_cjk() -> None:
    text = "请尽快确认负责人。我们还要补齐截止时间和下一步动作。"
    truncated = truncate_to_limit(text, 12, LengthMode.CJK_CHARS)
    assert truncated.endswith("。")
    assert count_units(truncated, LengthMode.CJK_CHARS) <= 12
    assert truncated == "请尽快确认负责人。"


def test_truncate_to_limit_falls_back_to_clause_boundary_for_cjk() -> None:
    text = "确认负责人，补齐截止时间，记录阻塞项，安排下一步"
    truncated = truncate_to_limit(text, 10, LengthMode.CJK_CHARS)
    assert truncated.endswith("，")
    assert count_units(truncated, LengthMode.CJK_CHARS) <= 10


def test_truncate_to_limit_hard_cut_with_ellipsis_for_cjk() -> None:
    text = "负责人和截止时间需要立刻明确而且阻塞项要记录清楚"
    truncated = truncate_to_limit(text, 8, LengthMode.CJK_CHARS)
    assert truncated.endswith("…")
    visible = count_units(truncated.replace("…", ""), LengthMode.CJK_CHARS)
    assert visible <= 8


def test_truncate_to_limit_en_words_prefers_punct_boundary() -> None:
    text = "Owner missing for launch risk. Need confirmation by Friday and next step."
    truncated = truncate_to_limit(text, 6, LengthMode.EN_WORDS)
    assert truncated == "Owner missing for launch risk."


def test_truncate_to_limit_en_words_hard_cut() -> None:
    text = "Quarterly review action item owner deadline confirmation needed urgently this week"
    truncated = truncate_to_limit(text, 4, LengthMode.EN_WORDS)
    assert truncated.endswith("…")
    assert len(truncated.split()) <= 4


def test_enforcer_no_op_when_within_limits() -> None:
    enforcer = PromptResultEnforcer()
    result = make_result(title="待明确", text="确认负责人和截止时间。")

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="zh-CN")

    assert outcome.enforced is False
    assert outcome.actions == [EnforcementAction.NO_OP]
    assert outcome.result is result
    assert outcome.metrics.final_text_units == outcome.metrics.original_text_units


def test_enforcer_passes_through_non_glasses_surface() -> None:
    enforcer = PromptResultEnforcer()
    long_text = "这是一段非常非常长的会后总结说明，超过了眼镜端的硬上限，但 app summary 不需要被截断。"
    result = make_result(text=long_text, granularity=ContentGranularity.CONCISE_BULLETS)

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.APP_SUMMARY_TAB, locale="zh-CN")

    assert outcome.enforced is False
    assert outcome.actions == [EnforcementAction.NOT_GLASSES_SURFACE]
    assert outcome.result.glasses_text == long_text


def test_enforcer_passes_through_when_not_prompting() -> None:
    enforcer = PromptResultEnforcer()
    result = PromptGenerationResult(
        should_prompt=False,
        prompt_category=None,
        content_granularity=ContentGranularity.NO_ACTION,
        glasses_title="",
        glasses_text="",
        app_detail_text="",
        source_refs=[],
        confidence=0.0,
        privacy_level=PrivacyLevel.LOW,
        privacy_risk=0.0,
        rationale="suppressed by policy",
        safety_flags=[],
    )

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="zh-CN")

    assert outcome.actions == [EnforcementAction.NOT_PROMPTING]
    assert outcome.enforced is False


def test_enforcer_truncates_overlong_chinese_text_and_fills_detail() -> None:
    enforcer = PromptResultEnforcer()
    long_text = (
        "这个风险的负责人需要立刻确认，下周五之前必须给出明确的下一步动作，"
        "否则会影响发布节奏，建议同步上级和相关方一起决定。"
    )
    result = make_result(text=long_text)

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="zh-CN")

    assert outcome.enforced is True
    assert EnforcementAction.TEXT_TRUNCATED in outcome.actions
    assert EnforcementAction.DETAIL_FALLBACK_FILLED in outcome.actions
    assert count_units(outcome.result.glasses_text, LengthMode.CJK_CHARS) <= 30
    # original verbose content preserved in detail
    assert outcome.result.app_detail_text == long_text
    # enforcer leaves a breadcrumb in safety_flags
    assert "enforcer:text_truncated" in outcome.result.safety_flags


def test_enforcer_truncates_overlong_title() -> None:
    enforcer = PromptResultEnforcer()
    long_title = "这个待明确的负责人和截止时间事项"
    result = make_result(title=long_title, text="确认负责人。")

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="zh-CN")

    assert EnforcementAction.TITLE_TRUNCATED in outcome.actions
    assert count_units(outcome.result.glasses_title, LengthMode.CJK_CHARS) <= 12
    assert outcome.result.glasses_title  # never empty


def test_enforcer_downgrades_granularity_when_text_becomes_unusable() -> None:
    enforcer = PromptResultEnforcer(
        limits=GlassesLengthLimits(text_cjk_chars=2, title_cjk_chars=12),
    )
    result = make_result(
        title="待明确",
        text="负责人和截止时间都还没有定下来",
        granularity=ContentGranularity.ONE_LINE_ANSWER,
    )

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="zh-CN")

    if outcome.result.glasses_text == "":
        assert EnforcementAction.GRANULARITY_DOWNGRADED in outcome.actions
        assert outcome.result.content_granularity == ContentGranularity.ICON_TITLE_ONLY
    else:
        assert EnforcementAction.TEXT_TRUNCATED in outcome.actions
        assert count_units(outcome.result.glasses_text, LengthMode.CJK_CHARS) <= 2


def test_enforcer_uses_en_words_for_english_locale() -> None:
    enforcer = PromptResultEnforcer()
    long_en_text = (
        "Owner has not been confirmed for the launch risk follow-up. "
        "Deadline is still open and the next step is pending stakeholder sync "
        "later this week before the release branch cut, please double check."
    )
    result = make_result(
        title="Owner pending",
        text=long_en_text,
    )

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="en-US")

    assert outcome.metrics.mode == LengthMode.EN_WORDS
    assert EnforcementAction.TEXT_TRUNCATED in outcome.actions
    assert count_units(outcome.result.glasses_text, LengthMode.EN_WORDS) <= 20


def test_enforcer_prefers_cjk_when_locale_en_but_text_cjk() -> None:
    enforcer = PromptResultEnforcer()
    cjk_title = "待明确"
    cjk_text = "这个风险还没有明确的负责人和截止时间，建议立刻同步。"
    result = make_result(title=cjk_title, text=cjk_text)

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="en-US")

    assert outcome.metrics.mode == LengthMode.CJK_CHARS


def test_enforcement_metadata_round_trips_actions_and_metrics() -> None:
    enforcer = PromptResultEnforcer()
    long_text = "这个风险的负责人需要立刻确认，下周五之前必须给出明确的下一步动作。"
    result = make_result(text=long_text)

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="zh-CN")
    payload = enforcement_metadata(outcome)

    assert "enforcement_actions" in payload
    assert payload["enforcement_enforced"] is True
    assert payload["enforcement_metrics"]["text_limit"] == 30
    assert payload["enforcement_metrics"]["mode"] == LengthMode.CJK_CHARS.value


def test_enforcer_keeps_title_nonempty_when_truncation_would_drop_everything() -> None:
    enforcer = PromptResultEnforcer(
        limits=GlassesLengthLimits(title_cjk_chars=1, text_cjk_chars=30),
    )
    result = make_result(title="待明确事项", text="确认负责人。")

    outcome = enforcer.enforce(result, prd_surface=PRDSurface.GLASSES_POPUP, locale="zh-CN")

    assert outcome.result.glasses_title
    assert outcome.result.should_prompt is True
