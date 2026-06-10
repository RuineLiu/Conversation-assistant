from __future__ import annotations

from dataclasses import dataclass
import re

from proactive_assistant.detection.contracts import CandidateTimingAction, DetectionRuleMatch, PromptPriority
from proactive_assistant.prompting import ContentGranularity, PrivacyLevel, PromptCategory
from proactive_assistant.schemas.scenario import ActivityPhase
from proactive_assistant.sessions import TranscriptSegmentRecord


@dataclass(frozen=True)
class RuleCandidate:
    rule_name: str
    prompt_category: PromptCategory
    activity_phase: ActivityPhase
    candidate_timing_action: CandidateTimingAction
    suggested_content_granularity: ContentGranularity
    priority: PromptPriority
    confidence: float
    matched_terms: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class PrivacyAssessment:
    privacy_level: PrivacyLevel
    privacy_risk: float
    matched_terms: tuple[str, ...]
    safety_flags: tuple[str, ...]


QUESTION_TERMS = (
    "?",
    "？",
    "谁",
    "什么",
    "为什么",
    "怎么",
    "如何",
    "多少",
    "多久",
    "哪个",
    "哪位",
    "什么时候",
    "是否",
    "能不能",
    "可不可以",
    "who",
    "what",
    "why",
    "how",
    "when",
    "whether",
    "can we",
    "should we",
)

FACT_TERMS = (
    "谁是",
    "哪一年",
    "什么时候",
    "多久",
    "多少",
    "金额",
    "预算",
    "上次",
    "之前",
    "上周会议",
    "还记得",
    "我们之前定",
    "有没有记录",
    "帮我回忆",
    "founded",
    "who is",
    "when did",
    "remember",
    "last meeting",
)

SUGGESTION_TERMS = (
    "建议",
    "怎么推进",
    "下一步",
    "怎么做",
    "怎么办",
    "如何处理",
    "方案",
    "选项",
    "优先级",
    "风险",
    "阻碍",
    "卡住",
    "依赖",
    "延期",
    "冲突",
    "没有对齐",
    "分歧",
    "怎么跟客户说",
    "怎么和领导沟通",
    "要不要提醒",
    "需要同步谁",
    "suggest",
    "next step",
    "risk",
    "option",
    "how should",
)

CONCEPT_TERMS = (
    "是什么意思",
    "什么意思",
    "怎么理解",
    "如何理解",
    "解释一下",
    "解释下",
    "不熟悉",
    "听不懂",
    "陌生词",
    "专业术语",
    "术语",
    "概念",
    "定义",
    "缩写",
    "abbreviation",
    "acronym",
    "what does",
    "what is",
    "means",
    "meaning",
    "explain",
    "definition",
)

GAP_TERMS = (
    "谁负责",
    "负责人",
    "owner",
    "谁跟进",
    "谁来做",
    "assign",
    "dri",
    "截止",
    "deadline",
    "due date",
    "时间点",
    "排期",
    "action item",
    "todo",
    "待办",
    "跟进",
    "同步",
    "确认",
    "落地",
    "还没定",
    "没有结论",
    "不确定",
    "待确认",
    "pending",
    "需要确认",
    "blocker",
    "gap",
    "没对齐",
    "遗漏",
    "漏掉",
)

END_TERMS = (
    "今天先到这",
    "总结一下",
    "会议结束",
    "最后确认",
    "recap",
    "wrap up",
)

PREP_TERMS = (
    "分钟后开会",
    "准备材料",
    "agenda",
    "会前",
)

HIGH_PRIVACY_TERMS = (
    "客户数据",
    "报价",
    "金额",
    "薪资",
    "裁员",
    "法务",
    "合规",
    "机密",
    "confidential",
    "salary",
    "legal",
    "compliance",
    "customer data",
)

MEDIUM_PRIVACY_TERMS = (
    "客户",
    "合同",
    "营收",
    "利润",
    "隐私",
    "内部",
    "竞品",
    "收购",
    "投资",
    "customer",
    "contract",
    "pricing",
    "revenue",
    "internal",
)


def detect_candidates(segment: TranscriptSegmentRecord) -> list[RuleCandidate]:
    text = _normalize(segment.text)
    candidates: list[RuleCandidate] = []
    candidates.extend(_gap_candidates(text))
    candidates.extend(_suggestion_candidates(text))
    candidates.extend(_fact_candidates(text))
    candidates.extend(_concept_candidates(text))
    candidates.extend(_question_candidates(text))
    return candidates


def assess_privacy(text: str, privacy_constraints: list[str]) -> PrivacyAssessment:
    normalized = _normalize(" ".join([text, *privacy_constraints]))
    high = _matched_terms(normalized, HIGH_PRIVACY_TERMS)
    if high:
        return PrivacyAssessment(
            privacy_level=PrivacyLevel.HIGH,
            privacy_risk=0.82,
            matched_terms=tuple(high),
            safety_flags=("sensitive_business_context",),
        )
    medium = _matched_terms(normalized, MEDIUM_PRIVACY_TERMS)
    if medium:
        return PrivacyAssessment(
            privacy_level=PrivacyLevel.MEDIUM,
            privacy_risk=0.52,
            matched_terms=tuple(medium),
            safety_flags=("business_context",),
        )
    return PrivacyAssessment(
        privacy_level=PrivacyLevel.LOW,
        privacy_risk=0.12,
        matched_terms=(),
        safety_flags=(),
    )


def to_rule_match(candidate: RuleCandidate) -> DetectionRuleMatch:
    return DetectionRuleMatch(
        rule_name=candidate.rule_name,
        matched_terms=list(candidate.matched_terms),
        confidence_delta=0.0,
        reason=candidate.reason,
    )


def category_rank(category: PromptCategory) -> int:
    ranks = {
        PromptCategory.SUMMARY_GAP_CHECK: 0,
        PromptCategory.SUGGESTION: 1,
        PromptCategory.PERSON_OR_FACT: 2,
        PromptCategory.CONCEPT_EXPLANATION: 3,
        PromptCategory.QUESTION_ANSWER: 4,
    }
    return ranks[category]


def priority_rank(priority: PromptPriority) -> int:
    return {PromptPriority.P0: 0, PromptPriority.P1: 1, PromptPriority.P2: 2}[priority]


def downgrade_for_privacy(candidate: RuleCandidate, privacy: PrivacyAssessment) -> RuleCandidate:
    if privacy.privacy_level != PrivacyLevel.HIGH:
        return candidate
    return RuleCandidate(
        rule_name=candidate.rule_name,
        prompt_category=candidate.prompt_category,
        activity_phase=candidate.activity_phase,
        candidate_timing_action=candidate.candidate_timing_action,
        suggested_content_granularity=min(
            candidate.suggested_content_granularity,
            ContentGranularity.ONE_LINE_ANSWER,
        ),
        priority=PromptPriority.P2,
        confidence=max(0.0, candidate.confidence - 0.12),
        matched_terms=candidate.matched_terms,
        reason=f"{candidate.reason}; downgraded because privacy risk is high",
    )


def _gap_candidates(text: str) -> list[RuleCandidate]:
    terms = _matched_terms(text, GAP_TERMS)
    if not terms:
        return []
    phase = ActivityPhase.IN_ACTIVITY
    timing = CandidateTimingAction.DURING_ACTIVITY
    granularity = ContentGranularity.ONE_LINE_ANSWER
    priority = PromptPriority.P0 if _matched_terms(text, ("谁负责", "owner", "deadline", "截止", "action item")) else PromptPriority.P1
    end_terms = _matched_terms(text, END_TERMS)
    prep_terms = _matched_terms(text, PREP_TERMS)
    if end_terms:
        phase = ActivityPhase.POST_ACTIVITY
        timing = CandidateTimingAction.AFTER_ACTIVITY
        granularity = ContentGranularity.CONCISE_BULLETS
        priority = PromptPriority.P0
        terms.extend(end_terms)
    elif prep_terms:
        phase = ActivityPhase.PRE_ACTIVITY
        timing = CandidateTimingAction.BEFORE_ACTIVITY
        granularity = ContentGranularity.CONCISE_BULLETS
        terms.extend(prep_terms)
    return [
        RuleCandidate(
            rule_name="gap_check_rule",
            prompt_category=PromptCategory.SUMMARY_GAP_CHECK,
            activity_phase=phase,
            candidate_timing_action=timing,
            suggested_content_granularity=granularity,
            priority=priority,
            confidence=0.84 if priority == PromptPriority.P0 else 0.74,
            matched_terms=tuple(_dedupe(terms)),
            reason="Detected owner, deadline, action item, or unresolved meeting gap.",
        )
    ]


def _suggestion_candidates(text: str) -> list[RuleCandidate]:
    terms = _matched_terms(text, SUGGESTION_TERMS)
    if not terms:
        return []
    return [
        RuleCandidate(
            rule_name="suggestion_rule",
            prompt_category=PromptCategory.SUGGESTION,
            activity_phase=ActivityPhase.IN_ACTIVITY,
            candidate_timing_action=CandidateTimingAction.DURING_ACTIVITY,
            suggested_content_granularity=ContentGranularity.CONCISE_BULLETS,
            priority=PromptPriority.P1,
            confidence=0.72,
            matched_terms=tuple(terms),
            reason="Detected request for next step, risk handling, or communication suggestion.",
        )
    ]


def _fact_candidates(text: str) -> list[RuleCandidate]:
    terms = _matched_terms(text, FACT_TERMS)
    if not terms:
        return []
    granularity = ContentGranularity.CONCISE_BULLETS if _matched_terms(text, ("上次", "之前", "last meeting", "remember")) else ContentGranularity.ONE_LINE_ANSWER
    return [
        RuleCandidate(
            rule_name="fact_recall_rule",
            prompt_category=PromptCategory.PERSON_OR_FACT,
            activity_phase=ActivityPhase.IN_ACTIVITY,
            candidate_timing_action=CandidateTimingAction.DURING_ACTIVITY,
            suggested_content_granularity=granularity,
            priority=PromptPriority.P1,
            confidence=0.7,
            matched_terms=tuple(terms),
            reason="Detected person, fact, date, number, or prior-context recall need.",
        )
    ]


def _concept_candidates(text: str) -> list[RuleCandidate]:
    terms = _matched_terms(text, CONCEPT_TERMS)
    if not terms and not _looks_like_term_explanation_need(text):
        return []
    return [
        RuleCandidate(
            rule_name="concept_explanation_rule",
            prompt_category=PromptCategory.CONCEPT_EXPLANATION,
            activity_phase=ActivityPhase.IN_ACTIVITY,
            candidate_timing_action=CandidateTimingAction.DURING_ACTIVITY,
            suggested_content_granularity=ContentGranularity.ONE_LINE_ANSWER,
            priority=PromptPriority.P1,
            confidence=0.72 if terms else 0.66,
            matched_terms=tuple(terms or ["term_explanation_pattern"]),
            reason="Detected unfamiliar term, concept, acronym, or definition explanation need.",
        )
    ]


def _question_candidates(text: str) -> list[RuleCandidate]:
    terms = _matched_terms(text, QUESTION_TERMS)
    if not terms:
        return []
    return [
        RuleCandidate(
            rule_name="question_rule",
            prompt_category=PromptCategory.QUESTION_ANSWER,
            activity_phase=ActivityPhase.IN_ACTIVITY,
            candidate_timing_action=CandidateTimingAction.DURING_ACTIVITY,
            suggested_content_granularity=ContentGranularity.ONE_LINE_ANSWER,
            priority=PromptPriority.P1,
            confidence=0.62 if terms in (["?"], ["？"]) else 0.68,
            matched_terms=tuple(terms),
            reason="Detected an explicit question in the transcript window.",
        )
    ]


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if _normalize(term) in text]


def _normalize(text: str) -> str:
    return text.strip().lower()


def _looks_like_term_explanation_need(text: str) -> bool:
    return bool(re.search(r"[a-z][a-z0-9+\-_/]{2,}\s*(是|指|代表|怎么|如何|what|mean)", text, re.IGNORECASE))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
