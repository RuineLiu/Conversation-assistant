from __future__ import annotations

import re
from hashlib import sha1

from proactive_assistant.meeting_state.contracts import (
    ActionItem,
    ActionItemStatus,
    Decision,
    MentionedRef,
    MentionedRefType,
    MeetingGap,
    MeetingGapPriority,
    MeetingGapType,
    MeetingUtterance,
    OpenQuestion,
    QuestionType,
    Risk,
)
from proactive_assistant.sessions import TranscriptSegmentRecord


QUESTION_TERMS = (
    "?",
    "？",
    "谁",
    "什么",
    "为什么",
    "怎么",
    "如何",
    "是否",
    "能不能",
    "是不是",
    "哪一年",
    "什么时候",
    "多少",
)

ACTION_TERMS = (
    "action item",
    "todo",
    "待办",
    "下一步",
    "跟进",
    "推进",
    "负责",
    "owner",
    "负责人",
    "deadline",
    "截止",
    "定一下",
    "确认一下",
)

ACTION_STRONG_TERMS = (
    "action item",
    "todo",
    "待办",
    "下一步",
    "跟进",
    "负责",
    "owner",
    "负责人",
    "deadline",
    "截止",
    "定一下",
    "确认一下",
    "谁负责",
    "谁跟进",
    "谁来做",
    "请",
    "由",
    "让",
)

DECISION_TERMS = (
    "决定",
    "结论",
    "定了",
    "定下来",
    "拍板",
    "对齐",
    "最终方案",
    "就按",
)

RISK_TERMS = (
    "风险",
    "阻塞",
    "blocker",
    "不确定",
    "担心",
    "延期",
    "延迟",
    "overdue",
)

REFERENCE_TERMS = (
    "上次",
    "之前",
    "上周",
    "上回",
    "刚才",
    "前面",
    "历史",
    "承诺",
    "说过",
    "定过",
)

END_TERMS = ("总结", "结束前", "会后", "收尾", "最后", "还有哪些", "还有什么")


def utterance_from_segment(segment: TranscriptSegmentRecord) -> MeetingUtterance:
    return MeetingUtterance(
        utterance_id=segment.segment_id,
        session_id=segment.session_id,
        speaker=segment.speaker,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        text=segment.text.strip(),
        topic=segment.topic,
        asr_confidence=segment.asr_confidence,
        metadata=segment.metadata,
    )


def extract_question(utterance: MeetingUtterance) -> OpenQuestion | None:
    text = normalize_text(utterance.text)
    if not looks_like_question(text):
        return None
    return OpenQuestion(
        question_id=_id("question", utterance.session_id, utterance.utterance_id, text),
        text=utterance.text,
        asker=utterance.speaker,
        source_utterance_id=utterance.utterance_id,
        ts_ms=utterance.start_ms,
        question_type=classify_question(text),
        metadata={"rule": "question_terms"},
    )


def extract_action_item(utterance: MeetingUtterance) -> ActionItem | None:
    text = normalize_text(utterance.text)
    if not any(term in text.lower() for term in ACTION_TERMS):
        return None
    if not has_action_signal(text):
        return None
    owner = extract_owner(text)
    deadline = extract_deadline(text)
    next_step = extract_next_step(text)
    status = ActionItemStatus.ASSIGNED if owner else ActionItemStatus.OPEN
    return ActionItem(
        action_item_id=_id("action", utterance.session_id, utterance.utterance_id, text),
        desc=utterance.text,
        source_utterance_id=utterance.utterance_id,
        source_ts_ms=utterance.start_ms,
        owner=owner,
        deadline=deadline,
        next_step=next_step,
        status=status,
        evidence=utterance.text,
        metadata={"rule": "action_terms"},
    )


def extract_decision(utterance: MeetingUtterance) -> Decision | None:
    text = normalize_text(utterance.text)
    if not any(term in text for term in DECISION_TERMS):
        return None
    conclusion = extract_decision_conclusion(text)
    return Decision(
        decision_id=_id("decision", utterance.session_id, utterance.utterance_id, text),
        topic=utterance.topic or compact_topic(text),
        source_utterance_id=utterance.utterance_id,
        ts_ms=utterance.start_ms,
        conclusion=conclusion,
        evidence=utterance.text,
        metadata={"rule": "decision_terms"},
    )


def extract_risk(utterance: MeetingUtterance) -> Risk | None:
    text = normalize_text(utterance.text)
    if not any(term in text.lower() for term in RISK_TERMS):
        return None
    return Risk(
        risk_id=_id("risk", utterance.session_id, utterance.utterance_id, text),
        desc=utterance.text,
        source_utterance_id=utterance.utterance_id,
        ts_ms=utterance.start_ms,
        closed=looks_like_risk_closed(text),
        evidence=utterance.text,
        metadata={"rule": "risk_terms"},
    )


def extract_refs(utterance: MeetingUtterance) -> list[MentionedRef]:
    text = normalize_text(utterance.text)
    refs: list[MentionedRef] = []
    if any(term in text for term in REFERENCE_TERMS):
        refs.append(_ref(utterance, MentionedRefType.PREVIOUS_MEETING, utterance.text, "reference_terms"))
    if "deadline" in text.lower() or "截止" in text:
        refs.append(_ref(utterance, MentionedRefType.DEADLINE, utterance.text, "deadline_terms"))
    if "owner" in text.lower() or "负责人" in text or "谁负责" in text or "跟进" in text or "承诺" in text:
        refs.append(_ref(utterance, MentionedRefType.COMMITMENT, utterance.text, "commitment_terms"))
    return dedupe_refs(refs)


def maybe_answer_questions(
    questions: list[OpenQuestion],
    utterance: MeetingUtterance,
    *,
    max_age_ms: int = 180_000,
) -> tuple[list[OpenQuestion], list[OpenQuestion]]:
    answered: list[OpenQuestion] = []
    updated: list[OpenQuestion] = []
    text = normalize_text(utterance.text)
    if looks_like_question(text):
        return questions, answered
    for question in questions:
        if question.answered:
            updated.append(question)
            continue
        if utterance.start_ms - question.ts_ms > max_age_ms:
            updated.append(question)
            continue
        if looks_like_answer_to_question(text, question):
            new_question = question.model_copy(
                update={
                    "answered": True,
                    "answer": utterance.text,
                    "answer_utterance_id": utterance.utterance_id,
                }
            )
            updated.append(new_question)
            answered.append(new_question)
        else:
            updated.append(question)
    return updated, answered


def scan_gaps(
    *,
    session_id: str,
    questions: list[OpenQuestion],
    action_items: list[ActionItem],
    decisions: list[Decision],
    risks: list[Risk],
    current_ms: int | None = None,
    scheduled_end_ms: int | None = None,
    force_end_summary: bool = False,
) -> list[MeetingGap]:
    gaps: list[MeetingGap] = []
    for question in questions:
        if not question.answered:
            gaps.append(
                MeetingGap(
                    gap_id=_id("gap", session_id, question.question_id, MeetingGapType.UNANSWERED_QUESTION.value),
                    session_id=session_id,
                    gap_type=MeetingGapType.UNANSWERED_QUESTION,
                    priority=MeetingGapPriority.P1,
                    object_type="open_question",
                    object_id=question.question_id,
                    text=question.text,
                    reason="Question has not been answered in the tracked meeting state.",
                    source_utterance_ids=[question.source_utterance_id],
                    first_seen_ms=question.ts_ms,
                )
            )
    for item in action_items:
        if item.owner is None:
            gaps.append(_action_gap(session_id, item, MeetingGapType.ACTION_MISSING_OWNER, "Action item has no owner."))
        if item.deadline is None:
            gaps.append(_action_gap(session_id, item, MeetingGapType.ACTION_MISSING_DEADLINE, "Action item has no deadline."))
        if item.next_step is None:
            gaps.append(_action_gap(session_id, item, MeetingGapType.ACTION_MISSING_NEXT_STEP, "Action item has no next step."))
    for decision in decisions:
        if decision.conclusion is None:
            gaps.append(
                MeetingGap(
                    gap_id=_id("gap", session_id, decision.decision_id, MeetingGapType.DECISION_MISSING_CONCLUSION.value),
                    session_id=session_id,
                    gap_type=MeetingGapType.DECISION_MISSING_CONCLUSION,
                    priority=MeetingGapPriority.P1,
                    object_type="decision",
                    object_id=decision.decision_id,
                    text=decision.topic,
                    reason="Decision topic was mentioned but no conclusion is tracked.",
                    source_utterance_ids=[decision.source_utterance_id],
                    first_seen_ms=decision.ts_ms,
                )
            )
    for risk in risks:
        if not risk.closed:
            gaps.append(
                MeetingGap(
                    gap_id=_id("gap", session_id, risk.risk_id, MeetingGapType.OPEN_RISK.value),
                    session_id=session_id,
                    gap_type=MeetingGapType.OPEN_RISK,
                    priority=MeetingGapPriority.P2,
                    object_type="risk",
                    object_id=risk.risk_id,
                    text=risk.desc,
                    reason="Risk is still open in the meeting state.",
                    source_utterance_ids=[risk.source_utterance_id],
                    first_seen_ms=risk.ts_ms,
                )
            )
    if should_emit_end_summary(gaps, current_ms=current_ms, scheduled_end_ms=scheduled_end_ms, force=force_end_summary):
        gaps.append(
            MeetingGap(
                gap_id=_id("gap", session_id, "end_summary", str(len(gaps))),
                session_id=session_id,
                gap_type=MeetingGapType.END_SUMMARY_NEEDED,
                priority=MeetingGapPriority.P0,
                object_type="meeting_state",
                object_id=session_id,
                text="Meeting has unresolved questions, actions, decisions, or risks.",
                reason="Meeting appears close to ending while tracked gaps remain.",
                source_utterance_ids=[],
                first_seen_ms=current_ms,
                metadata={"tracked_gap_count": len(gaps)},
            )
        )
    return dedupe_gaps(gaps)


def looks_like_end_signal(text: str) -> bool:
    normalized = normalize_text(text)
    return any(term in normalized for term in END_TERMS)


def looks_like_question(text: str) -> bool:
    normalized = normalize_text(text)
    return any(term in normalized for term in QUESTION_TERMS)


def has_action_signal(text: str) -> bool:
    normalized = normalize_text(text)
    lower = normalized.lower()
    return any(term in lower for term in ACTION_STRONG_TERMS)


def classify_question(text: str) -> QuestionType:
    normalized = normalize_text(text)
    if any(term in normalized for term in ("谁", "哪位", "负责人", "owner")):
        return QuestionType.PERSON
    if any(term in normalized for term in ("为什么", "原因", "为何")):
        return QuestionType.REASON
    if any(term in normalized for term in ("数据", "指标", "多少", "同比", "环比", "变化")):
        return QuestionType.DATA
    if any(term in normalized for term in ("什么意思", "是什么", "概念", "定义", "缩写")):
        return QuestionType.CONCEPT
    if any(term in normalized for term in ("哪一年", "什么时候", "成立", "日期")):
        return QuestionType.FACT
    return QuestionType.OTHER


def looks_like_answer_to_question(text: str, question: OpenQuestion) -> bool:
    normalized = normalize_text(text)
    qtype = QuestionType(question.question_type)
    if qtype == QuestionType.PERSON:
        return bool(extract_owner(normalized)) or any(term in normalized for term in ("我来", "他来", "她来", "我们来"))
    if qtype == QuestionType.REASON:
        return any(term in normalized for term in ("因为", "原因是", "主要是", "由于"))
    if qtype == QuestionType.DATA:
        return bool(re.search(r"\d", normalized)) or any(term in normalized for term in ("上涨", "下降", "变化", "口径"))
    if qtype == QuestionType.CONCEPT:
        return any(term in normalized for term in ("意思是", "指的是", "定义", "就是"))
    if qtype == QuestionType.FACT:
        return bool(re.search(r"\d{4}|\d+", normalized)) or any(term in normalized for term in ("成立", "时间是"))
    return any(term in normalized for term in ("是", "可以", "不能", "已经", "结论"))


def extract_owner(text: str) -> str | None:
    normalized = normalize_text(text)
    patterns = (
        r"(?:由|让|请)?(?P<owner>[A-Za-z\u4e00-\u9fff]{1,12})(?:来)?负责",
        r"负责人(?:是|:|：)?(?P<owner>[A-Za-z\u4e00-\u9fff]{1,12})",
        r"owner(?:是|:|：)?(?P<owner>[A-Za-z\u4e00-\u9fff]{1,12})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        owner = match.group("owner").strip()
        if owner and "谁" not in owner and owner not in {"哪位", "大家", "我们", "这个", "问题"}:
            return owner
    if "我来" in normalized:
        return "speaker_self"
    return None


def extract_deadline(text: str) -> str | None:
    normalized = normalize_text(text)
    patterns = (
        r"(?P<deadline>下周[一二三四五六日天])",
        r"(?P<deadline>本周[一二三四五六日天])",
        r"(?P<deadline>周[一二三四五六日天])",
        r"(?P<deadline>明天|今天|后天|月底|月末|本月底|下月底)",
        r"(?P<deadline>\d{1,2}月\d{1,2}[日号])",
        r"deadline(?:是|:|：|前)?(?P<deadline>[A-Za-z0-9\u4e00-\u9fff\-/]{1,16})",
        r"截止(?:到|在|是|:|：)?(?P<deadline>[A-Za-z0-9\u4e00-\u9fff\-/]{1,16})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group("deadline").strip()
    return None


def extract_next_step(text: str) -> str | None:
    normalized = normalize_text(text)
    if "下一步" in normalized:
        return text.strip()
    if any(term in normalized for term in ("跟进", "推进", "确认", "定一下", "安排", "同步")):
        return text.strip()
    return None


def extract_decision_conclusion(text: str) -> str | None:
    normalized = normalize_text(text)
    if any(term in normalized for term in ("未定", "待定", "还没定", "不能定")):
        return None
    if any(term in normalized for term in ("就按", "定了", "定下来", "决定", "结论是", "拍板")):
        return text.strip()
    return None


def compact_topic(text: str, *, max_chars: int = 42) -> str:
    normalized = normalize_text(text)
    return normalized if len(normalized) <= max_chars else normalized[: max_chars - 1] + "…"


def looks_like_risk_closed(text: str) -> bool:
    normalized = normalize_text(text)
    return any(term in normalized for term in ("风险解除", "已解决", "没有风险", "可控", "闭环"))


def should_emit_end_summary(
    gaps: list[MeetingGap],
    *,
    current_ms: int | None,
    scheduled_end_ms: int | None,
    force: bool,
) -> bool:
    if not gaps:
        return False
    if force:
        return True
    if current_ms is None or scheduled_end_ms is None:
        return False
    return scheduled_end_ms - current_ms <= 5 * 60 * 1000


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedupe_gaps(gaps: list[MeetingGap]) -> list[MeetingGap]:
    seen: set[str] = set()
    deduped: list[MeetingGap] = []
    for gap in gaps:
        if gap.gap_id in seen:
            continue
        seen.add(gap.gap_id)
        deduped.append(gap)
    return deduped


def dedupe_refs(refs: list[MentionedRef]) -> list[MentionedRef]:
    seen: set[tuple[str, str]] = set()
    deduped: list[MentionedRef] = []
    for ref in refs:
        key = (ref.ref_type, ref.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _action_gap(session_id: str, item: ActionItem, gap_type: MeetingGapType, reason: str) -> MeetingGap:
    priority = MeetingGapPriority.P0 if gap_type in {MeetingGapType.ACTION_MISSING_OWNER, MeetingGapType.ACTION_MISSING_DEADLINE} else MeetingGapPriority.P1
    return MeetingGap(
        gap_id=_id("gap", session_id, item.action_item_id, gap_type.value),
        session_id=session_id,
        gap_type=gap_type,
        priority=priority,
        object_type="action_item",
        object_id=item.action_item_id,
        text=item.desc,
        reason=reason,
        source_utterance_ids=[item.source_utterance_id],
        first_seen_ms=item.source_ts_ms,
    )


def _ref(utterance: MeetingUtterance, ref_type: MentionedRefType, text: str, rule: str) -> MentionedRef:
    return MentionedRef(
        mentioned_ref_id=_id("ref", utterance.session_id, utterance.utterance_id, ref_type.value, text),
        ref_type=ref_type,
        text=text,
        source_utterance_id=utterance.utterance_id,
        ts_ms=utterance.start_ms,
        metadata={"rule": rule},
    )


def _id(prefix: str, *parts: str) -> str:
    digest = sha1(":".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"
