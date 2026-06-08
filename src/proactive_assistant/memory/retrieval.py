from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from proactive_assistant.memory.contracts import (
    MemoryContext,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalIntent,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
    MemoryUsePolicy,
)
from proactive_assistant.memory.store import MemoryRepository
from proactive_assistant.memory.vector_store import (
    MemoryVectorRecord,
    MemoryVectorStore,
    cosine_similarity,
    memory_embedding_content_hash,
    memory_embedding_text,
)
from proactive_assistant.model_gateway.embeddings import EmbeddingClient
from proactive_assistant.prompting import PRDSurface, PrivacyLevel, PromptCategory


HISTORY_TERMS = {"之前", "上次", "原来", "当初", "定的", "说过", "记得", "回忆", "历史"}
DEADLINE_TERMS = {"截止", "deadline", "什么时候", "哪天", "几号", "日期", "时间"}
OWNER_TERMS = {"谁", "owner", "负责人", "负责", "归谁", "谁来"}
STATUS_TERMS = {"进展", "状态", "做完", "完成", "闭环", "推进", "处理到哪"}
RATIONALE_TERMS = {"为什么", "当时", "怎么决定", "原因", "依据", "背景"}
ANAPHORA_TERMS = {"这个", "那个", "它", "上面", "刚才说的", "这件事", "那件事", "这事", "该事项"}
ROLE_TERMS = {"负责", "负责人", "owner", "deadline", "截止", "谁", "什么时候", "哪天", "几号", "日期", "时间"}
STOP_TERMS = {"这个", "那个", "我们", "你们", "他们", "一个", "一下", "是否", "可以", "需要", "什么", "怎么"}

KNOWN_CHINESE_TERMS = [
    "负责人",
    "截止时间",
    "下周五",
    "本周五",
    "下周一",
    "本周一",
    "定案",
    "风险",
    "项目",
    "会议",
    "行动项",
    "结论",
    "决策",
    "背景",
    "原因",
    "客户",
    "报价",
    "隐私",
    "提示",
    "眼镜端",
    "查漏补缺",
]


@dataclass(frozen=True)
class StructuredMemoryQuery:
    intent: MemoryRetrievalIntent
    target_entity: str | None
    target_confidence: float
    keywords: list[str]
    normalized_dates: list[str]
    preferred_memory_types: list[str]
    reference_time: datetime


class MemoryRetriever:
    """Memory retrieval and ranking engine.

    Exact lookup intents take a conservative filtered path. Open recall uses
    explainable lexical/type/scope features and can optionally blend semantic
    similarity from a pluggable embedding client plus vector store.
    """

    def __init__(
        self,
        store: MemoryRepository,
        *,
        embedding_client: EmbeddingClient | None = None,
        vector_store: MemoryVectorStore | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.store = store
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._embedding_model = embedding_model

    def retrieve(self, query: MemoryQuery) -> MemoryContext:
        structured = build_structured_query(query)
        candidates = self._candidate_pool(query, structured)
        semantic_scores = self._semantic_scores(query, candidates)
        if structured.intent in {
            MemoryRetrievalIntent.LOOKUP_DEADLINE,
            MemoryRetrievalIntent.LOOKUP_OWNER,
            MemoryRetrievalIntent.LOOKUP_STATUS,
        }:
            results = self._exact_lookup(candidates, query, structured)
        else:
            results = self._open_recall(candidates, query, structured, semantic_scores)
        return MemoryContext(
            memory_context=[_context_line(result) for result in results],
            memory_refs=[f"memory:{result.memory.memory_id}" for result in results],
            results=results,
        )

    def _candidate_pool(self, query: MemoryQuery, structured: StructuredMemoryQuery) -> list[MemoryRecord]:
        include_pending = query.include_pending or _default_include_pending(query.prompt_category, query.activity_phase)
        base_query = query.model_copy(
            update={
                "query_text": "",
                "include_pending": include_pending,
                "limit": 200,
            }
        )
        return self.store.list_memories(base_query)

    def _exact_lookup(
        self,
        candidates: list[MemoryRecord],
        query: MemoryQuery,
        structured: StructuredMemoryQuery,
    ) -> list[MemorySearchResult]:
        exact_candidates = [memory for memory in candidates if _supports_exact_intent(memory, structured.intent)]
        scored: list[MemorySearchResult] = []
        for memory in exact_candidates:
            target_strength = _target_strength(memory, structured.target_entity)
            if structured.target_entity and target_strength < 0.25:
                continue
            feature_score = _exact_feature_score(memory, structured)
            if feature_score <= 0.0:
                continue
            confidence_gate = target_strength if structured.target_entity else 0.45
            if confidence_gate < 0.25:
                continue
            score = round(min(1.0, 0.65 * feature_score + 0.25 * target_strength + 0.10 * memory.confidence), 4)
            use_policy = _use_policy(memory, query)
            if structured.target_entity and target_strength < 0.55:
                use_policy = MemoryUsePolicy.DISPLAY_REF_ONLY
            scored.append(
                MemorySearchResult(
                    memory=memory,
                    score=score,
                    matched_terms=_matched_terms(memory, structured),
                    reason=f"exact_{structured.intent.value}",
                    rank_features={
                        "exact_feature": round(feature_score, 4),
                        "target_strength": round(target_strength, 4),
                        "confidence": memory.confidence,
                    },
                    use_policy=use_policy,
                    provenance=_provenance(memory),
                    intent=structured.intent,
                    target_entity=structured.target_entity,
                )
            )
        return sorted(
            scored,
            key=lambda item: (-item.score, -item.memory.created_at.timestamp(), item.memory.memory_id),
        )[: query.limit]

    def _open_recall(
        self,
        candidates: list[MemoryRecord],
        query: MemoryQuery,
        structured: StructuredMemoryQuery,
        semantic_scores: dict[str, float],
    ) -> list[MemorySearchResult]:
        scored: list[MemorySearchResult] = []
        for memory in candidates:
            score, features = _open_recall_score(
                memory,
                query,
                structured,
                semantic_score=semantic_scores.get(memory.memory_id, 0.0),
            )
            matched_terms = _matched_terms(memory, structured)
            if (
                structured.keywords
                and not matched_terms
                and features["source_ref"] == 0.0
                and features.get("semantic", 0.0) < query.semantic_min_score
            ):
                continue
            scored.append(
                MemorySearchResult(
                    memory=memory,
                    score=score,
                    matched_terms=matched_terms,
                    reason=_reason_for_open_recall(query, matched_terms),
                    rank_features=features,
                    use_policy=_use_policy(memory, query),
                    provenance=_provenance(memory),
                    intent=structured.intent,
                    target_entity=structured.target_entity,
                )
            )
        ranked = sorted(
            scored,
            key=lambda item: (
                -item.score,
                -item.memory.importance,
                -item.memory.confidence,
                -item.memory.created_at.timestamp(),
                item.memory.memory_id,
            ),
        )
        return _diversify(ranked, query.limit)

    def _semantic_scores(self, query: MemoryQuery, candidates: list[MemoryRecord]) -> dict[str, float]:
        if (
            not query.use_semantic_retrieval
            or self._embedding_client is None
            or self._vector_store is None
            or not self._embedding_model
            or not candidates
        ):
            return {}

        query_text = _semantic_query_text(query)
        if not query_text:
            return {}

        query_embedding = self._embedding_client.embed_texts([query_text], model=self._embedding_model).embeddings[0]
        memory_embeddings: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        missing_memories: list[tuple[MemoryRecord, str]] = []
        for memory in candidates:
            content_hash = memory_embedding_content_hash(memory)
            stored = self._vector_store.get_embedding(memory.memory_id, self._embedding_model)
            if stored is not None and stored.content_hash == content_hash:
                memory_embeddings[memory.memory_id] = stored.embedding
                continue
            missing_texts.append(memory_embedding_text(memory))
            missing_memories.append((memory, content_hash))

        if missing_texts:
            response = self._embedding_client.embed_texts(missing_texts, model=self._embedding_model)
            for (memory, content_hash), embedding in zip(missing_memories, response.embeddings, strict=True):
                record = MemoryVectorRecord(
                    memory_id=memory.memory_id,
                    embedding_model=self._embedding_model,
                    dimensions=len(embedding),
                    embedding=embedding,
                    content_hash=content_hash,
                    updated_at=datetime.now(UTC),
                )
                self._vector_store.upsert_embedding(record)
                memory_embeddings[memory.memory_id] = embedding

        return {
            memory_id: round(cosine_similarity(query_embedding, embedding), 4)
            for memory_id, embedding in memory_embeddings.items()
        }


def build_structured_query(query: MemoryQuery) -> StructuredMemoryQuery:
    reference_time = query.reference_time or datetime.now(UTC)
    target_entity, target_confidence = _resolve_target_entity(query)
    return StructuredMemoryQuery(
        intent=_classify_intent(query.query_text, query.prompt_category),
        target_entity=_normalize_entity_text(target_entity) if target_entity else None,
        target_confidence=target_confidence,
        keywords=_extract_keywords(" ".join([query.query_text, query.recent_transcript_text])),
        normalized_dates=_normalize_dates(query.query_text, reference_time),
        preferred_memory_types=_preferred_memory_types(query.prompt_category),
        reference_time=reference_time,
    )


def _classify_intent(query_text: str, prompt_category: PromptCategory | str | None) -> MemoryRetrievalIntent:
    lowered = query_text.lower()
    has_back = _has_any(lowered, HISTORY_TERMS)
    if has_back and _has_any(lowered, DEADLINE_TERMS):
        return MemoryRetrievalIntent.LOOKUP_DEADLINE
    if has_back and _has_any(lowered, OWNER_TERMS):
        return MemoryRetrievalIntent.LOOKUP_OWNER
    if has_back and _has_any(lowered, STATUS_TERMS):
        return MemoryRetrievalIntent.LOOKUP_STATUS
    if _has_any(lowered, RATIONALE_TERMS):
        return MemoryRetrievalIntent.LOOKUP_RATIONALE
    return MemoryRetrievalIntent.OPEN_RECALL


def _resolve_target_entity(query: MemoryQuery) -> tuple[str | None, float]:
    if query.target_entity:
        return query.target_entity, 1.0
    query_text = query.query_text.lower()
    entities = [_normalize_entity_payload(entity) for entity in query.active_entities]
    for entity in entities:
        names = [entity["canonical_name"], *entity["aliases"]]
        for name in names:
            if name and _normalize_text(name) in _normalize_text(query_text):
                return entity["canonical_name"], 1.0
    if _has_any(query_text, ANAPHORA_TERMS) and entities:
        sorted_entities = sorted(entities, key=lambda item: item["last_ts"], reverse=True)
        if len(sorted_entities) == 1 or sorted_entities[0]["last_ts"] - sorted_entities[1]["last_ts"] > 1500:
            return sorted_entities[0]["canonical_name"], 0.7
        return None, 0.2
    if entities:
        sorted_entities = sorted(entities, key=lambda item: item["last_ts"], reverse=True)
        return sorted_entities[0]["canonical_name"], 0.35
    return None, 0.0


def _normalize_entity_payload(entity: dict[str, Any]) -> dict[str, Any]:
    name = str(entity.get("canonical_name") or entity.get("name") or entity.get("id") or "")
    aliases = entity.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    last_ts = entity.get("last_ts", entity.get("last_mentioned_ts", entity.get("ts_ms", 0)))
    return {
        "canonical_name": name,
        "aliases": [str(alias) for alias in aliases if str(alias)],
        "last_ts": int(last_ts) if isinstance(last_ts, int | float | str) and str(last_ts).isdigit() else 0,
    }


def _default_include_pending(prompt_category: PromptCategory | str | None, phase: str) -> bool:
    category = str(prompt_category or "")
    if category == PromptCategory.SUMMARY_GAP_CHECK.value:
        return True
    return phase == "closing"


def _preferred_memory_types(prompt_category: PromptCategory | str | None) -> list[str]:
    category = str(prompt_category or "")
    return {
        PromptCategory.QUESTION_ANSWER.value: [MemoryType.MEETING_FACT.value, MemoryType.PERSON_OR_FACT.value, MemoryType.DECISION.value],
        PromptCategory.PERSON_OR_FACT.value: [MemoryType.PERSON_OR_FACT.value, MemoryType.PROJECT_CONTEXT.value, MemoryType.MEETING_FACT.value],
        PromptCategory.SUGGESTION.value: [MemoryType.USER_PREFERENCE.value, MemoryType.ACTION_ITEM.value, MemoryType.PROJECT_CONTEXT.value],
        PromptCategory.SUMMARY_GAP_CHECK.value: [MemoryType.ACTION_ITEM.value, MemoryType.DECISION.value, MemoryType.MEETING_FACT.value, MemoryType.SUMMARY.value],
        PromptCategory.CONCEPT_EXPLANATION.value: [MemoryType.PROJECT_CONTEXT.value, MemoryType.MEETING_FACT.value, MemoryType.USER_PREFERENCE.value],
    }.get(category, [])


def _supports_exact_intent(memory: MemoryRecord, intent: MemoryRetrievalIntent) -> bool:
    metadata = memory.metadata
    memory_type = str(memory.memory_type)
    if intent == MemoryRetrievalIntent.LOOKUP_DEADLINE:
        return memory_type in {MemoryType.ACTION_ITEM.value, MemoryType.MEETING_FACT.value, MemoryType.DECISION.value} and bool(
            metadata.get("normalized_deadline") or metadata.get("deadline") or _has_any(memory.text.lower(), DEADLINE_TERMS)
        )
    if intent == MemoryRetrievalIntent.LOOKUP_OWNER:
        return memory_type in {MemoryType.ACTION_ITEM.value, MemoryType.MEETING_FACT.value, MemoryType.PERSON_OR_FACT.value} and bool(
            metadata.get("owner") or metadata.get("assignee") or _has_any(memory.text.lower(), OWNER_TERMS)
        )
    if intent == MemoryRetrievalIntent.LOOKUP_STATUS:
        return bool(metadata.get("status") or _has_any(memory.text.lower(), STATUS_TERMS))
    return False


def _exact_feature_score(memory: MemoryRecord, structured: StructuredMemoryQuery) -> float:
    metadata = memory.metadata
    if structured.intent == MemoryRetrievalIntent.LOOKUP_DEADLINE:
        if metadata.get("normalized_deadline") or metadata.get("deadline"):
            return 1.0
        return 0.55 if _has_any(memory.text.lower(), DEADLINE_TERMS) else 0.0
    if structured.intent == MemoryRetrievalIntent.LOOKUP_OWNER:
        if metadata.get("owner") or metadata.get("assignee"):
            return 1.0
        return 0.55 if _has_any(memory.text.lower(), OWNER_TERMS) else 0.0
    if structured.intent == MemoryRetrievalIntent.LOOKUP_STATUS:
        if metadata.get("status"):
            return 1.0
        return 0.55 if _has_any(memory.text.lower(), STATUS_TERMS) else 0.0
    return 0.0


def _open_recall_score(
    memory: MemoryRecord,
    query: MemoryQuery,
    structured: StructuredMemoryQuery,
    *,
    semantic_score: float = 0.0,
) -> tuple[float, dict[str, float]]:
    lexical = _lexical_score(memory, structured)
    type_fit = _type_fit(memory, structured)
    scope_fit = _scope_fit(memory)
    recency = _recency_score(memory, structured.reference_time, query.activity_phase)
    source_ref = _source_ref_score(memory, query)
    feedback = _feedback_affinity(memory)
    redundancy = 1.0 if memory.memory_id in set(query.shown_memory_ids) else 0.0
    features = {
        "lexical": lexical,
        "type_fit": type_fit,
        "scope_fit": scope_fit,
        "recency": recency,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "feedback": feedback,
        "source_ref": source_ref,
        "redundancy": redundancy,
        "semantic": semantic_score if query.use_semantic_retrieval else 0.0,
    }
    if query.use_semantic_retrieval:
        weights = {
            "semantic": 0.30,
            "lexical": 0.17,
            "type_fit": 0.13,
            "scope_fit": 0.12,
            "recency": 0.10,
            "importance": 0.08,
            "confidence": 0.06,
            "feedback": 0.03,
            "source_ref": 0.01,
        }
    else:
        weights = {
            "lexical": 0.25,
            "type_fit": 0.15,
            "scope_fit": 0.15,
            "recency": 0.12,
            "importance": 0.11,
            "confidence": 0.10,
            "feedback": 0.07,
            "source_ref": 0.05,
        }
    score = sum(weights[key] * features[key] for key in weights) - 0.12 * redundancy
    return round(_clamp(score), 4), {key: round(value, 4) for key, value in features.items()}


def _lexical_score(memory: MemoryRecord, structured: StructuredMemoryQuery) -> float:
    query_terms = list(dict.fromkeys([*structured.keywords, *structured.normalized_dates]))
    if not query_terms:
        return 0.0
    memory_terms = _memory_terms(memory)
    total = sum(_term_weight(term) for term in query_terms)
    if total <= 0:
        return 0.0
    matched = sum(_term_weight(term) for term in query_terms if term in memory_terms)
    return _clamp(matched / total)


def _matched_terms(memory: MemoryRecord, structured: StructuredMemoryQuery) -> list[str]:
    memory_terms = _memory_terms(memory)
    return sorted({term for term in [*structured.keywords, *structured.normalized_dates] if term in memory_terms})


def _memory_terms(memory: MemoryRecord) -> set[str]:
    metadata_values = []
    for key in [
        "normalized_entity",
        "canonical_entity",
        "entity",
        "normalized_deadline",
        "deadline",
        "owner",
        "assignee",
        "status",
        "project",
        "topic",
    ]:
        value = memory.metadata.get(key)
        if value is not None:
            metadata_values.append(str(value))
    text = " ".join([memory.text, *memory.tags, *memory.source_ids, *metadata_values])
    terms = set(_extract_keywords(text))
    terms.update(_normalize_dates(text, memory.created_at))
    return terms


def _target_strength(memory: MemoryRecord, target_entity: str | None) -> float:
    if not target_entity:
        return 0.0
    target_terms = set(_extract_keywords(target_entity)) or {_normalize_text(target_entity)}
    memory_entity_text = " ".join(
        str(memory.metadata.get(key, ""))
        for key in ["normalized_entity", "canonical_entity", "entity", "project", "topic"]
    )
    memory_terms = set(_extract_keywords(memory_entity_text)) | _memory_terms(memory)
    if not target_terms:
        return 0.0
    return _clamp(len(target_terms & memory_terms) / len(target_terms))


def _type_fit(memory: MemoryRecord, structured: StructuredMemoryQuery) -> float:
    if not structured.preferred_memory_types:
        return 0.5
    return 1.0 if str(memory.memory_type) in set(structured.preferred_memory_types) else 0.3


def _scope_fit(memory: MemoryRecord) -> float:
    return {
        MemoryScope.SESSION.value: 1.0,
        MemoryScope.USER.value: 0.7,
        MemoryScope.ORG.value: 0.4,
        MemoryScope.GLOBAL.value: 0.3,
    }.get(str(memory.scope), 0.3)


def _recency_score(memory: MemoryRecord, now: datetime, phase: str) -> float:
    half_life_days = 7.0 if phase == "closing" else 30.0
    age_seconds = max(0.0, (now - memory.created_at).total_seconds())
    age_days = age_seconds / 86400.0
    return _clamp(math.pow(0.5, age_days / half_life_days))


def _source_ref_score(memory: MemoryRecord, query: MemoryQuery) -> float:
    return 1.0 if query.source_ids and set(query.source_ids).intersection(memory.source_ids) else 0.0


def _feedback_affinity(memory: MemoryRecord) -> float:
    if "feedback_affinity" in memory.metadata:
        try:
            return _clamp(float(memory.metadata["feedback_affinity"]))
        except (TypeError, ValueError):
            return 0.5
    memory_type = str(memory.memory_type)
    if memory_type in {MemoryType.USER_PREFERENCE.value, MemoryType.PRIVACY_PREFERENCE.value, MemoryType.NEGATIVE_PREFERENCE.value}:
        return 0.75
    return 0.5


def _use_policy(memory: MemoryRecord, query: MemoryQuery) -> MemoryUsePolicy:
    surface = str(query.prd_surface or "")
    if memory.privacy_level == PrivacyLevel.HIGH.value and surface in {
        PRDSurface.GLASSES_POPUP.value,
        PRDSurface.GLASSES_STARTING.value,
        PRDSurface.GLASSES_PERSISTENT.value,
    }:
        return MemoryUsePolicy.DISPLAY_REF_ONLY
    if memory.privacy_level == PrivacyLevel.HIGH.value and any("眼镜" in item or "客户" in item for item in query.privacy_constraints):
        return MemoryUsePolicy.DISPLAY_REF_ONLY
    if str(memory.memory_type) in {MemoryType.USER_PREFERENCE.value, MemoryType.NEGATIVE_PREFERENCE.value, MemoryType.PRIVACY_PREFERENCE.value}:
        return MemoryUsePolicy.POLICY_HINT
    return MemoryUsePolicy.PROMPT_CONTEXT


def _context_line(result: MemorySearchResult) -> str:
    memory = result.memory
    if result.use_policy == MemoryUsePolicy.DISPLAY_REF_ONLY.value:
        provenance = f" source={','.join(result.provenance)}" if result.provenance else ""
        return f"[memory:{memory.memory_id}] ({memory.memory_type}/{memory.scope}) [ref_only]{provenance}"
    if result.use_policy == MemoryUsePolicy.BLOCKED_BY_PRIVACY.value:
        return f"[memory:{memory.memory_id}] ({memory.memory_type}/{memory.scope}) [blocked_by_privacy]"
    return f"[memory:{memory.memory_id}] ({memory.memory_type}/{memory.scope}) {memory.text}"


def _provenance(memory: MemoryRecord) -> list[str]:
    provenance = memory.metadata.get("provenance")
    if isinstance(provenance, list):
        return [str(item) for item in provenance if str(item)]
    if isinstance(provenance, str) and provenance:
        return [provenance]
    return list(memory.source_ids)


def _reason_for_open_recall(query: MemoryQuery, matched_terms: list[str]) -> str:
    if not query.query_text:
        return "ranked by importance and confidence"
    if query.use_semantic_retrieval and not matched_terms:
        return "open_recall: semantic hybrid ranking"
    if query.use_semantic_retrieval:
        return "open_recall: semantic/keyword hybrid ranking"
    if matched_terms:
        return "open_recall: keyword/type/scope/recency ranking"
    return "open_recall: source or context fit"


def _semantic_query_text(query: MemoryQuery) -> str:
    parts = [
        query.query_text,
        query.recent_transcript_text,
        " ".join(str(entity.get("canonical_name") or entity.get("name") or "") for entity in query.active_entities),
        " ".join(query.current_gap_types),
        str(query.prompt_category or ""),
        query.activity_phase,
    ]
    return " ".join(part for part in parts if part).strip()


def _diversify(results: list[MemorySearchResult], limit: int) -> list[MemorySearchResult]:
    selected: list[MemorySearchResult] = []
    type_counts: dict[str, int] = {}
    max_per_type = max(2, math.ceil(limit / 2))
    for result in results:
        memory = result.memory
        memory_type = str(memory.memory_type)
        if type_counts.get(memory_type, 0) >= max_per_type:
            continue
        if _near_duplicate(memory.text, [item.memory.text for item in selected]):
            continue
        selected.append(result)
        type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
        if len(selected) >= limit:
            return selected
    return selected


def _near_duplicate(text: str, prior_texts: list[str]) -> bool:
    terms = set(_extract_keywords(text))
    if not terms:
        return False
    for prior in prior_texts:
        prior_terms = set(_extract_keywords(prior))
        if not prior_terms:
            continue
        overlap = len(terms & prior_terms) / max(len(terms | prior_terms), 1)
        if overlap >= 0.85:
            return True
    return False


def _extract_keywords(text: str) -> list[str]:
    normalized = text.lower()
    terms: list[str] = []
    terms.extend(re.findall(r"[a-z0-9_]+", normalized))
    for term in KNOWN_CHINESE_TERMS:
        if term in text:
            terms.append(term)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if chunk in STOP_TERMS:
            continue
        for size in (2, 3):
            if len(chunk) >= size:
                terms.extend(chunk[index : index + size] for index in range(0, len(chunk) - size + 1))
    return [term for term in dict.fromkeys(terms) if term and term not in STOP_TERMS]


def _normalize_dates(text: str, reference_time: datetime) -> list[str]:
    dates: list[str] = []
    dates.extend(re.findall(r"20\d{2}-\d{1,2}-\d{1,2}", text))
    for month, day in re.findall(r"(\d{1,2})月(\d{1,2})[日号]", text):
        dates.append(f"{reference_time.year:04d}-{int(month):02d}-{int(day):02d}")
    for month, day in re.findall(r"(\d{1,2})/(\d{1,2})", text):
        dates.append(f"{reference_time.year:04d}-{int(month):02d}-{int(day):02d}")
    weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    for prefix, weekday in re.findall(r"(下周|本周|这周)([一二三四五六日天])", text):
        base = reference_time.date()
        week_start = base - timedelta(days=base.weekday())
        if prefix == "下周":
            week_start = week_start + timedelta(days=7)
        resolved = week_start + timedelta(days=weekday_map[weekday])
        dates.append(resolved.isoformat())
    return sorted(set(dates))


def _term_weight(term: str) -> float:
    if term in ROLE_TERMS:
        return 0.3
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", term):
        return 1.2
    return 1.0


def _has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _normalize_entity_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))
