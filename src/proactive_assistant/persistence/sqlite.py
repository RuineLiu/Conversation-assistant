from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from proactive_assistant.memory import (
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryQuery,
    MemoryRecord,
    MemoryRecordUpdate,
    MemorySearchResult,
    MemoryWriteStatus,
)
from proactive_assistant.memory.store import _matches_query, _score_memory, _score_reason, _sort_records, _terms
from proactive_assistant.runtime import (
    MemoryCandidate,
    PromptDecisionRecord,
    RewardObservation,
    RuntimeFeedbackEvent,
)
from proactive_assistant.runtime.store import (
    DecisionRecordAlreadyExistsError,
    DecisionRecordNotFoundError,
    RuntimeFeedbackEventAlreadyExistsError,
    RuntimeStoreError,
)
from proactive_assistant.sessions import AssistantSession, TranscriptSegmentRecord
from proactive_assistant.sessions.store import SessionAlreadyExistsError, SessionNotFoundError


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    scene TEXT NOT NULL,
    source TEXT NOT NULL,
    locale TEXT NOT NULL,
    title TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(session_id, segment_id)
);
CREATE INDEX IF NOT EXISTS idx_transcript_segments_session_time
    ON transcript_segments(session_id, start_ms, end_ms, segment_id);

CREATE TABLE IF NOT EXISTS prompt_decisions (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    opportunity_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    display_status TEXT NOT NULL,
    prompt_category TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompt_decisions_session ON prompt_decisions(session_id, row_id);

CREATE TABLE IF NOT EXISTS feedback_events (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    decision_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_events_decision ON feedback_events(decision_id, row_id);
CREATE INDEX IF NOT EXISTS idx_feedback_events_session ON feedback_events(session_id, row_id);

CREATE TABLE IF NOT EXISTS reward_observations (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT UNIQUE NOT NULL,
    decision_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    final_reward REAL NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reward_observations_decision ON reward_observations(decision_id, row_id);
CREATE INDEX IF NOT EXISTS idx_reward_observations_session ON reward_observations(session_id, row_id);

CREATE TABLE IF NOT EXISTS memory_candidates (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_candidate_id TEXT UNIQUE NOT NULL,
    decision_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    candidate_type TEXT NOT NULL,
    write_policy TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_decision ON memory_candidates(decision_id, row_id);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_session ON memory_candidates(session_id, row_id);

CREATE TABLE IF NOT EXISTS memories (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT UNIQUE NOT NULL,
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT,
    memory_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    write_status TEXT NOT NULL,
    privacy_level TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(org_id, user_id, row_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id, row_id);
CREATE INDEX IF NOT EXISTS idx_memories_type_status ON memories(memory_type, write_status, row_id);
"""


def initialize_sqlite_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SQLITE_SCHEMA)
    connection.commit()


class SQLiteRepositoryBase:
    def __init__(self, path: str | Path | sqlite3.Connection) -> None:
        if isinstance(path, sqlite3.Connection):
            self._connection = path
            self._owns_connection = False
        else:
            resolved = Path(path)
            if str(resolved) != ":memory:":
                resolved.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(resolved))
            self._owns_connection = True
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        initialize_sqlite_schema(self._connection)

    def close(self) -> None:
        if self._owns_connection:
            self._connection.close()


class SQLiteSessionStore(SQLiteRepositoryBase):
    """SQLite-backed SessionRepository implementation."""

    def create_session(self, session: AssistantSession) -> AssistantSession:
        try:
            self._connection.execute(
                """
                INSERT INTO sessions(session_id, created_at, status, scene, source, locale, title, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    _dt(session.created_at),
                    str(session.status),
                    str(session.scene),
                    str(session.source),
                    session.locale,
                    session.title,
                    _dump_model(session),
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            raise SessionAlreadyExistsError(f"session already exists: {session.session_id}") from exc
        return _clone_model(session)

    def update_session(self, session: AssistantSession) -> AssistantSession:
        cursor = self._connection.execute(
            """
            UPDATE sessions
            SET created_at = ?, status = ?, scene = ?, source = ?, locale = ?, title = ?, payload = ?
            WHERE session_id = ?
            """,
            (
                _dt(session.created_at),
                str(session.status),
                str(session.scene),
                str(session.source),
                session.locale,
                session.title,
                _dump_model(session),
                session.session_id,
            ),
        )
        self._connection.commit()
        if cursor.rowcount == 0:
            raise SessionNotFoundError(f"session not found: {session.session_id}")
        return _clone_model(session)

    def get_session(self, session_id: str) -> AssistantSession:
        row = self._connection.execute("SELECT payload FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFoundError(f"session not found: {session_id}")
        return _load_model(row, AssistantSession)

    def list_sessions(self) -> list[AssistantSession]:
        rows = self._connection.execute("SELECT payload FROM sessions ORDER BY created_at, session_id").fetchall()
        return [_load_model(row, AssistantSession) for row in rows]

    def append_transcript(self, segment: TranscriptSegmentRecord) -> TranscriptSegmentRecord:
        self.get_session(segment.session_id)
        self._connection.execute(
            """
            INSERT INTO transcript_segments(segment_id, session_id, start_ms, end_ms, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                segment.segment_id,
                segment.session_id,
                segment.start_ms,
                segment.end_ms,
                _dt(segment.created_at),
                _dump_model(segment),
            ),
        )
        self._connection.commit()
        return _clone_model(segment)

    def list_transcript(self, session_id: str) -> list[TranscriptSegmentRecord]:
        self.get_session(session_id)
        rows = self._connection.execute(
            """
            SELECT payload FROM transcript_segments
            WHERE session_id = ?
            ORDER BY start_ms, end_ms, segment_id
            """,
            (session_id,),
        ).fetchall()
        return [_load_model(row, TranscriptSegmentRecord) for row in rows]


class SQLiteRuntimeStore(SQLiteRepositoryBase):
    """SQLite-backed RuntimeRepository implementation."""

    def add_decision(self, decision: PromptDecisionRecord) -> PromptDecisionRecord:
        try:
            self._connection.execute(
                """
                INSERT INTO prompt_decisions(
                    decision_id, session_id, candidate_id, opportunity_id, created_at,
                    display_status, prompt_category, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.session_id,
                    decision.candidate_id,
                    decision.opportunity_id,
                    _dt(decision.created_at),
                    str(decision.display_status),
                    decision.prompt_category,
                    _dump_model(decision),
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            raise DecisionRecordAlreadyExistsError(f"decision already exists: {decision.decision_id}") from exc
        return decision

    def get_decision(self, decision_id: str) -> PromptDecisionRecord:
        row = self._connection.execute(
            "SELECT payload FROM prompt_decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        if row is None:
            raise DecisionRecordNotFoundError(f"unknown decision: {decision_id}")
        return _load_model(row, PromptDecisionRecord)

    def list_decisions(self, *, session_id: str | None = None) -> list[PromptDecisionRecord]:
        if session_id is None:
            rows = self._connection.execute("SELECT payload FROM prompt_decisions ORDER BY row_id").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT payload FROM prompt_decisions WHERE session_id = ? ORDER BY row_id",
                (session_id,),
            ).fetchall()
        return [_load_model(row, PromptDecisionRecord) for row in rows]

    def add_feedback_event(self, event: RuntimeFeedbackEvent) -> RuntimeFeedbackEvent:
        decision = self.get_decision(event.decision_id)
        if event.session_id != decision.session_id:
            raise RuntimeStoreError("feedback event session_id must match decision session_id")
        try:
            self._connection.execute(
                """
                INSERT INTO feedback_events(event_id, decision_id, session_id, signal_type, occurred_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.decision_id,
                    event.session_id,
                    str(event.signal_type),
                    _dt(event.occurred_at),
                    _dump_model(event),
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            raise RuntimeFeedbackEventAlreadyExistsError(f"feedback event already exists: {event.event_id}") from exc
        return event

    def list_feedback_events(
        self,
        *,
        decision_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RuntimeFeedbackEvent]:
        rows = _select_runtime_rows(
            self._connection,
            "feedback_events",
            decision_id=decision_id,
            session_id=session_id,
        )
        return [_load_model(row, RuntimeFeedbackEvent) for row in rows]

    def add_reward_observation(self, observation: RewardObservation) -> RewardObservation:
        decision = self.get_decision(observation.decision_id)
        if observation.session_id != decision.session_id:
            raise RuntimeStoreError("reward observation session_id must match decision session_id")
        self._connection.execute(
            """
            INSERT INTO reward_observations(observation_id, decision_id, session_id, final_reward, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                observation.observation_id,
                observation.decision_id,
                observation.session_id,
                observation.final_reward,
                _dt(observation.created_at),
                _dump_model(observation),
            ),
        )
        self._connection.commit()
        return observation

    def list_reward_observations(
        self,
        *,
        decision_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RewardObservation]:
        rows = _select_runtime_rows(
            self._connection,
            "reward_observations",
            decision_id=decision_id,
            session_id=session_id,
        )
        return [_load_model(row, RewardObservation) for row in rows]

    def add_memory_candidate(self, candidate: MemoryCandidate) -> MemoryCandidate:
        decision = self.get_decision(candidate.decision_id)
        if candidate.session_id != decision.session_id:
            raise RuntimeStoreError("memory candidate session_id must match decision session_id")
        self._connection.execute(
            """
            INSERT OR REPLACE INTO memory_candidates(
                memory_candidate_id, decision_id, session_id, candidate_type, write_policy, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.memory_candidate_id,
                candidate.decision_id,
                candidate.session_id,
                str(candidate.candidate_type),
                str(candidate.write_policy),
                _dump_model(candidate),
            ),
        )
        self._connection.commit()
        return candidate

    def list_memory_candidates(
        self,
        *,
        decision_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryCandidate]:
        rows = _select_runtime_rows(
            self._connection,
            "memory_candidates",
            decision_id=decision_id,
            session_id=session_id,
        )
        return [_load_model(row, MemoryCandidate) for row in rows]


class SQLiteMemoryStore(SQLiteRepositoryBase):
    """SQLite-backed MemoryRepository implementation."""

    def add_memory(self, memory: MemoryRecord) -> MemoryRecord:
        try:
            self._connection.execute(
                """
                INSERT INTO memories(
                    memory_id, org_id, user_id, session_id, memory_type, scope,
                    write_status, privacy_level, created_at, updated_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _memory_row_values(memory),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            raise MemoryAlreadyExistsError(f"memory already exists: {memory.memory_id}") from exc
        return _clone_model(memory)

    def get_memory(self, memory_id: str) -> MemoryRecord:
        memory = self._get_memory(memory_id)
        updated = memory.model_copy(update={"last_accessed_at": _now()})
        self._write_memory(updated)
        return _clone_model(updated)

    def list_memories(self, query: MemoryQuery | None = None) -> list[MemoryRecord]:
        resolved_query = query or MemoryQuery(limit=200)
        rows = self._connection.execute("SELECT payload FROM memories ORDER BY row_id").fetchall()
        memories = [_load_model(row, MemoryRecord) for row in rows]
        filtered = [memory for memory in memories if _matches_query(memory, resolved_query)]
        return [_clone_model(memory) for memory in _sort_records(filtered)[: resolved_query.limit]]

    def search_memories(self, query: MemoryQuery) -> list[MemorySearchResult]:
        query_terms = _terms(query.query_text)
        results: list[MemorySearchResult] = []
        for memory in self.list_memories(query):
            matched_terms = sorted(query_terms & _terms(" ".join([memory.text, *memory.tags])))
            if query_terms and not matched_terms:
                continue
            results.append(
                MemorySearchResult(
                    memory=memory,
                    score=_score_memory(memory, matched_terms, query_terms),
                    matched_terms=matched_terms,
                    reason=_score_reason(matched_terms, query_terms),
                )
            )
        return sorted(
            results,
            key=lambda item: (
                -item.score,
                -item.memory.importance,
                -item.memory.confidence,
                -item.memory.created_at.timestamp(),
                item.memory.memory_id,
            ),
        )[: query.limit]

    def update_memory(self, memory_id: str, update: MemoryRecordUpdate) -> MemoryRecord:
        memory = self._get_memory(memory_id)
        updates = update.model_dump(exclude_none=True, mode="python")
        if not updates:
            return _clone_model(memory)
        updates["updated_at"] = _now()
        updated = memory.model_copy(update=updates)
        updated = MemoryRecord.model_validate(updated.model_dump(mode="python"))
        self._write_memory(updated)
        return _clone_model(updated)

    def archive_memory(self, memory_id: str, *, reason: str = "") -> MemoryRecord:
        memory = self._get_memory(memory_id)
        metadata = dict(memory.metadata)
        if reason:
            metadata["archive_reason"] = reason
        return self.update_memory(
            memory_id,
            MemoryRecordUpdate(write_status=MemoryWriteStatus.ARCHIVED, metadata=metadata),
        )

    def _get_memory(self, memory_id: str) -> MemoryRecord:
        row = self._connection.execute("SELECT payload FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"memory not found: {memory_id}")
        return _load_model(row, MemoryRecord)

    def _write_memory(self, memory: MemoryRecord) -> None:
        cursor = self._connection.execute(
            """
            UPDATE memories
            SET org_id = ?, user_id = ?, session_id = ?, memory_type = ?, scope = ?,
                write_status = ?, privacy_level = ?, created_at = ?, updated_at = ?, payload = ?
            WHERE memory_id = ?
            """,
            (
                memory.org_id,
                memory.user_id,
                memory.session_id,
                str(memory.memory_type),
                str(memory.scope),
                str(memory.write_status),
                str(memory.privacy_level),
                _dt(memory.created_at),
                _dt(memory.updated_at),
                _dump_model(memory),
                memory.memory_id,
            ),
        )
        self._connection.commit()
        if cursor.rowcount == 0:
            raise MemoryNotFoundError(f"memory not found: {memory.memory_id}")


def _select_runtime_rows(
    connection: sqlite3.Connection,
    table: str,
    *,
    decision_id: str | None,
    session_id: str | None,
) -> list[sqlite3.Row]:
    where: list[str] = []
    params: list[str] = []
    if decision_id is not None:
        where.append("decision_id = ?")
        params.append(decision_id)
    if session_id is not None:
        where.append("session_id = ?")
        params.append(session_id)
    sql = f"SELECT payload FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY row_id"
    return list(connection.execute(sql, params).fetchall())


def _memory_row_values(memory: MemoryRecord) -> tuple[Any, ...]:
    return (
        memory.memory_id,
        memory.org_id,
        memory.user_id,
        memory.session_id,
        str(memory.memory_type),
        str(memory.scope),
        str(memory.write_status),
        str(memory.privacy_level),
        _dt(memory.created_at),
        _dt(memory.updated_at),
        _dump_model(memory),
    )


def _dump_model(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def _load_model(row: sqlite3.Row, model_type: type[BaseModel]) -> Any:
    return model_type.model_validate(json.loads(row["payload"]))


def _clone_model(model: BaseModel) -> Any:
    return model.__class__.model_validate(model.model_dump(mode="python"))


def _dt(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _now() -> datetime:
    return datetime.now(UTC)
