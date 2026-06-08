from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from proactive_assistant.memory.contracts import MemoryRecord


class MemoryVectorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    embedding: list[float]
    content_hash: str = Field(min_length=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class MemoryVectorStore(Protocol):
    def upsert_embedding(self, record: MemoryVectorRecord) -> MemoryVectorRecord: ...

    def get_embedding(self, memory_id: str, embedding_model: str) -> MemoryVectorRecord | None: ...

    def delete_embedding(self, memory_id: str, embedding_model: str | None = None) -> int: ...


class InMemoryMemoryVectorStore:
    """In-memory vector index used by tests and local non-persistent runs."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], MemoryVectorRecord] = {}

    def upsert_embedding(self, record: MemoryVectorRecord) -> MemoryVectorRecord:
        validated = _validated_vector_record(record)
        self._records[(validated.memory_id, validated.embedding_model)] = deepcopy(validated)
        return deepcopy(validated)

    def get_embedding(self, memory_id: str, embedding_model: str) -> MemoryVectorRecord | None:
        record = self._records.get((memory_id, embedding_model))
        return deepcopy(record) if record is not None else None

    def delete_embedding(self, memory_id: str, embedding_model: str | None = None) -> int:
        keys = [
            key
            for key in self._records
            if key[0] == memory_id and (embedding_model is None or key[1] == embedding_model)
        ]
        for key in keys:
            del self._records[key]
        return len(keys)


def memory_embedding_text(memory: MemoryRecord) -> str:
    metadata_parts: list[str] = []
    for key in [
        "canonical_entity",
        "normalized_entity",
        "entity",
        "topic",
        "project",
        "owner",
        "assignee",
        "deadline",
        "normalized_deadline",
        "status",
    ]:
        value = memory.metadata.get(key)
        if value:
            metadata_parts.append(f"{key}: {value}")
    return "\n".join(
        [
            f"type: {memory.memory_type}",
            f"scope: {memory.scope}",
            f"text: {memory.text}",
            f"tags: {', '.join(memory.tags)}",
            *metadata_parts,
        ]
    ).strip()


def memory_embedding_content_hash(memory: MemoryRecord) -> str:
    payload = {
        "memory_type": str(memory.memory_type),
        "scope": str(memory.scope),
        "text": memory.text,
        "tags": memory.tags,
        "metadata": {
            key: memory.metadata.get(key)
            for key in sorted(memory.metadata)
            if key
            in {
                "canonical_entity",
                "normalized_entity",
                "entity",
                "topic",
                "project",
                "owner",
                "assignee",
                "deadline",
                "normalized_deadline",
                "status",
            }
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _validated_vector_record(record: MemoryVectorRecord) -> MemoryVectorRecord:
    if len(record.embedding) != record.dimensions:
        raise ValueError("embedding dimensions must match embedding length")
    return record
