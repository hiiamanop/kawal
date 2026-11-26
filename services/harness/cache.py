from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.harness import (
    ArbitrationOutput,
    ExtractionOutput,
    LocationQuality,
    SourceIntent,
    TriageOutput,
    VerificationOutput,
)
from contracts.models import Category


EMBEDDING_DIMENSIONS = 768


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def compute_cache_key(
    namespace: str,
    analysis_version: str,
    embedding_model: str,
    text_hash: str,
) -> str:
    payload = "\x1f".join((namespace, analysis_version, embedding_model, text_hash))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    triage: TriageOutput
    extraction: ExtractionOutput
    verification: VerificationOutput
    arbitration: ArbitrationOutput | None = None
    reason_codes: tuple[str, ...] = ()


class CacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_key: str = Field(min_length=64, max_length=64)
    namespace: str = Field(min_length=1)
    analysis_version: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    text_hash: str = Field(min_length=64, max_length=64)
    source_post_id: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    embedding: tuple[float, ...]
    category: Category
    rule_category: Category
    source_intent: SourceIntent
    location_quality: LocationQuality
    location_entities: tuple[str, ...] = ()
    evidence_spans: tuple[str, ...] = ()
    analysis: CachedAnalysis
    record_guards: dict[str, object] = Field(default_factory=dict)
    trust_decision: Literal["AGREED"] = "AGREED"
    status: Literal["ACCEPTED"] = "ACCEPTED"
    hit_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_hit_at: datetime | None = None

    @field_validator("embedding")
    @classmethod
    def _embedding_size(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding must have {EMBEDDING_DIMENSIONS} values")
        return value


class CacheQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: str = Field(min_length=1)
    analysis_version: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding: tuple[float, ...]
    category: Category
    source_intent: SourceIntent
    min_similarity: float = Field(default=0.92, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=50)

    @field_validator("embedding")
    @classmethod
    def _embedding_size(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding must have {EMBEDDING_DIMENSIONS} values")
        return value


class CacheCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entry: CacheEntry
    similarity: float = Field(ge=0.0, le=1.0)


class CacheLookupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hit: bool
    match_type: Literal["EXACT", "SEMANTIC", "NONE"] = "NONE"
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    entry: CacheEntry | None = None
    reason_codes: tuple[str, ...] = ()


class CacheRecordResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stored: bool
    entry: CacheEntry | None = None
    reason_codes: tuple[str, ...] = ()


class HarnessCacheStore(Protocol):
    def find_exact(
        self,
        namespace: str,
        analysis_version: str,
        embedding_model: str,
        text_hash: str,
    ) -> CacheEntry | None: ...

    def find_similar(self, query: CacheQuery) -> list[CacheCandidate]: ...

    def save(self, entry: CacheEntry) -> None: ...

    def increment_hit(self, cache_key: str) -> None: ...

    def reset(self, namespace: str, analysis_version: str) -> int: ...


def _lookup_key(
    namespace: str, analysis_version: str, embedding_model: str, text_hash: str
) -> tuple[str, str, str, str]:
    return (namespace, analysis_version, embedding_model, text_hash)


__all__ = [
    "CachedAnalysis",
    "CacheCandidate",
    "CacheEntry",
    "CacheLookupResult",
    "CacheQuery",
    "CacheRecordResult",
    "HarnessCacheStore",
    "compute_cache_key",
    "compute_text_hash",
    "normalize_text",
    "_lookup_key",
]
