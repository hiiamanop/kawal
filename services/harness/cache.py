from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.harness import (
    ArbitrationOutput,
    ComplaintCase,
    ExtractionOutput,
    LocationQuality,
    SourceIntent,
    TriageOutput,
    VerificationOutput,
)
from contracts.models import Category
from services.harness.security import contains_pii
from services.intelligence.embedder import TextEmbedder


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


class HarnessAnalysisCache:
    """Versioned semantic cache for accepted harness analyses."""

    def __init__(
        self,
        *,
        store: HarnessCacheStore,
        embedder: TextEmbedder,
        namespace: str,
        analysis_version: str,
        semantic_threshold: float = 0.92,
        allow_fallback_semantic: bool = False,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must not be empty")
        if not analysis_version:
            raise ValueError("analysis_version must not be empty")
        if not 0.0 <= semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be between 0 and 1")
        self._store = store
        self._embedder = embedder
        self._namespace = namespace
        self._analysis_version = analysis_version
        self._semantic_threshold = semantic_threshold
        self._allow_fallback_semantic = allow_fallback_semantic

    @property
    def store(self) -> HarnessCacheStore:
        return self._store

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def analysis_version(self) -> str:
        return self._analysis_version

    @property
    def embedding_model(self) -> str:
        return self._embedder.model_id

    def lookup(
        self,
        text: str,
        category: Category,
        source_intent: SourceIntent,
    ) -> CacheLookupResult:
        text_hash = compute_text_hash(text)
        exact = self._store.find_exact(
            self._namespace,
            self._analysis_version,
            self.embedding_model,
            text_hash,
        )
        if exact is not None:
            return CacheLookupResult(
                hit=True,
                match_type="EXACT",
                similarity=1.0,
                entry=exact,
                reason_codes=("CACHE_EXACT",),
            )

        if self._embedder.is_fallback and not self._allow_fallback_semantic:
            return CacheLookupResult(
                hit=False,
                match_type="NONE",
                reason_codes=("SEMANTIC_DISABLED_FALLBACK",),
            )

        embedding = tuple(self._embedder.embed_text(text))
        candidates = self._store.find_similar(
            CacheQuery(
                namespace=self._namespace,
                analysis_version=self._analysis_version,
                embedding_model=self.embedding_model,
                embedding=embedding,
                category=category,
                source_intent=source_intent,
                min_similarity=self._semantic_threshold,
                limit=5,
            )
        )
        if not candidates:
            return CacheLookupResult(
                hit=False,
                match_type="NONE",
                reason_codes=("CACHE_MISS",),
            )

        best = candidates[0]
        return CacheLookupResult(
            hit=True,
            match_type="SEMANTIC",
            similarity=best.similarity,
            entry=best.entry,
            reason_codes=("CACHE_SEMANTIC",),
        )

    def record(
        self,
        *,
        case: ComplaintCase,
        rule_triage: TriageOutput,
        rule_verification: VerificationOutput,
        run_id: str,
        location_entities: Sequence[str],
    ) -> CacheRecordResult:
        reasons: list[str] = []

        if case.status != "ACCEPTED" or case.trust_decision != "AGREED":
            reasons.append("CACHE_NOT_ACCEPTED")
        if case.location_quality != "SPECIFIC":
            reasons.append("CACHE_NOT_SPECIFIC")
        if case.source_intent != "CITIZEN_COMPLAINT":
            reasons.append("CACHE_NOT_CITIZEN")
        if not rule_triage.is_complaint or not rule_verification.is_complaint:
            reasons.append("CACHE_RULE_NOT_COMPLAINT")
        if (
            rule_triage.category != case.triage.category
            or rule_verification.category != case.triage.category
        ):
            reasons.append("CACHE_RULE_DISAGREEMENT")
        if not rule_verification.evidence_supported:
            reasons.append("CACHE_EVIDENCE_UNSUPPORTED")
        if not case.verification.evidence_supported:
            reasons.append("CACHE_EVIDENCE_UNSUPPORTED")

        normalized_locations = tuple(
            sorted(
                {
                    normalize_text(location)
                    for location in location_entities
                    if location.strip()
                }
            )
        )
        if not normalized_locations:
            reasons.append("CACHE_NO_LOCATION")

        pii_free = not contains_pii(case.model_dump_json()) and not any(
            contains_pii(location) for location in normalized_locations
        )
        if not pii_free:
            reasons.append("CACHE_PII_BLOCKED")

        if reasons:
            return CacheRecordResult(
                stored=False,
                reason_codes=tuple(dict.fromkeys(reasons)),
            )

        text_hash = compute_text_hash(case.post.text)
        cache_key = compute_cache_key(
            self._namespace,
            self._analysis_version,
            self.embedding_model,
            text_hash,
        )
        entry = CacheEntry(
            cache_key=cache_key,
            namespace=self._namespace,
            analysis_version=self._analysis_version,
            embedding_model=self.embedding_model,
            text_hash=text_hash,
            source_post_id=case.post.post_id,
            source_run_id=run_id,
            embedding=tuple(self._embedder.embed_text(case.post.text)),
            category=case.triage.category,
            rule_category=rule_triage.category,
            source_intent=case.source_intent,
            location_quality=case.location_quality,
            location_entities=normalized_locations,
            evidence_spans=case.extraction.evidence_spans,
            analysis=CachedAnalysis(
                triage=case.triage,
                extraction=case.extraction,
                verification=case.verification,
                arbitration=case.arbitration,
                reason_codes=case.reason_codes,
            ),
            record_guards={
                "source_case_id": case.case_id,
                "status": case.status,
                "trust_decision": case.trust_decision,
                "location_quality": case.location_quality,
                "source_intent": case.source_intent,
                "accepted_category": case.triage.category.value,
                "rule_triage_category": rule_triage.category.value,
                "rule_verification_category": rule_verification.category.value,
                "rule_triage_is_complaint": rule_triage.is_complaint,
                "rule_verification_is_complaint": rule_verification.is_complaint,
                "evidence_supported": rule_verification.evidence_supported,
                "pii_free": pii_free,
            },
        )
        self._store.save(entry)
        return CacheRecordResult(
            stored=True,
            entry=entry,
            reason_codes=("CACHE_RECORDED",),
        )

    def increment_hit(self, entry: CacheEntry) -> None:
        self._store.increment_hit(entry.cache_key)

    def reset(self) -> int:
        return self._store.reset(self._namespace, self._analysis_version)


__all__ = [
    "CachedAnalysis",
    "CacheCandidate",
    "CacheEntry",
    "CacheLookupResult",
    "CacheQuery",
    "CacheRecordResult",
    "HarnessAnalysisCache",
    "HarnessCacheStore",
    "compute_cache_key",
    "compute_text_hash",
    "normalize_text",
    "_lookup_key",
]
