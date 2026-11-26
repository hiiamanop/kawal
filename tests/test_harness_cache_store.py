from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contracts.harness import (
    ExtractionOutput,
    TriageOutput,
    VerificationOutput,
)
from contracts.models import Category, RiskLevel
from services.harness.cache import (
    CachedAnalysis,
    CacheEntry,
    CacheQuery,
    compute_cache_key,
    compute_text_hash,
)
from services.harness.cache_store import InMemoryHarnessCacheStore


def _analysis(category: Category = Category.ROAD) -> CachedAnalysis:
    return CachedAnalysis(
        triage=TriageOutput(
            is_complaint=True,
            category=category,
            risk=RiskLevel.HIGH,
            confidence=0.9,
            reason_codes=("ROAD_DAMAGE",),
        ),
        extraction=ExtractionOutput(
            location="Kopo",
            object="jalan rusak",
            time=None,
            evidence_spans=("Jl. Kopo", "lubang"),
            confidence=0.9,
        ),
        verification=VerificationOutput(
            is_complaint=True,
            category=category,
            evidence_supported=True,
            issues=(),
            confidence=0.85,
        ),
    )


def _entry(
    *,
    cache_key: str | None = None,
    namespace: str = "harness-x",
    analysis_version: str = "v1",
    embedding_model: str = "deterministic-hash-768-v1",
    text_hash: str | None = None,
    category: Category = Category.ROAD,
    source_intent: str = "CITIZEN_COMPLAINT",
) -> CacheEntry:
    normalized_hash = text_hash or compute_text_hash("Jalan rusak di Kopo")
    return CacheEntry(
        cache_key=cache_key
        or compute_cache_key(
            namespace, analysis_version, embedding_model, normalized_hash
        ),
        namespace=namespace,
        analysis_version=analysis_version,
        embedding_model=embedding_model,
        text_hash=normalized_hash,
        source_post_id="post-1",
        source_run_id="run-1",
        embedding=tuple([1.0] + [0.0] * 767),
        category=category,
        rule_category=category,
        source_intent=source_intent,
        location_quality="SPECIFIC",
        location_entities=("kopo",),
        evidence_spans=("Jl. Kopo", "lubang"),
        analysis=_analysis(category),
        record_guards={"rule_agreement": True},
        created_at=datetime.now(timezone.utc),
    )


def test_in_memory_exact_lookup_is_scoped_by_version() -> None:
    store = InMemoryHarnessCacheStore()
    store.save(_entry(text_hash="a" * 64))
    assert (
        store.find_exact("harness-x", "v1", "deterministic-hash-768-v1", "a" * 64)
        is not None
    )
    assert (
        store.find_exact("harness-x", "v2", "deterministic-hash-768-v1", "a" * 64)
        is None
    )


def test_in_memory_semantic_lookup_filters_category_and_source_intent() -> None:
    store = InMemoryHarnessCacheStore()
    store.save(_entry(category=Category.ROAD))
    store.save(_entry(cache_key="b" * 64, category=Category.WASTE))
    results = store.find_similar(
        CacheQuery(
            namespace="harness-x",
            analysis_version="v1",
            embedding_model="deterministic-hash-768-v1",
            embedding=tuple([1.0] + [0.0] * 767),
            category=Category.ROAD,
            source_intent="CITIZEN_COMPLAINT",
            min_similarity=0.9,
            limit=5,
        )
    )
    assert [candidate.entry.category for candidate in results] == [Category.ROAD]


def test_in_memory_reset_only_removes_matching_namespace_and_version() -> None:
    store = InMemoryHarnessCacheStore()
    store.save(_entry(namespace="harness-x", analysis_version="v1"))
    store.save(_entry(cache_key="c" * 64, namespace="other", analysis_version="v1"))
    assert store.reset("harness-x", "v1") == 1
    assert store.count == 1


def test_in_memory_namespace_and_embedding_model_are_isolated() -> None:
    store = InMemoryHarnessCacheStore()
    store.save(_entry(namespace="harness-x", text_hash="a" * 64))
    assert (
        store.find_exact("harness-x", "v1", "deterministic-hash-768-v1", "a" * 64)
        is not None
    )
    assert (
        store.find_exact("other", "v1", "deterministic-hash-768-v1", "a" * 64) is None
    )
    query = CacheQuery(
        namespace="other",
        analysis_version="v1",
        embedding_model="deterministic-hash-768-v1",
        embedding=tuple([1.0] + [0.0] * 767),
        category=Category.ROAD,
        source_intent="CITIZEN_COMPLAINT",
        min_similarity=0.9,
        limit=5,
    )
    assert store.find_similar(query) == []
    query = query.model_copy(update={"namespace": "harness-x"})
    assert store.find_similar(query)
    query = query.model_copy(update={"embedding_model": "other-model"})
    assert store.find_similar(query) == []


def test_in_memory_below_threshold_is_a_miss() -> None:
    store = InMemoryHarnessCacheStore()
    store.save(_entry())
    query = CacheQuery(
        namespace="harness-x",
        analysis_version="v1",
        embedding_model="deterministic-hash-768-v1",
        embedding=tuple([0.0, 1.0] + [0.0] * 766),
        category=Category.ROAD,
        source_intent="CITIZEN_COMPLAINT",
        min_similarity=0.9,
        limit=5,
    )
    assert store.find_similar(query) == []


def test_cache_entry_has_no_full_post_body_field() -> None:
    entry = _entry()
    dumped = entry.model_dump()
    assert "raw_text" not in dumped
    assert "post_text" not in dumped


def test_cache_entry_rejects_wrong_embedding_size() -> None:
    with pytest.raises(ValueError):
        CacheEntry.model_validate({**_entry().model_dump(), "embedding": [1.0] * 10})
