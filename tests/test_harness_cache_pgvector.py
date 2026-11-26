from __future__ import annotations

from datetime import datetime, timezone
import json
from unittest.mock import MagicMock

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
from services.harness.cache_store import PgVectorHarnessCacheStore


def _analysis() -> CachedAnalysis:
    return CachedAnalysis(
        triage=TriageOutput(
            is_complaint=True,
            category=Category.ROAD,
            risk=RiskLevel.HIGH,
            confidence=0.9,
            reason_codes=("ROAD_DAMAGE",),
        ),
        extraction=ExtractionOutput(
            location="Kopo",
            object="jalan rusak",
            evidence_spans=("Jl. Kopo", "lubang"),
            confidence=0.9,
        ),
        verification=VerificationOutput(
            is_complaint=True,
            category=Category.ROAD,
            evidence_supported=True,
            issues=(),
            confidence=0.85,
        ),
    )


def _entry() -> CacheEntry:
    text_hash = compute_text_hash("Jalan rusak di Kopo")
    return CacheEntry(
        cache_key=compute_cache_key(
            "harness-x", "v1", "deterministic-hash-768-v1", text_hash
        ),
        namespace="harness-x",
        analysis_version="v1",
        embedding_model="deterministic-hash-768-v1",
        text_hash=text_hash,
        source_post_id="post-1",
        source_run_id="run-1",
        embedding=tuple([1.0] + [0.0] * 767),
        category=Category.ROAD,
        rule_category=Category.ROAD,
        source_intent="CITIZEN_COMPLAINT",
        location_quality="SPECIFIC",
        location_entities=("kopo",),
        evidence_spans=("Jl. Kopo", "lubang"),
        analysis=_analysis(),
        record_guards={"rule_agreement": True},
        created_at=datetime.now(timezone.utc),
    )


def _query() -> CacheQuery:
    return CacheQuery(
        namespace="harness-x",
        analysis_version="v1",
        embedding_model="deterministic-hash-768-v1",
        embedding=tuple([1.0] + [0.0] * 767),
        category=Category.ROAD,
        source_intent="CITIZEN_COMPLAINT",
        min_similarity=0.9,
        limit=5,
    )


def _mock_store() -> tuple[PgVectorHarnessCacheStore, MagicMock]:
    connection = MagicMock()
    cursor = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    return PgVectorHarnessCacheStore(lambda: connection), cursor


def _row(entry: CacheEntry, similarity: float | None = None) -> tuple[object, ...]:
    values: tuple[object, ...] = (
        entry.cache_key,
        entry.namespace,
        entry.analysis_version,
        entry.embedding_model,
        entry.text_hash,
        entry.source_post_id,
        entry.source_run_id,
        list(entry.embedding),
        entry.category.value,
        entry.rule_category.value,
        entry.source_intent,
        entry.location_quality,
        json.dumps(entry.location_entities),
        json.dumps(entry.evidence_spans),
        json.dumps(entry.analysis.model_dump(mode="json")),
        json.dumps(entry.record_guards),
        entry.trust_decision,
        entry.status,
        entry.hit_count,
        entry.created_at,
        entry.last_hit_at,
    )
    return values + ((similarity,) if similarity is not None else ())


def test_pgvector_find_similar_uses_cosine_operator() -> None:
    store, cursor = _mock_store()
    cursor.fetchall.return_value = []
    store.find_similar(_query())
    sql = cursor.execute.call_args.args[0].lower()
    assert "<=>" in sql
    assert "::vector" in sql
    assert "namespace = %s" in sql
    assert "category = %s" in sql
    assert "source_intent = %s" in sql


def test_pgvector_save_upserts_without_raw_text() -> None:
    store, cursor = _mock_store()
    store.save(_entry())
    sql = cursor.execute.call_args.args[0].lower()
    assert "insert into harness_case_cache" in sql
    assert "on conflict (cache_key)" in sql
    assert "raw_text" not in sql


def test_pgvector_find_exact_round_trips_entry() -> None:
    entry = _entry()
    store, cursor = _mock_store()
    cursor.fetchone.return_value = _row(entry)
    found = store.find_exact(
        entry.namespace,
        entry.analysis_version,
        entry.embedding_model,
        entry.text_hash,
    )
    assert found is not None
    assert found.cache_key == entry.cache_key
    assert found.analysis.triage.category == Category.ROAD
    assert found.location_entities == ("kopo",)


def test_pgvector_find_similar_returns_similarity() -> None:
    entry = _entry()
    store, cursor = _mock_store()
    cursor.fetchall.return_value = [_row(entry, similarity=0.97)]
    results = store.find_similar(_query())
    assert len(results) == 1
    assert results[0].similarity == 0.97
    assert results[0].entry.cache_key == entry.cache_key


def test_pgvector_errors_fail_closed() -> None:
    def failing_connection() -> object:
        raise RuntimeError("database down")

    store = PgVectorHarnessCacheStore(failing_connection)
    assert store.find_exact("harness-x", "v1", "model", "a" * 64) is None
    assert store.find_similar(_query()) == []
    store.save(_entry())
    store.increment_hit("cache-key")
    assert store.reset("harness-x", "v1") == 0
