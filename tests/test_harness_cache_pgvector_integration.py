from __future__ import annotations

from datetime import datetime, timezone
import os
from uuid import uuid4

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
from services.harness.cache_store import (
    PgVectorHarnessCacheStore,
    _vector_literal,
)


pytestmark = pytest.mark.integration


def test_pgvector_cache_round_trip() -> None:
    dsn = os.getenv("KAWAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("KAWAL_TEST_DATABASE_URL is not set")
    psycopg = pytest.importorskip("psycopg")

    namespace = f"harness-x-integration-{uuid4().hex[:8]}"
    text = f"Jalan rusak di Kopo {uuid4().hex}"
    text_hash = compute_text_hash(text)
    entry = CacheEntry(
        cache_key=compute_cache_key(
            namespace, "v1", "deterministic-hash-768-v1", text_hash
        ),
        namespace=namespace,
        analysis_version="v1",
        embedding_model="deterministic-hash-768-v1",
        text_hash=text_hash,
        source_post_id=f"post-{uuid4().hex[:8]}",
        source_run_id=f"run-{uuid4().hex[:8]}",
        embedding=tuple([1.0] + [0.0] * 767),
        category=Category.ROAD,
        rule_category=Category.ROAD,
        source_intent="CITIZEN_COMPLAINT",
        location_quality="SPECIFIC",
        location_entities=("kopo",),
        evidence_spans=("Kopo",),
        analysis=CachedAnalysis(
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
                evidence_spans=("Kopo",),
                confidence=0.9,
            ),
            verification=VerificationOutput(
                is_complaint=True,
                category=Category.ROAD,
                evidence_supported=True,
                issues=(),
                confidence=0.85,
            ),
        ),
        record_guards={"pii_free": True},
        created_at=datetime.now(timezone.utc),
    )
    store = PgVectorHarnessCacheStore(lambda: psycopg.connect(dsn))

    try:
        store.save(entry)
        assert (
            store.find_exact(namespace, "v1", "deterministic-hash-768-v1", text_hash)
            is not None
        )
        results = store.find_similar(
            CacheQuery(
                namespace=namespace,
                analysis_version="v1",
                embedding_model="deterministic-hash-768-v1",
                embedding=entry.embedding,
                category=Category.ROAD,
                source_intent="CITIZEN_COMPLAINT",
                min_similarity=0.9,
                limit=5,
            )
        )
        assert results
        assert results[0].entry.cache_key == entry.cache_key
    finally:
        assert store.reset(namespace, "v1") == 1


def test_pgvector_planner_uses_hnsw_index() -> None:
    dsn = os.getenv("KAWAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("KAWAL_TEST_DATABASE_URL is not set")
    psycopg = pytest.importorskip("psycopg")
    vector = _vector_literal(tuple([1.0] + [0.0] * 767))

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")
            cur.execute(
                """
                EXPLAIN SELECT cache_key
                FROM harness_case_cache
                ORDER BY embedding <=> %s::vector
                LIMIT 5
                """,
                (vector,),
            )
            plan = "\n".join(str(row[0]) for row in cur.fetchall())
    assert "ix_harness_cache_hnsw" in plan
