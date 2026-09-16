from __future__ import annotations

import pytest

from contracts.models import Category, RiskLevel
from services.intelligence.embedder import (
    DeterministicFallbackEmbedder,
    IndoBertEmbedder,
    cosine_similarity,
)
from services.intelligence.semantic_cache import (
    InMemoryCacheStore,
    SemanticCacheEngine,
)


@pytest.fixture
def memory_engine() -> SemanticCacheEngine:
    store = InMemoryCacheStore()
    embedder = DeterministicFallbackEmbedder(dim=768)
    return SemanticCacheEngine(
        store=store,
        embedder=embedder,
        semantic_threshold=0.70,
        duplicate_threshold=0.60,
    )


def test_deterministic_fallback_embedder() -> None:
    embedder = DeterministicFallbackEmbedder(dim=768)
    v1 = embedder.embed_text("jalan rusak berlubang di kopo")
    v2 = embedder.embed_text("jalan rusak berlubang di kopo")
    v3 = embedder.embed_text("sampah menumpuk busuk")

    assert len(v1) == 768
    assert cosine_similarity(v1, v2) == pytest.approx(1.0, rel=1e-3)
    assert cosine_similarity(v1, v3) < 0.5


def test_exact_match_lookup(memory_engine: SemanticCacheEngine) -> None:
    text = "lapor gorong-gorong mampet air meluap di pasteur"
    memory_engine.record_case(
        tenant_id="tenant-1",
        case_id="case-101",
        text=text,
        category=Category.DRAINAGE_FLOOD,
        risk=RiskLevel.HIGH,
        completeness="SUFFICIENT",
        location_entities=["pasteur"],
        root_cause_category=Category.DRAINAGE_FLOOD,
        root_cause_summary="Saluran air mampet tersumbat sampah",
        ticket_id="TKT-FL-101",
    )

    result = memory_engine.lookup("tenant-1", text)
    assert result.hit is True
    assert result.match_type == "EXACT"
    assert result.similarity == 1.0
    assert result.matched_entry is not None
    assert result.matched_entry.case_id == "case-101"
    assert result.is_duplicate_incident is True
    assert result.suggested_action == "LINK_DUPLICATE_TICKET"
    assert result.matched_entry.hit_count == 1


def test_semantic_reuse_without_active_ticket(memory_engine: SemanticCacheEngine) -> None:
    memory_engine.record_case(
        tenant_id="tenant-1",
        case_id="case-102",
        text="jalan berlubang besar aspal rusak di kopo",
        category=Category.ROAD,
        risk=RiskLevel.MEDIUM,
        completeness="SUFFICIENT",
        location_entities=["kopo"],
        root_cause_category=Category.ROAD,
        root_cause_summary="Aspal tergerus hujan",
        ticket_id=None,  # No active ticket yet
    )

    # Similar complaint text
    query = "jalan berlubang besar aspal bolong di kopo"
    result = memory_engine.lookup("tenant-1", query, location_hints=["kopo"])

    assert result.hit is True
    assert result.match_type in ("EXACT", "SEMANTIC")
    assert result.similarity >= 0.70
    assert result.matched_entry is not None
    assert result.matched_entry.category == Category.ROAD
    assert result.matched_entry.root_cause_category == Category.ROAD
    # Since ticket_id was None, it's reused for analysis, not duplicate ticket
    assert result.suggested_action == "REUSE_ANALYSIS"


def test_duplicate_incident_requires_location_overlap(memory_engine: SemanticCacheEngine) -> None:
    memory_engine.record_case(
        tenant_id="tenant-1",
        case_id="case-103",
        text="lampu lalu lintas mati bikin macet parah di jl riau bandung",
        category=Category.TRANSPORTATION,
        risk=RiskLevel.MEDIUM,
        completeness="SUFFICIENT",
        location_entities=["jl riau", "bandung"],
        ticket_id="TKT-TRANS-200",
    )

    # Same issue, same location -> duplicate incident
    res_dup = memory_engine.lookup(
        tenant_id="tenant-1",
        text="lampu lalu lintas mati bikin macet di riau",
        location_hints=["riau"],
    )
    assert res_dup.hit is True
    assert res_dup.is_duplicate_incident is True
    assert res_dup.suggested_action == "LINK_DUPLICATE_TICKET"

    # Same issue, DIFFERENT location -> not a duplicate incident
    res_diff_loc = memory_engine.lookup(
        tenant_id="tenant-1",
        text="lampu lalu lintas mati bikin macet di buah batu",
        location_hints=["buah batu"],
    )
    assert res_diff_loc.is_duplicate_incident is False


def test_unrelated_complaint_is_cache_miss(memory_engine: SemanticCacheEngine) -> None:
    memory_engine.record_case(
        tenant_id="tenant-1",
        case_id="case-104",
        text="kebakaran gudang di kopo dekat perumahan",
        category=Category.FIRE_RESCUE,
        risk=RiskLevel.URGENT,
        completeness="SUFFICIENT",
        location_entities=["kopo"],
        ticket_id="TKT-FIRE-99",
    )

    result = memory_engine.lookup(
        tenant_id="tenant-1",
        text="antrean puskesmas sangat panjang obat habis",
        location_hints=["cikutra"],
    )
    assert result.hit is False
    assert result.match_type == "NONE"
    assert result.suggested_action == "RUN_FULL_PIPELINE"


def test_tenant_isolation(memory_engine: SemanticCacheEngine) -> None:
    memory_engine.record_case(
        tenant_id="tenant-bandung",
        case_id="case-bdg-1",
        text="jalan amblas di pasteur",
        category=Category.ROAD,
        risk=RiskLevel.HIGH,
        completeness="SUFFICIENT",
    )

    # Query from different tenant
    result = memory_engine.lookup("tenant-jakarta", "jalan amblas di pasteur")
    assert result.hit is False


def test_indobert_semantic_similarity_real_model() -> None:
    embedder = IndoBertEmbedder.get_instance()
    store = InMemoryCacheStore()
    engine = SemanticCacheEngine(
        store=store,
        embedder=embedder,
        semantic_threshold=0.88,
        duplicate_threshold=0.85,
    )

    # Record real complaint
    engine.record_case(
        tenant_id="tenant-bdg",
        case_id="case-road-real-1",
        text="lapor min jalan terusan buah batu berlubang parah sering bikin motor jatuh",
        category=Category.ROAD,
        risk=RiskLevel.HIGH,
        completeness="SUFFICIENT",
        location_entities=["terusan buah batu", "bandung"],
        root_cause_category=Category.ROAD,
        root_cause_summary="Jalan rusak berlubang",
        ticket_id="TKT-BB-999",
    )

    # Similar complaint in same location
    query = "min tolong perbaiki jalan buah batu bolong-bolong bahaya kecelakaan"
    res = engine.lookup("tenant-bdg", query, location_hints=["buah batu"])

    assert res.hit is True
    assert res.is_duplicate_incident is True
    assert res.suggested_action == "LINK_DUPLICATE_TICKET"
    assert res.matched_entry.ticket_id == "TKT-BB-999"
    assert res.similarity > 0.85


def test_postgres_cache_store_mock() -> None:
    from unittest.mock import MagicMock
    from services.intelligence.semantic_cache import KnowledgeBankEntry, PostgresCacheStore

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    factory = lambda: mock_conn

    store = PostgresCacheStore(factory)
    entry = KnowledgeBankEntry(
        tenant_id="tenant-1",
        case_id="case-pg-1",
        text_hash="abc123",
        raw_text="lapor jalan",
        embedding=[0.1] * 768,
        category=Category.ROAD,
        risk=RiskLevel.HIGH,
        completeness="SUFFICIENT",
    )
    store.save_entry(entry)
    assert mock_cur.execute.called
    assert mock_conn.commit.called


def test_postgres_cache_store_resilience() -> None:
    from services.intelligence.semantic_cache import KnowledgeBankEntry, PostgresCacheStore

    def failing_conn():
        raise RuntimeError("Database connection failure")

    store = PostgresCacheStore(failing_conn)
    assert store.find_by_hash("tenant-1", "abc123") is None
    assert store.find_similar("tenant-1", [0.1] * 768, 0.8) == []

    entry = KnowledgeBankEntry(
        tenant_id="tenant-1",
        case_id="case-pg-2",
        text_hash="abc123",
        raw_text="lapor jalan",
        embedding=[0.1] * 768,
        category=Category.ROAD,
        risk=RiskLevel.HIGH,
        completeness="SUFFICIENT",
    )
    # Does not throw exception
    store.save_entry(entry)
    store.increment_hit("entry-1")
    store.link_to_cluster("entry-1", "cluster-1")
