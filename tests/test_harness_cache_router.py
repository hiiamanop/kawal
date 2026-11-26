from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contracts.harness import (
    ComplaintCase,
    CostReport,
    ExtractionOutput,
    HarnessRun,
    SocialPost,
    TriageOutput,
    VerificationOutput,
)
from contracts.models import Category, RiskLevel
from services.harness.agents import (
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.cache import (
    HarnessAnalysisCache,
    compute_cache_key,
    compute_text_hash,
)
from services.harness.cache_router import CacheFirstRunner, locations_overlap
from services.harness.cache_store import InMemoryHarnessCacheStore
from services.harness.orchestrator import HarnessOrchestrator
from services.intelligence.embedder import DeterministicFallbackEmbedder


TEXT = "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar. Tolong segera diperbaiki."


def _post(text: str = TEXT) -> SocialPost:
    return SocialPost(
        post_id="router-post",
        platform="x",
        url="https://x.com/example/status/router",
        author_handle="warga",
        text=text,
        collected_at=datetime(2026, 11, 26, tzinfo=timezone.utc),
    )


def _case(
    *,
    post: SocialPost | None = None,
    risk: RiskLevel = RiskLevel.HIGH,
    category: Category = Category.ROAD,
) -> ComplaintCase:
    post = post or _post()
    return ComplaintCase(
        case_id="case-router-1",
        post=post,
        triage=TriageOutput(
            is_complaint=True,
            category=category,
            risk=risk,
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
        location_quality="SPECIFIC",
        source_intent="CITIZEN_COMPLAINT",
        trust_decision="AGREED",
        status="ACCEPTED",
    )


def _cache() -> HarnessAnalysisCache:
    return HarnessAnalysisCache(
        store=InMemoryHarnessCacheStore(),
        embedder=DeterministicFallbackEmbedder(dim=768),
        namespace="harness-x",
        analysis_version="v1",
        semantic_threshold=0.0,
        allow_fallback_semantic=True,
    )


def _seed(
    cache: HarnessAnalysisCache, case: ComplaintCase | None = None
) -> ComplaintCase:
    case = case or _case()
    result = cache.record(
        case=case,
        rule_triage=case.triage,
        rule_verification=case.verification,
        run_id="run-seed",
        location_entities=("Kopo",),
    )
    assert result.stored is True
    return case


class ExplodingInner:
    def __init__(self) -> None:
        self.called = False

    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun:
        del post, run_id
        self.called = True
        raise AssertionError("inner orchestrator must not run on a cache hit")


class RecordingInner:
    def __init__(
        self, case: ComplaintCase | None = None, status: str = "COMPLETED"
    ) -> None:
        self.called = False
        self._case = case
        self._status = status

    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun:
        self.called = True
        case = self._case or _case(post=post)
        return HarnessRun(
            run_id=run_id or "inner-run",
            post=post,
            status=self._status,  # type: ignore[arg-type]
            case=case if self._status == "COMPLETED" else None,
            cost=CostReport(total_cost_usd=0.001),
        )


def _runner(inner: object, cache: HarnessAnalysisCache) -> CacheFirstRunner:
    return CacheFirstRunner(
        inner=inner,  # type: ignore[arg-type]
        cache=cache,
        rule_triage=RuleTriageAgent(),
        rule_extraction=RuleExtractionAgent(),
        rule_verifier=RuleVerifierAgent(),
    )


def test_generic_place_tokens_do_not_overlap() -> None:
    assert locations_overlap("Stasiun Bandung", ["stasiun cimahi"]) is False
    assert locations_overlap("Stasiun Bandung", ["stasiun bandung"]) is True


def test_street_prefixes_are_normalized() -> None:
    assert locations_overlap("Jl. Merdeka", ["jalan merdeka"]) is True
    assert locations_overlap("Terusan Buah Batu", ["buah batu"]) is True


def test_exact_cache_hit_skips_inner_orchestrator() -> None:
    cache = _cache()
    case = _seed(cache)
    inner = ExplodingInner()
    run = _runner(inner, cache).run(case.post)
    assert inner.called is False
    assert run.status == "COMPLETED"
    assert run.cache is not None
    assert run.cache.reused is True
    assert run.cache.match_type == "EXACT"
    assert run.cost.total_cost_usd == 0.0


def test_semantic_cache_hit_skips_inner_orchestrator() -> None:
    cache = _cache()
    _seed(cache)
    inner = ExplodingInner()
    run = _runner(inner, cache).run(_post(f"{TEXT} ya"))
    assert inner.called is False
    assert run.cache is not None
    assert run.cache.reused is True
    assert run.cache.match_type == "SEMANTIC"


def test_location_mismatch_falls_back() -> None:
    cache = _cache()
    _seed(cache)
    inner = RecordingInner()
    run = _runner(inner, cache).run(
        _post("Min, jalan di Jl. Cimahi rusak parah, ada lubang besar.")
    )
    assert inner.called is True
    assert run.cache is not None
    assert run.cache.reused is False


def test_source_intent_mismatch_falls_back() -> None:
    cache = _cache()
    _seed(cache)
    inner = RecordingInner()
    run = _runner(inner, cache).run(
        _post("Dilansir, jalan rusak di Kopo memaksa warga mencari jalur alternatif.")
    )
    assert inner.called is True
    assert run.cache is not None
    assert run.cache.reused is False


def test_rule_category_mismatch_falls_back() -> None:
    cache = _cache()
    _seed(cache)
    inner = RecordingInner()
    run = _runner(inner, cache).run(
        _post("Min, banjir di Jl. Kopo Bandung, air tergenang parah.")
    )
    assert inner.called is True
    assert run.cache is not None
    assert run.cache.reused is False


def test_exact_evidence_mismatch_falls_back() -> None:
    cache = _cache()
    case = _case()
    _seed(cache, case)
    query_text = "Min, jalan di Jl. Kopo Bandung rusak parah."
    query_hash = compute_text_hash(query_text)
    entry = cache.store.find_exact(
        "harness-x", "v1", cache.embedding_model, compute_text_hash(TEXT)
    )
    assert entry is not None
    cache.store.save(
        entry.model_copy(
            update={
                "text_hash": query_hash,
                "cache_key": compute_cache_key(
                    "harness-x", "v1", cache.embedding_model, query_hash
                ),
            }
        )
    )
    inner = RecordingInner()
    run = _runner(inner, cache).run(_post(query_text))
    assert inner.called is True
    assert run.cache is not None
    assert run.cache.reused is False
    assert "CACHE_EVIDENCE_MISMATCH" in run.cache.reason_codes


def test_cached_hit_elevates_risk_from_current_rule_triage() -> None:
    cache = _cache()
    _seed(cache, _case(risk=RiskLevel.MEDIUM))
    run = _runner(ExplodingInner(), cache).run(_post())
    assert run.case is not None
    assert run.case.triage.risk == RiskLevel.HIGH


def test_inner_accepted_run_is_recorded() -> None:
    cache = _cache()
    inner = RecordingInner()
    post = _post()
    run = _runner(inner, cache).run(post)
    assert inner.called is True
    assert run.cache is not None
    assert run.cache.recorded is True
    assert cache.store.count == 1


def test_inner_rejected_run_is_not_recorded() -> None:
    cache = _cache()
    inner = RecordingInner(status="REJECTED")
    run = _runner(inner, cache).run(_post())
    assert run.cache is not None
    assert run.cache.recorded is False
    assert cache.store.count == 0
