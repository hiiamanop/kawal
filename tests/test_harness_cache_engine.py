from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from contracts.harness import (
    ComplaintCase,
    ExtractionOutput,
    SocialPost,
    TriageOutput,
    VerificationOutput,
)
from contracts.models import Category, RiskLevel
from services.harness.cache import HarnessAnalysisCache, HarnessCacheStoreError
from services.harness.cache_store import InMemoryHarnessCacheStore
from services.intelligence.embedder import DeterministicFallbackEmbedder


TEXT = "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar."


def _post(text: str = TEXT) -> SocialPost:
    return SocialPost(
        post_id="cache-post-1",
        platform="x",
        url="https://x.com/example/status/1",
        author_handle="warga",
        text=text,
        collected_at=datetime(2026, 11, 26, tzinfo=timezone.utc),
    )


def _case(*, text: str = TEXT, status: str = "ACCEPTED") -> ComplaintCase:
    triage = TriageOutput(
        is_complaint=True,
        category=Category.ROAD,
        risk=RiskLevel.HIGH,
        confidence=0.9,
        reason_codes=("ROAD_DAMAGE",),
    )
    extraction = ExtractionOutput(
        location="Kopo",
        object="jalan rusak",
        time=None,
        evidence_spans=("Jl. Kopo", "lubang"),
        confidence=0.9,
    )
    verification = VerificationOutput(
        is_complaint=True,
        category=Category.ROAD,
        evidence_supported=True,
        issues=(),
        confidence=0.85,
    )
    return ComplaintCase(
        case_id="case-cache-1",
        post=_post(text),
        triage=triage,
        extraction=extraction,
        verification=verification,
        location_quality="SPECIFIC",
        source_intent="CITIZEN_COMPLAINT",
        trust_decision="AGREED",
        status=status,  # type: ignore[arg-type]
    )


def _rule_outputs() -> SimpleNamespace:
    return SimpleNamespace(
        triage=_case().triage,
        verification=_case().verification,
    )


def _cache(
    *, version: str = "v1", allow_fallback_semantic: bool = False
) -> HarnessAnalysisCache:
    return HarnessAnalysisCache(
        store=InMemoryHarnessCacheStore(),
        embedder=DeterministicFallbackEmbedder(dim=768),
        namespace="harness-x",
        analysis_version=version,
        semantic_threshold=0.5,
        allow_fallback_semantic=allow_fallback_semantic,
    )


def test_record_stores_accepted_rule_consistent_case() -> None:
    cache = _cache()
    case = _case()
    result = cache.record(
        case=case,
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    assert result.stored is True
    assert result.entry is not None
    assert result.entry.analysis.triage.category == Category.ROAD
    lookup = cache.lookup(TEXT, Category.ROAD, "CITIZEN_COMPLAINT")
    assert lookup.hit is True
    assert lookup.match_type == "EXACT"


def test_record_rejects_rule_category_disagreement() -> None:
    cache = _cache()
    rule = _rule_outputs()
    result = cache.record(
        case=_case(),
        rule_triage=rule.triage.model_copy(
            update={"category": Category.DRAINAGE_FLOOD}
        ),
        rule_verification=rule.verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    assert result.stored is False
    assert "CACHE_RULE_DISAGREEMENT" in result.reason_codes


def test_record_rejects_pii() -> None:
    cache = _cache()
    result = cache.record(
        case=_case(text="NIK 3273010101010001, jalan rusak di Kopo"),
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    assert result.stored is False
    assert "CACHE_PII_BLOCKED" in result.reason_codes


def test_semantic_lookup_is_disabled_for_fallback_by_default() -> None:
    cache = _cache()
    case = _case()
    cache.record(
        case=case,
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    result = cache.lookup(
        "jalan rusak parah di Kopo ya", Category.ROAD, "CITIZEN_COMPLAINT"
    )
    assert result.hit is False
    assert "SEMANTIC_DISABLED_FALLBACK" in result.reason_codes


def test_semantic_lookup_works_when_fallback_is_explicitly_enabled() -> None:
    cache = _cache(allow_fallback_semantic=True)
    case = _case()
    cache.record(
        case=case,
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    result = cache.lookup(f"{TEXT} ya", Category.ROAD, "CITIZEN_COMPLAINT")
    assert result.hit is True
    assert result.match_type == "SEMANTIC"


def test_lookup_filters_analysis_version() -> None:
    cache = _cache(version="v1")
    case = _case()
    cache.record(
        case=case,
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    other = HarnessAnalysisCache(
        store=cache.store,
        embedder=DeterministicFallbackEmbedder(dim=768),
        namespace="harness-x",
        analysis_version="v2",
        semantic_threshold=0.5,
        allow_fallback_semantic=True,
    )
    assert other.lookup(TEXT, Category.ROAD, "CITIZEN_COMPLAINT").hit is False


def test_record_rejects_full_body_in_evidence_spans() -> None:
    cache = _cache()
    case = _case()
    case = case.model_copy(
        update={
            "extraction": case.extraction.model_copy(update={"evidence_spans": (TEXT,)})
        }
    )
    result = cache.record(
        case=case,
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    assert result.stored is False
    assert "CACHE_FULL_TEXT_BLOCKED" in result.reason_codes


def test_recorded_entry_does_not_contain_full_post_text() -> None:
    cache = _cache()
    case = _case()
    result = cache.record(
        case=case,
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    assert result.stored is True
    assert result.entry is not None
    assert TEXT not in result.entry.model_dump_json()


class FailingStore(InMemoryHarnessCacheStore):
    def save(self, entry) -> None:
        raise HarnessCacheStoreError("CACHE_STORE_WRITE_FAILED", "database down")


def test_record_reports_store_write_failure() -> None:
    cache = HarnessAnalysisCache(
        store=FailingStore(),
        embedder=DeterministicFallbackEmbedder(dim=768),
        namespace="harness-x",
        analysis_version="v1",
    )
    result = cache.record(
        case=_case(),
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    assert result.stored is False
    assert "CACHE_STORE_WRITE_FAILED" in result.reason_codes


def test_record_rejects_accepted_case_with_non_complaint_outputs() -> None:
    cache = _cache()
    case = _case()
    case = case.model_copy(
        update={
            "triage": case.triage.model_copy(update={"is_complaint": False}),
            "verification": case.verification.model_copy(
                update={"is_complaint": False}
            ),
        }
    )
    result = cache.record(
        case=case,
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    assert result.stored is False
    assert "CACHE_NOT_COMPLAINT" in result.reason_codes


def test_record_rejects_needs_review_case() -> None:
    cache = _cache()
    result = cache.record(
        case=_case(status="NEEDS_REVIEW"),
        rule_triage=_rule_outputs().triage,
        rule_verification=_rule_outputs().verification,
        run_id="run-1",
        location_entities=("Kopo",),
    )
    assert result.stored is False
    assert "CACHE_NOT_ACCEPTED" in result.reason_codes
