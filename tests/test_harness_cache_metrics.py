from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contracts.harness import (
    CacheProvenance,
    CostReport,
    HarnessRun,
    SocialPost,
    TraceEvent,
)
from services.harness.cache_metrics import score_cache_evaluation
from services.harness.metrics import FixtureLabel, LabelSet


def _post(post_id: str) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform="x",
        text="Jalan rusak di Kopo",
        collected_at=datetime(2026, 11, 26, tzinfo=timezone.utc),
    )


def _labels(
    *, is_complaint: bool = True, acceptable: tuple[str, ...] = ("COMPLETED",)
) -> LabelSet:
    return LabelSet(
        labels={
            "cache-hit": FixtureLabel(
                is_complaint=is_complaint,
                location_quality="SPECIFIC",
                acceptable_statuses=acceptable,
            )
        }
    )


def _warmup_run(*, cost: float, recorded: bool) -> HarnessRun:
    return HarnessRun(
        run_id="warmup",
        post=_post("cache-hit"),
        status="COMPLETED",
        cost=CostReport(total_cost_usd=cost, total_latency_ms=100.0, calls=3),
        trace=(
            TraceEvent(event="agent.completed", data={"provider": "openai"}),
            TraceEvent(event="agent.completed", data={"provider": "local"}),
        ),
        cache=CacheProvenance(attempted=True, recorded=recorded),
    )


def _cache_run(
    *, reused: bool, status: str = "COMPLETED", cost: float = 0.0
) -> HarnessRun:
    return HarnessRun(
        run_id="cache-eval",
        post=_post("cache-hit"),
        status=status,  # type: ignore[arg-type]
        cost=CostReport(total_cost_usd=cost, total_latency_ms=1.0, calls=0),
        cache=CacheProvenance(
            attempted=True,
            hit=reused,
            reused=reused,
            match_type="EXACT" if reused else "NONE",
            similarity=1.0 if reused else 0.0,
        ),
    )


def test_cache_metrics_counts_hits_and_cost_saved() -> None:
    report = score_cache_evaluation(
        [_warmup_run(cost=0.002, recorded=True)],
        [_cache_run(reused=True)],
        _labels(),
    )
    assert report.cache_eligible == 1
    assert report.exact_hits == 1
    assert report.cache_eligible_hit_rate == 1.0
    assert report.overall_hit_rate == 1.0
    assert report.cost_saved_usd == pytest.approx(0.002)
    assert report.llm_calls_saved == 1


def test_overall_and_eligible_hit_rates_differ() -> None:
    report = score_cache_evaluation(
        [_warmup_run(cost=0.002, recorded=True)],
        [_cache_run(reused=True), _cache_run(reused=False)],
        _labels(),
    )
    assert report.cache_eligible == 1
    assert report.overall_hit_rate == 0.5
    assert report.cache_eligible_hit_rate == 1.0


def test_false_reuse_counts_only_reused_runs() -> None:
    labels = LabelSet(
        labels={
            "cache-hit": FixtureLabel(
                is_complaint=False,
                location_quality="MISSING",
                acceptable_statuses=("REJECTED",),
            )
        }
    )
    report = score_cache_evaluation([], [_cache_run(reused=True)], labels)
    assert report.false_reuse == 1
    assert report.false_reuse_rate == 1.0


def test_completed_unreused_run_is_not_false_reuse() -> None:
    labels = LabelSet(
        labels={
            "cache-hit": FixtureLabel(
                is_complaint=False,
                location_quality="MISSING",
                acceptable_statuses=("REJECTED",),
            )
        }
    )
    report = score_cache_evaluation([], [_cache_run(reused=False)], labels)
    assert report.false_reuse == 0
