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


def _post(post_id: str, text: str = "Jalan rusak di Kopo") -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform="x",
        text=text,
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
    *,
    reused: bool,
    status: str = "COMPLETED",
    cost: float = 0.0,
    post_id: str = "cache-hit",
    text: str = "Jalan rusak di Kopo",
) -> HarnessRun:
    return HarnessRun(
        run_id="cache-eval",
        post=_post(post_id, text),
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


def test_cache_metrics_reports_accepted_precision() -> None:
    report = score_cache_evaluation(
        [_warmup_run(cost=0.0, recorded=True)],
        [_cache_run(reused=True)],
        _labels(),
        baseline_runs=[_warmup_run(cost=0.002, recorded=False)],
    )
    assert report.baseline.accepted_precision == 1.0
    assert report.cached.accepted_precision == 1.0


def test_score_cache_evaluation_requires_explicit_baseline() -> None:
    with pytest.raises(TypeError):
        score_cache_evaluation([], [], _labels())  # type: ignore[call-arg]


def test_cache_metrics_counts_hits_and_cost_saved() -> None:
    report = score_cache_evaluation(
        [_warmup_run(cost=0.0, recorded=True)],
        [_cache_run(reused=True)],
        _labels(),
        baseline_runs=[_warmup_run(cost=0.002, recorded=False)],
    )
    assert report.cache_eligible == 1
    assert report.exact_hits == 1
    assert report.cache_eligible_hit_rate == 1.0
    assert report.overall_hit_rate == 1.0
    assert report.cost_saved_usd == pytest.approx(0.002)
    assert report.llm_calls_saved == 1


def test_distinct_post_ids_with_same_text_do_not_exceed_hit_rate() -> None:
    labels = LabelSet(
        labels={
            "cache-hit": FixtureLabel(
                is_complaint=True,
                location_quality="SPECIFIC",
                acceptable_statuses=("COMPLETED",),
            ),
            "cache-repost": FixtureLabel(
                is_complaint=True,
                location_quality="SPECIFIC",
                acceptable_statuses=("COMPLETED",),
            ),
        }
    )
    report = score_cache_evaluation(
        [_warmup_run(cost=0.0, recorded=True)],
        [
            _cache_run(reused=True),
            _cache_run(reused=True, post_id="cache-repost"),
        ],
        labels,
        baseline_runs=[_warmup_run(cost=0.002, recorded=False)],
    )
    assert report.cache_eligible == 2
    assert report.cache_eligible_hit_rate == 1.0
    assert 0.0 <= report.cache_eligible_hit_rate <= 1.0


def test_explicit_baseline_is_used_for_savings() -> None:
    report = score_cache_evaluation(
        [_warmup_run(cost=0.0, recorded=True)],
        [_cache_run(reused=True)],
        _labels(),
        baseline_runs=[_warmup_run(cost=0.004, recorded=False)],
    )
    assert report.cost_saved_usd == pytest.approx(0.004)
    assert report.llm_calls_saved == 1


def test_overall_and_eligible_hit_rates_differ() -> None:
    report = score_cache_evaluation(
        [_warmup_run(cost=0.0, recorded=True)],
        [
            _cache_run(reused=True),
            _cache_run(
                reused=False,
                post_id="not-cacheable",
                text="Informasi cuaca cerah hari ini",
            ),
        ],
        _labels(),
        baseline_runs=[],
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
    report = score_cache_evaluation(
        [], [_cache_run(reused=True)], labels, baseline_runs=[]
    )
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
    report = score_cache_evaluation(
        [], [_cache_run(reused=False)], labels, baseline_runs=[]
    )
    assert report.false_reuse == 0
