from __future__ import annotations

from datetime import datetime, timezone

from contracts.harness import (
    CacheProvenance,
    CostReport,
    HarnessRun,
    SocialPost,
    TraceEvent,
)
from services.harness.batch import HarnessBatchRunner


def _post() -> SocialPost:
    return SocialPost(
        post_id="cache-provenance-post",
        platform="x",
        text="Jalan rusak di Kopo",
        collected_at=datetime(2026, 11, 26, tzinfo=timezone.utc),
    )


def _run(
    *,
    cache: CacheProvenance | None,
    trace: tuple[TraceEvent, ...] = (),
) -> HarnessRun:
    return HarnessRun(
        run_id="run-cache-provenance",
        post=_post(),
        status="COMPLETED",
        cost=CostReport(total_cost_usd=0.0),
        cache=cache,
        trace=trace,
    )


def test_harness_run_accepts_cache_provenance() -> None:
    run = _run(
        cache=CacheProvenance(
            attempted=True,
            hit=True,
            reused=True,
            match_type="EXACT",
            similarity=1.0,
            recorded=True,
        )
    )
    assert run.cache is not None
    assert run.cache.reused is True
    assert run.cache.recorded is True


def test_batch_summary_copies_cache_fields() -> None:
    summary = HarnessBatchRunner._summarize(
        _post(),
        _run(
            cache=CacheProvenance(
                attempted=True,
                hit=True,
                reused=True,
                match_type="SEMANTIC",
                similarity=0.95,
            )
        ),
    )
    assert summary.cache_attempted is True
    assert summary.cache_hit is True
    assert summary.cache_reused is True
    assert summary.cache_match_type == "SEMANTIC"
    assert summary.cache_similarity == 0.95


def test_batch_summary_defaults_to_no_cache() -> None:
    summary = HarnessBatchRunner._summarize(_post(), _run(cache=None))
    assert summary.cache_attempted is False
    assert summary.cache_hit is False
    assert summary.cache_match_type is None
    assert summary.cache_similarity is None


def test_batch_summary_counts_only_remote_agent_calls() -> None:
    run = _run(
        cache=None,
        trace=(
            TraceEvent(event="agent.completed", data={"provider": "openai"}),
            TraceEvent(event="agent.completed", data={"provider": "local"}),
            TraceEvent(event="cache.lookup", data={}),
        ),
    )
    summary = HarnessBatchRunner._summarize(_post(), run)
    assert summary.llm_calls == 1
