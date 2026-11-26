from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from contracts.harness import HarnessRun
from services.harness.cache import compute_text_hash
from services.harness.metrics import LabelSet


class CacheModeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int = Field(ge=0)
    accepted: int = Field(ge=0)
    decision_correct: int = Field(ge=0)
    decision_accuracy: float = Field(ge=0.0, le=1.0)
    false_accepts: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    llm_calls: int = Field(ge=0)


class CacheEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: str = ""
    analysis_version: str = ""
    threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    total_posts: int = Field(ge=0)
    cache_eligible: int = Field(ge=0)
    exact_hits: int = Field(ge=0)
    semantic_hits: int = Field(ge=0)
    rejected_hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    overall_hit_rate: float = Field(ge=0.0, le=1.0)
    cache_eligible_hit_rate: float = Field(ge=0.0, le=1.0)
    false_reuse: int = Field(ge=0)
    false_reuse_rate: float = Field(ge=0.0, le=1.0)
    baseline: CacheModeMetrics
    cached: CacheModeMetrics
    llm_calls_saved: int
    cost_saved_usd: float
    latency_saved_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _llm_calls(run: HarnessRun) -> int:
    return sum(
        1
        for event in run.trace
        if event.event == "agent.completed" and event.data.get("provider") != "local"
    )


def _score_mode(runs: list[HarnessRun], labels: LabelSet) -> CacheModeMetrics:
    accepted = 0
    correct = 0
    false_accepts = 0
    cost = 0.0
    latency = 0.0
    calls = 0

    for run in runs:
        cost += run.cost.total_cost_usd
        latency += run.cost.total_latency_ms
        calls += _llm_calls(run)
        label = labels.labels.get(run.post.post_id) if run.post is not None else None
        if run.status == "COMPLETED":
            accepted += 1
        if label is not None and run.status in label.acceptable_statuses:
            correct += 1
        if run.status == "COMPLETED" and (
            label is None or "COMPLETED" not in label.acceptable_statuses
        ):
            false_accepts += 1

    total = len(runs)
    return CacheModeMetrics(
        total=total,
        accepted=accepted,
        decision_correct=correct,
        decision_accuracy=(correct / total) if total else 0.0,
        false_accepts=false_accepts,
        total_cost_usd=cost,
        total_latency_ms=latency,
        llm_calls=calls,
    )


def score_cache_evaluation(
    warmup_runs: list[HarnessRun],
    eval_runs: list[HarnessRun],
    labels: LabelSet,
    *,
    baseline_runs: list[HarnessRun],
    namespace: str = "",
    analysis_version: str = "",
    threshold: float = 0.92,
) -> CacheEvaluationReport:
    recorded_text_hashes = {
        compute_text_hash(run.post.text)
        for run in warmup_runs
        if run.post is not None and run.cache is not None and run.cache.recorded
    }
    cache_eligible = sum(
        1
        for run in eval_runs
        if run.post is not None
        and compute_text_hash(run.post.text) in recorded_text_hashes
    )
    exact_hits = 0
    semantic_hits = 0
    rejected_hits = 0
    misses = 0
    false_reuse = 0
    reused_count = 0

    for run in eval_runs:
        cache = run.cache
        if cache is None or not cache.attempted:
            misses += 1
            continue
        if cache.reused:
            reused_count += 1
            if cache.match_type == "EXACT":
                exact_hits += 1
            elif cache.match_type == "SEMANTIC":
                semantic_hits += 1
            label = (
                labels.labels.get(run.post.post_id) if run.post is not None else None
            )
            if label is None or run.status not in label.acceptable_statuses:
                false_reuse += 1
        elif cache.hit:
            rejected_hits += 1
        else:
            misses += 1

    total = len(eval_runs)
    baseline = _score_mode(baseline_runs, labels)
    cached = _score_mode(eval_runs, labels)
    return CacheEvaluationReport(
        namespace=namespace,
        analysis_version=analysis_version,
        threshold=threshold,
        total_posts=total,
        cache_eligible=cache_eligible,
        exact_hits=exact_hits,
        semantic_hits=semantic_hits,
        rejected_hits=rejected_hits,
        misses=misses,
        overall_hit_rate=(reused_count / total) if total else 0.0,
        cache_eligible_hit_rate=min(
            1.0, (reused_count / cache_eligible) if cache_eligible else 0.0
        ),
        false_reuse=false_reuse,
        false_reuse_rate=(false_reuse / reused_count) if reused_count else 0.0,
        baseline=baseline,
        cached=cached,
        llm_calls_saved=baseline.llm_calls - cached.llm_calls,
        cost_saved_usd=baseline.total_cost_usd - cached.total_cost_usd,
        latency_saved_ms=baseline.total_latency_ms - cached.total_latency_ms,
    )
