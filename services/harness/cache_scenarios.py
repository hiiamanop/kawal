from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from contracts.harness import (
    ComplaintCase,
    CostReport,
    ExtractionOutput,
    HarnessRun,
    SocialPost,
    TriageOutput,
    VerificationOutput,
    deterministic_case_id,
)
from services.harness.agents import (
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.cache import HarnessAnalysisCache
from services.harness.cache_router import CacheFirstRunner
from services.harness.cache_store import InMemoryHarnessCacheStore
from services.harness.location import assess_location
from services.harness.ports import AgentTask, RunOrchestrator
from services.harness.source_intent import assess_source_intent
from services.intelligence.embedder import TextEmbedder


class CacheScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    seed: SocialPost
    query: SocialPost
    expected_reuse: bool
    notes: str = ""


class CacheScenarioSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenarios: tuple[CacheScenario, ...] = Field(min_length=1)


class ScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    expected_reuse: bool
    actual_reuse: bool
    seed_stored: bool
    match_type: str = "NONE"
    similarity: float = 0.0
    decision: str = "REJECTED"
    inner_called: bool = True
    passed: bool
    reason_codes: tuple[str, ...] = ()


class ThresholdMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0.0, le=1.0)
    total: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    results: tuple[ScenarioResult, ...] = ()


class ScenarioReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenarios: int = Field(ge=0)
    thresholds: tuple[ThresholdMetrics, ...] = ()
    best_threshold: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class _RejectingInner:
    def __init__(self) -> None:
        self.called = False

    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun:
        self.called = True
        return HarnessRun(
            run_id=run_id or f"scenario-inner-{post.post_id}",
            post=post,
            status="REJECTED",
            cost=CostReport(),
        )


def _rule_outputs(
    post: SocialPost,
) -> tuple[TriageOutput, ExtractionOutput, VerificationOutput]:
    triage_result = RuleTriageAgent().run(
        AgentTask(role="triage", source_text=post.text)
    )
    extraction_result = RuleExtractionAgent().run(
        AgentTask(role="extraction", source_text=post.text)
    )
    if triage_result.status != "OK" or extraction_result.status != "OK":
        raise ValueError(f"rule prefilter failed for {post.post_id}")
    triage = TriageOutput.model_validate(triage_result.output)
    extraction = ExtractionOutput.model_validate(extraction_result.output)
    verifier_result = RuleVerifierAgent().run(
        AgentTask(
            role="verifier",
            source_text=post.text,
            context={"extraction": extraction.model_dump(mode="json")},
        )
    )
    if verifier_result.status != "OK":
        raise ValueError(f"rule verifier failed for {post.post_id}")
    return triage, extraction, VerificationOutput.model_validate(verifier_result.output)


def evaluate_scenario(
    scenario: CacheScenario,
    *,
    embedder: TextEmbedder,
    threshold: float,
    allow_fallback_semantic: bool = False,
) -> ScenarioResult:
    cache = HarnessAnalysisCache(
        store=InMemoryHarnessCacheStore(),
        embedder=embedder,
        namespace=f"scenario-{scenario.scenario_id}",
        analysis_version="scenario-v1",
        semantic_threshold=threshold,
        allow_fallback_semantic=allow_fallback_semantic,
    )
    seed_triage, seed_extraction, seed_verification = _rule_outputs(scenario.seed)
    seed_location = assess_location(
        seed_extraction.location,
        seed_extraction.evidence_spans,
        scenario.seed.text,
    )
    seed_case = ComplaintCase(
        case_id=deterministic_case_id(scenario.seed),
        post=scenario.seed,
        triage=seed_triage,
        extraction=seed_extraction,
        verification=seed_verification,
        location_quality=seed_location.quality,
        source_intent=assess_source_intent(
            scenario.seed.text, scenario.seed.author_handle
        ).intent,
        trust_decision="AGREED",
        status="ACCEPTED",
    )
    record_result = cache.record(
        case=seed_case,
        rule_triage=seed_triage,
        rule_verification=seed_verification,
        run_id=f"seed-{scenario.scenario_id}",
        location_entities=(seed_extraction.location,)
        if seed_extraction.location
        else (),
        rule_extraction=seed_extraction,
    )

    inner = _RejectingInner()
    runner = CacheFirstRunner(inner=inner, cache=cache)
    run = runner.run(scenario.query, run_id=f"query-{scenario.scenario_id}")
    cache_provenance = run.cache
    actual_reuse = bool(cache_provenance and cache_provenance.reused)
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        expected_reuse=scenario.expected_reuse,
        actual_reuse=actual_reuse,
        seed_stored=record_result.stored,
        match_type=cache_provenance.match_type if cache_provenance else "NONE",
        similarity=cache_provenance.similarity if cache_provenance else 0.0,
        decision=run.status,
        inner_called=inner.called,
        passed=actual_reuse == scenario.expected_reuse,
        reason_codes=cache_provenance.reason_codes if cache_provenance else (),
    )


def evaluate_scenarios(
    scenarios: Sequence[CacheScenario],
    *,
    embedder: TextEmbedder,
    thresholds: Sequence[float],
    allow_fallback_semantic: bool = False,
) -> ScenarioReport:
    metrics: list[ThresholdMetrics] = []
    for threshold in thresholds:
        results = [
            evaluate_scenario(
                scenario,
                embedder=embedder,
                threshold=threshold,
                allow_fallback_semantic=allow_fallback_semantic,
            )
            for scenario in scenarios
        ]
        true_positive = sum(
            result.expected_reuse and result.actual_reuse for result in results
        )
        false_positive = sum(
            not result.expected_reuse and result.actual_reuse for result in results
        )
        false_negative = sum(
            result.expected_reuse and not result.actual_reuse for result in results
        )
        true_negative = sum(
            not result.expected_reuse and not result.actual_reuse for result in results
        )
        total = len(results)
        metrics.append(
            ThresholdMetrics(
                threshold=threshold,
                total=total,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                true_negative=true_negative,
                precision=(
                    true_positive / (true_positive + false_positive)
                    if true_positive + false_positive
                    else 0.0
                ),
                recall=(
                    true_positive / (true_positive + false_negative)
                    if true_positive + false_negative
                    else 0.0
                ),
                accuracy=(true_positive + true_negative) / total if total else 0.0,
                results=tuple(results),
            )
        )
    best = max(
        metrics,
        key=lambda metric: (
            metric.accuracy,
            metric.precision,
            metric.recall,
            metric.threshold,
        ),
        default=None,
    )
    return ScenarioReport(
        scenarios=len(scenarios),
        thresholds=tuple(metrics),
        best_threshold=best.threshold if best is not None else None,
    )


__all__ = [
    "CacheScenario",
    "CacheScenarioSet",
    "ScenarioReport",
    "ScenarioResult",
    "ThresholdMetrics",
    "evaluate_scenario",
    "evaluate_scenarios",
]
