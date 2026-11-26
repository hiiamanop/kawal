from __future__ import annotations

from pathlib import Path

from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.cache import HarnessAnalysisCache
from services.harness.cache_router import CacheFirstRunner
from services.harness.cache_store import InMemoryHarnessCacheStore
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sources import ReplayXSource
from services.intelligence.embedder import DeterministicFallbackEmbedder


FIXTURE = (
    Path(__file__).parent / "fixtures" / "x_posts" / "synthetic_road_bandung_001.json"
)


def test_cache_e2e_two_passes() -> None:
    post = ReplayXSource(FIXTURE).fetch()[0]
    cache = HarnessAnalysisCache(
        store=InMemoryHarnessCacheStore(),
        embedder=DeterministicFallbackEmbedder(dim=768),
        namespace="harness-x-e2e",
        analysis_version="m6-e2e-v1",
    )
    inner = HarnessOrchestrator(
        triage=RuleTriageAgent(),
        extraction=RuleExtractionAgent(),
        verifier=RuleVerifierAgent(),
        arbiter=RuleArbiterAgent(),
        max_cost_usd=0.05,
    )
    runner = CacheFirstRunner(inner=inner, cache=cache)

    first = runner.run(post)
    assert first.cache is not None
    assert first.cache.recorded is True

    second = runner.run(post)
    assert second.cache is not None
    assert second.cache.match_type == "EXACT"
    assert second.cache.reused is True
    assert second.cost.total_cost_usd == 0.0
    assert second.status == "COMPLETED"
