from __future__ import annotations

import pytest

from services.evaluation.stress import (
    LoadProfile,
    run_full_stress_suite,
    run_stress_profile,
)


def test_run_stress_profile_basic() -> None:
    profile = LoadProfile(name="test_small", target_cases=15, concurrent_workers=2)
    result = run_stress_profile(profile)

    assert result.profile_name == "test_small"
    assert result.total_cases == 15
    assert result.successful_cases == 15
    assert result.failed_cases == 0
    assert result.zero_duplicates_invariant is True
    assert result.all_cases_accounted is True
    assert result.cases_per_second > 0
    assert result.messages_per_second > 0

    assert result.latency.p50_ms <= result.latency.p95_ms <= result.latency.p99_ms
    assert result.latency.min_ms <= result.latency.mean_ms <= result.latency.max_ms


def test_run_stress_burst_concurrency_invariants() -> None:
    burst_profile = LoadProfile(
        name="test_burst",
        target_cases=30,
        concurrent_workers=6,
        burst_multiplier=2.0,
    )
    result = run_stress_profile(burst_profile)

    assert result.profile_name == "test_burst"
    assert result.total_cases == 30
    assert result.successful_cases == 30
    assert result.duplicate_tickets == 0
    assert result.zero_duplicates_invariant is True

    # Check that diverse decisions were produced (EXECUTE, CLARIFICATION, REJECT)
    dist = result.decision_distribution
    assert len(dist) >= 2


def test_run_full_stress_suite_scaled() -> None:
    mini_profiles = (
        LoadProfile(name="25%", target_cases=6, concurrent_workers=2),
        LoadProfile(name="50%", target_cases=12, concurrent_workers=3),
    )
    results = run_full_stress_suite(mini_profiles)

    assert len(results) == 2
    assert results[0].profile_name == "25%"
    assert results[1].profile_name == "50%"
    assert all(r.zero_duplicates_invariant for r in results)
    assert all(r.all_cases_accounted for r in results)
