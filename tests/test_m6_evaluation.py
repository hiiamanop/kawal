from __future__ import annotations

import pytest

from contracts.models import Category, DecisionMode, RiskLevel
from services.evaluation.contracts import (
    EvalSample,
    SystemExecutionResult,
    SystemID,
)
from services.evaluation.robustness import run_robustness_invariant_checks
from services.evaluation.runner import (
    generate_standard_thesis_eval_samples,
    run_thesis_evaluation,
)
from services.evaluation.statistics import (
    compute_action_macro_f1,
    compute_paired_cluster_bootstrap,
    compute_single_system_cluster_bootstrap,
    compute_system_metrics,
)
from services.evaluation.systems import execute_system


def _make_dummy_sample(case_id: str, family_id: str, mode: DecisionMode) -> EvalSample:
    return EvalSample(
        case_id=case_id,
        family_id=family_id,
        text="Laporan aduan jalan berlubang di Jl. Dago",
        gold_category=Category.ROAD,
        gold_risk=RiskLevel.MEDIUM,
        gold_completeness="SUFFICIENT",
        gold_action=mode,
    )


def test_action_macro_f1_computation() -> None:
    targets = [DecisionMode.EXECUTE, DecisionMode.REQUEST_CLARIFICATION]
    preds = [DecisionMode.EXECUTE, DecisionMode.REQUEST_CLARIFICATION]
    assert compute_action_macro_f1(preds, targets) == 0.5

    all_targets = [
        DecisionMode.EXECUTE,
        DecisionMode.RE_EVALUATE,
        DecisionMode.REQUEST_CLARIFICATION,
        DecisionMode.REJECT_IGNORE,
    ]
    assert compute_action_macro_f1(all_targets, all_targets) == 1.0


def test_single_system_cluster_bootstrap() -> None:
    samples = [
        _make_dummy_sample("c1", "fam1", DecisionMode.EXECUTE),
        _make_dummy_sample("c2", "fam1", DecisionMode.EXECUTE),
        _make_dummy_sample("c3", "fam2", DecisionMode.REQUEST_CLARIFICATION),
        _make_dummy_sample("c4", "fam2", DecisionMode.REQUEST_CLARIFICATION),
    ]
    results = [
        SystemExecutionResult(
            case_id="c1", family_id="fam1", system_id=SystemID.P,
            predicted_mode=DecisionMode.EXECUTE, predicted_category=Category.ROAD,
            predicted_risk=RiskLevel.MEDIUM, is_correct_action=True,
            is_prohibited_action_attempted=False, is_prohibited_action_passed=False,
            cost_usd=0.001, latency_ms=50.0, egress_bytes=0, re_evaluation_count=0,
        ),
        SystemExecutionResult(
            case_id="c2", family_id="fam1", system_id=SystemID.P,
            predicted_mode=DecisionMode.EXECUTE, predicted_category=Category.ROAD,
            predicted_risk=RiskLevel.MEDIUM, is_correct_action=True,
            is_prohibited_action_attempted=False, is_prohibited_action_passed=False,
            cost_usd=0.001, latency_ms=50.0, egress_bytes=0, re_evaluation_count=0,
        ),
        SystemExecutionResult(
            case_id="c3", family_id="fam2", system_id=SystemID.P,
            predicted_mode=DecisionMode.REQUEST_CLARIFICATION, predicted_category=Category.ROAD,
            predicted_risk=RiskLevel.MEDIUM, is_correct_action=True,
            is_prohibited_action_attempted=False, is_prohibited_action_passed=False,
            cost_usd=0.001, latency_ms=50.0, egress_bytes=0, re_evaluation_count=0,
        ),
        SystemExecutionResult(
            case_id="c4", family_id="fam2", system_id=SystemID.P,
            predicted_mode=DecisionMode.REQUEST_CLARIFICATION, predicted_category=Category.ROAD,
            predicted_risk=RiskLevel.MEDIUM, is_correct_action=True,
            is_prohibited_action_attempted=False, is_prohibited_action_passed=False,
            cost_usd=0.001, latency_ms=50.0, egress_bytes=0, re_evaluation_count=0,
        ),
    ]

    ci = compute_single_system_cluster_bootstrap(samples, results, "cost_usd", n_resamples=500, seed=42)
    assert ci.metric == "cost_usd"
    assert ci.confidence_level == 0.95
    assert ci.ci_lower <= ci.mean <= ci.ci_upper


def test_paired_cluster_bootstrap_detects_significant_difference() -> None:
    samples = [
        _make_dummy_sample("c1", "fam1", DecisionMode.EXECUTE),
        _make_dummy_sample("c2", "fam2", DecisionMode.REQUEST_CLARIFICATION),
    ]
    res_a = [
        SystemExecutionResult(
            case_id="c1", family_id="fam1", system_id=SystemID.P,
            predicted_mode=DecisionMode.EXECUTE, predicted_category=Category.ROAD,
            predicted_risk=RiskLevel.MEDIUM, is_correct_action=True,
            is_prohibited_action_attempted=False, is_prohibited_action_passed=False,
            cost_usd=0.001, latency_ms=50.0, egress_bytes=0, re_evaluation_count=0,
        ),
        SystemExecutionResult(
            case_id="c2", family_id="fam2", system_id=SystemID.P,
            predicted_mode=DecisionMode.REQUEST_CLARIFICATION, predicted_category=Category.ROAD,
            predicted_risk=RiskLevel.MEDIUM, is_correct_action=True,
            is_prohibited_action_attempted=False, is_prohibited_action_passed=False,
            cost_usd=0.001, latency_ms=50.0, egress_bytes=0, re_evaluation_count=0,
        ),
    ]
    res_b = [
        SystemExecutionResult(
            case_id="c1", family_id="fam1", system_id=SystemID.B3,
            predicted_mode=DecisionMode.REJECT_IGNORE, predicted_category=Category.ROAD,
            predicted_risk=RiskLevel.MEDIUM, is_correct_action=False,
            is_prohibited_action_attempted=False, is_prohibited_action_passed=False,
            cost_usd=0.005, latency_ms=100.0, egress_bytes=0, re_evaluation_count=0,
        ),
        SystemExecutionResult(
            case_id="c2", family_id="fam2", system_id=SystemID.B3,
            predicted_mode=DecisionMode.REJECT_IGNORE, predicted_category=Category.ROAD,
            predicted_risk=RiskLevel.MEDIUM, is_correct_action=False,
            is_prohibited_action_attempted=False, is_prohibited_action_passed=False,
            cost_usd=0.005, latency_ms=100.0, egress_bytes=0, re_evaluation_count=0,
        ),
    ]

    paired = compute_paired_cluster_bootstrap(samples, res_a, res_b, "action_macro_f1", n_resamples=500, seed=42)
    assert paired.system_a == SystemID.P
    assert paired.system_b == SystemID.B3
    assert paired.diff_mean > 0.0
    assert paired.ci_lower > 0.0
    assert paired.significant_95 is True


def test_all_systems_execute_sample() -> None:
    sample = _make_dummy_sample("case-all", "fam-01", DecisionMode.EXECUTE)
    for sys_id in SystemID:
        res = execute_system(sample, sys_id)
        assert res.system_id == sys_id
        assert res.case_id == "case-all"
        assert res.cost_usd >= 0.0
        assert res.latency_ms > 0.0


def test_robustness_invariants_all_pass() -> None:
    invariants = run_robustness_invariant_checks()
    assert len(invariants) == 5
    assert invariants["zero_duplicate_tickets"] is True
    assert invariants["hard_policy_fail_closed"] is True
    assert invariants["audit_lineage_reproducibility"] is True
    assert invariants["circuit_breaker_containment"] is True
    assert invariants["dlq_quarantine_integrity"] is True


def test_run_thesis_evaluation_fast() -> None:
    samples = generate_standard_thesis_eval_samples(num_families=6, samples_per_family=8)
    report = run_thesis_evaluation(samples=samples, n_bootstrap_resamples=200, seed=42)

    assert report.total_cases == 48
    assert report.total_families == 6
    assert len(report.systems_evaluated) == 11
    assert all(h.passed for h in report.hypotheses.values())
    assert all(report.robustness_invariants.values())
