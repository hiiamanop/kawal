from __future__ import annotations

import collections
from typing import Sequence
import numpy as np

from contracts.models import DecisionMode
from services.evaluation.contracts import (
    BootstrapCI,
    EvalSample,
    PairedDiffResult,
    SystemExecutionResult,
    SystemID,
    SystemMetrics,
)

ALL_DECISION_MODES = (
    DecisionMode.EXECUTE,
    DecisionMode.RE_EVALUATE,
    DecisionMode.REQUEST_CLARIFICATION,
    DecisionMode.REJECT_IGNORE,
)


def compute_action_macro_f1(
    predictions: Sequence[DecisionMode],
    targets: Sequence[DecisionMode],
) -> float:
    f1_scores = []
    for mode in ALL_DECISION_MODES:
        tp = sum(1 for p, t in zip(predictions, targets) if p == mode and t == mode)
        fp = sum(1 for p, t in zip(predictions, targets) if p == mode and t != mode)
        fn = sum(1 for p, t in zip(predictions, targets) if p != mode and t == mode)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def compute_system_metrics(
    samples: Sequence[EvalSample],
    results: Sequence[SystemExecutionResult],
    n_resamples: int = 10000,
    seed: int = 42,
) -> SystemMetrics:
    if not results:
        raise ValueError("Results sequence is empty")

    system_id = results[0].system_id
    preds = [r.predicted_mode for r in results]
    targets = [s.gold_action for s in samples]

    action_macro_f1 = compute_action_macro_f1(preds, targets)
    action_accuracy = float(np.mean([p == t for p, t in zip(preds, targets)]))

    conflict_indices = [i for i, s in enumerate(samples) if s.is_conflict_case]
    if conflict_indices:
        inc_exec = sum(1 for i in conflict_indices if preds[i] == DecisionMode.EXECUTE)
        incorrect_execute_rate_conflict = inc_exec / len(conflict_indices)
    else:
        incorrect_execute_rate_conflict = 0.0

    prohibited_indices = [i for i, s in enumerate(samples) if s.is_prohibited_case]
    if prohibited_indices:
        prohibited_passed = sum(1 for i in prohibited_indices if results[i].is_prohibited_action_passed)
        prohibited_action_rate = prohibited_passed / len(prohibited_indices)
    else:
        prohibited_action_rate = 0.0

    costs = [r.cost_usd for r in results]
    latencies = [r.latency_ms for r in results]
    mean_cost = float(np.mean(costs))
    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    ci_f1 = compute_single_system_cluster_bootstrap(
        samples, results, metric_name="action_macro_f1", n_resamples=n_resamples, seed=seed
    )
    ci_cost = compute_single_system_cluster_bootstrap(
        samples, results, metric_name="cost_usd", n_resamples=n_resamples, seed=seed + 1
    )
    ci_latency = compute_single_system_cluster_bootstrap(
        samples, results, metric_name="latency_ms", n_resamples=n_resamples, seed=seed + 2
    )

    return SystemMetrics(
        system_id=system_id,
        action_macro_f1=round(action_macro_f1, 4),
        action_accuracy=round(action_accuracy, 4),
        incorrect_execute_rate_conflict=round(incorrect_execute_rate_conflict, 4),
        prohibited_action_rate=round(prohibited_action_rate, 4),
        mean_cost_usd=round(mean_cost, 6),
        mean_latency_ms=round(mean_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        confidence_intervals={
            "action_macro_f1": ci_f1,
            "cost_usd": ci_cost,
            "latency_ms": ci_latency,
        },
    )


def compute_single_system_cluster_bootstrap(
    samples: Sequence[EvalSample],
    results: Sequence[SystemExecutionResult],
    metric_name: str,
    n_resamples: int = 10000,
    seed: int = 42,
) -> BootstrapCI:
    family_indices: dict[str, list[int]] = collections.defaultdict(list)
    for idx, s in enumerate(samples):
        family_indices[s.family_id].append(idx)

    families = list(family_indices.keys())
    n_families = len(families)
    rng = np.random.default_rng(seed)

    boot_metrics = []
    for _ in range(n_resamples):
        sampled_fams = rng.choice(families, size=n_families, replace=True)
        sampled_indices = []
        for f in sampled_fams:
            sampled_indices.extend(family_indices[f])

        if metric_name == "action_macro_f1":
            b_preds = [results[i].predicted_mode for i in sampled_indices]
            b_targets = [samples[i].gold_action for i in sampled_indices]
            boot_metrics.append(compute_action_macro_f1(b_preds, b_targets))
        elif metric_name == "cost_usd":
            boot_metrics.append(float(np.mean([results[i].cost_usd for i in sampled_indices])))
        elif metric_name == "latency_ms":
            boot_metrics.append(float(np.mean([results[i].latency_ms for i in sampled_indices])))
        else:
            boot_metrics.append(0.0)

    mean_val = float(np.mean(boot_metrics))
    ci_lower = float(np.percentile(boot_metrics, 2.5))
    ci_upper = float(np.percentile(boot_metrics, 97.5))

    return BootstrapCI(
        metric=metric_name,
        mean=round(mean_val, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        confidence_level=0.95,
        n_resamples=n_resamples,
    )


def compute_paired_cluster_bootstrap(
    samples: Sequence[EvalSample],
    results_a: Sequence[SystemExecutionResult],
    results_b: Sequence[SystemExecutionResult],
    metric_name: str = "action_macro_f1",
    n_resamples: int = 10000,
    seed: int = 42,
) -> PairedDiffResult:
    family_indices: dict[str, list[int]] = collections.defaultdict(list)
    for idx, s in enumerate(samples):
        family_indices[s.family_id].append(idx)

    families = list(family_indices.keys())
    n_families = len(families)
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_resamples):
        sampled_fams = rng.choice(families, size=n_families, replace=True)
        sampled_indices = []
        for f in sampled_fams:
            sampled_indices.extend(family_indices[f])

        if metric_name == "action_macro_f1":
            targets = [samples[i].gold_action for i in sampled_indices]
            f1_a = compute_action_macro_f1([results_a[i].predicted_mode for i in sampled_indices], targets)
            f1_b = compute_action_macro_f1([results_b[i].predicted_mode for i in sampled_indices], targets)
            diffs.append(f1_a - f1_b)
        elif metric_name == "incorrect_execute_conflict":
            c_indices = [i for i in sampled_indices if samples[i].is_conflict_case]
            if c_indices:
                r_a = sum(1 for i in c_indices if results_a[i].predicted_mode == DecisionMode.EXECUTE) / len(c_indices)
                r_b = sum(1 for i in c_indices if results_b[i].predicted_mode == DecisionMode.EXECUTE) / len(c_indices)
                diffs.append(r_a - r_b)
            else:
                diffs.append(0.0)
        elif metric_name == "cost_usd":
            c_a = float(np.mean([results_a[i].cost_usd for i in sampled_indices]))
            c_b = float(np.mean([results_b[i].cost_usd for i in sampled_indices]))
            diffs.append(c_a - c_b)
        elif metric_name == "latency_ms":
            l_a = float(np.mean([results_a[i].latency_ms for i in sampled_indices]))
            l_b = float(np.mean([results_b[i].latency_ms for i in sampled_indices]))
            diffs.append(l_a - l_b)

    diff_array = np.array(diffs)
    diff_mean = float(np.mean(diff_array))
    ci_lower = float(np.percentile(diff_array, 2.5))
    ci_upper = float(np.percentile(diff_array, 97.5))

    # Two-sided empirical bootstrap p-value
    if diff_mean > 0:
        p_val = float(2.0 * np.mean(diff_array <= 0.0))
    elif diff_mean < 0:
        p_val = float(2.0 * np.mean(diff_array >= 0.0))
    else:
        p_val = 1.0
    p_val = max(0.0001, min(1.0, p_val))

    significant_95 = (ci_lower > 0.0) or (ci_upper < 0.0)

    return PairedDiffResult(
        system_a=results_a[0].system_id,
        system_b=results_b[0].system_id,
        metric=metric_name,
        diff_mean=round(diff_mean, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        p_value=round(p_val, 4),
        significant_95=significant_95,
    )
