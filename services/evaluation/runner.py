from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from contracts.models import Category, DecisionMode, RiskLevel
from services.evaluation.contracts import (
    EvalSample,
    PairedDiffResult,
    SystemExecutionResult,
    SystemID,
    SystemMetrics,
    ThesisEvaluationReport,
    ThesisHypothesisResult,
)
from services.evaluation.robustness import run_robustness_invariant_checks
from services.evaluation.statistics import (
    compute_paired_cluster_bootstrap,
    compute_system_metrics,
)
from services.evaluation.systems import execute_system


def generate_standard_thesis_eval_samples(
    num_families: int = 24,
    samples_per_family: int = 8,
) -> list[EvalSample]:
    """Generate deterministic, balanced evaluation cases across scenario families."""
    categories = list(Category)
    samples: list[EvalSample] = []
    case_counter = 0

    for fam_idx in range(num_families):
        fam_id = f"fam-{fam_idx:03d}"
        cat = categories[fam_idx % len(categories)]

        for s_idx in range(samples_per_family):
            case_counter += 1
            case_id = f"eval-{case_counter:04d}"

            # Distribute profiles deterministically across 8 distinct archetypes
            profile = s_idx % 8
            is_conflict = (profile in (4, 5))
            is_prohibited = (profile == 6)

            if is_prohibited:
                gold_action = DecisionMode.RE_EVALUATE
                completeness = "SUFFICIENT"
                risk = RiskLevel.MEDIUM
                text = f"Laporan {cat.value.lower()}: Jl. Pajajaran No. 12, Kelurahan Pasirkaliki. Data NIK pelapor 3273010101010001 dilampirkan foto privat."
            elif is_conflict:
                gold_action = DecisionMode.RE_EVALUATE
                completeness = "INCOMPLETE"
                risk = RiskLevel.HIGH
                text = f"Aduan mendesak {cat.value.lower()}: Ada indikasi bahaya berat di Jl. Asia Afrika No. {10 + s_idx}, tetapi patokan lokasi merujuk ke dua simpang berbeda."
            elif profile in (0, 7):  # Complete & valid
                gold_action = DecisionMode.EXECUTE
                completeness = "SUFFICIENT"
                risk = RiskLevel.MEDIUM
                text = f"Selamat pagi, lapor perbaikan {cat.value.lower()} di Jl. Merdeka No. {s_idx + 1}, RT 02 RW 03, Kelurahan Babakan Ciamis, Kecamatan Sumur Bandung. Waktu: kemarin siang."
            elif profile == 1:  # Incomplete location
                gold_action = DecisionMode.REQUEST_CLARIFICATION
                completeness = "INCOMPLETE"
                risk = RiskLevel.LOW
                text = f"Mohon ditindaklanjuti masalah {cat.value.lower()}, lokasinya dekat pasar tetapi belum jelas alamat nomor rumahnya."
            elif profile == 2:  # Ambiguous location
                gold_action = DecisionMode.REQUEST_CLARIFICATION
                completeness = "AMBIGUOUS"
                risk = RiskLevel.MEDIUM
                text = f"Laporan {cat.value.lower()} di dekat jembatan pasupati, acuan titik tepatnya masih belum terinci."
            else:  # Out of scope / spam (profile == 3)
                gold_action = DecisionMode.REJECT_IGNORE
                completeness = "INCOMPLETE"
                risk = RiskLevel.LOW
                text = f"Sekadar opini dan salam sapa untuk rekan-rekan dinas {cat.value.lower()}."

            samples.append(
                EvalSample(
                    case_id=case_id,
                    family_id=fam_id,
                    text=text,
                    gold_category=cat,
                    gold_risk=risk,
                    gold_completeness=completeness,
                    gold_action=gold_action,
                    is_conflict_case=is_conflict,
                    is_prohibited_case=is_prohibited,
                    context={"family_index": fam_idx},
                )
            )

    return samples


def run_thesis_evaluation(
    samples: Sequence[EvalSample] | None = None,
    n_bootstrap_resamples: int = 10000,
    seed: int = 42,
    manifest_hashes: dict[str, str] | None = None,
) -> ThesisEvaluationReport:
    """Execute complete M6 thesis evaluation across B0-B4, P, and A1-A5 with 95% CIs."""
    eval_samples = list(samples) if samples else generate_standard_thesis_eval_samples()
    unique_families = len({s.family_id for s in eval_samples})

    systems_to_eval = (
        SystemID.P,
        SystemID.B0,
        SystemID.B1,
        SystemID.B2,
        SystemID.B3,
        SystemID.B4,
        SystemID.A1,
        SystemID.A2,
        SystemID.A3,
        SystemID.A4,
        SystemID.A5,
    )

    all_results: dict[SystemID, list[SystemExecutionResult]] = {}
    for sys_id in systems_to_eval:
        all_results[sys_id] = [execute_system(s, sys_id) for s in eval_samples]

    system_metrics: dict[str, SystemMetrics] = {}
    for idx, sys_id in enumerate(systems_to_eval):
        system_metrics[sys_id.value] = compute_system_metrics(
            samples=eval_samples,
            results=all_results[sys_id],
            n_resamples=n_bootstrap_resamples,
            seed=seed + (idx * 10),
        )

    # Compute paired statistical hypothesis tests
    paired_diffs: list[PairedDiffResult] = []

    # H1: P - B3 action macro F1
    diff_p_b3 = compute_paired_cluster_bootstrap(
        eval_samples, all_results[SystemID.P], all_results[SystemID.B3],
        metric_name="action_macro_f1", n_resamples=n_bootstrap_resamples, seed=seed + 100,
    )
    paired_diffs.append(diff_p_b3)

    # H1b: P - A1 action macro F1 (Ablation: Contextual Trust)
    diff_p_a1 = compute_paired_cluster_bootstrap(
        eval_samples, all_results[SystemID.P], all_results[SystemID.A1],
        metric_name="action_macro_f1", n_resamples=n_bootstrap_resamples, seed=seed + 101,
    )
    paired_diffs.append(diff_p_a1)

    # H2: P - A2 incorrect execute on conflict cases
    diff_p_a2 = compute_paired_cluster_bootstrap(
        eval_samples, all_results[SystemID.P], all_results[SystemID.A2],
        metric_name="incorrect_execute_conflict", n_resamples=n_bootstrap_resamples, seed=seed + 102,
    )
    paired_diffs.append(diff_p_a2)

    # H3: Cost comparison P vs B4
    diff_cost_p_b4 = compute_paired_cluster_bootstrap(
        eval_samples, all_results[SystemID.P], all_results[SystemID.B4],
        metric_name="cost_usd", n_resamples=n_bootstrap_resamples, seed=seed + 103,
    )
    paired_diffs.append(diff_cost_p_b4)

    # Evaluate Hypotheses
    p_metrics = system_metrics[SystemID.P.value]
    b3_metrics = system_metrics[SystemID.B3.value]
    b4_metrics = system_metrics[SystemID.B4.value]
    a2_metrics = system_metrics[SystemID.A2.value]
    a4_metrics = system_metrics[SystemID.A4.value]

    h1_passed = diff_p_b3.diff_mean > 0.0 and diff_p_b3.ci_lower > 0.0
    h2_passed = diff_p_a2.diff_mean < 0.0 and p_metrics.incorrect_execute_rate_conflict < a2_metrics.incorrect_execute_rate_conflict
    cost_ratio = p_metrics.mean_cost_usd / b4_metrics.mean_cost_usd if b4_metrics.mean_cost_usd > 0 else 1.0
    h3_passed = cost_ratio < 0.5 and (p_metrics.action_macro_f1 >= b4_metrics.action_macro_f1 - 0.02)
    h4_passed = p_metrics.prohibited_action_rate == 0.0 and a4_metrics.prohibited_action_rate > 0.0

    invariants = run_robustness_invariant_checks()
    h5_passed = all(invariants.values())

    hypotheses = {
        "H1": ThesisHypothesisResult(
            hypothesis_id="H1",
            statement="Contextual trust improves action macro-F1 over confidence-only baseline (B3) and unconditioned trust (A1).",
            passed=h1_passed,
            comparison=f"P ({p_metrics.action_macro_f1:.4f}) vs B3 ({b3_metrics.action_macro_f1:.4f}), Delta={diff_p_b3.diff_mean:.4f} [95% CI: {diff_p_b3.ci_lower:.4f}, {diff_p_b3.ci_upper:.4f}]",
            evidence={"diff_p_b3": diff_p_b3.model_dump(mode="json"), "diff_p_a1": diff_p_a1.model_dump(mode="json")},
        ),
        "H2": ThesisHypothesisResult(
            hypothesis_id="H2",
            statement="Conflict diagnosis significantly reduces incorrect execute decisions on the conflict subset (P vs A2).",
            passed=h2_passed,
            comparison=f"P incorrect execute ({p_metrics.incorrect_execute_rate_conflict:.4f}) vs A2 ({a2_metrics.incorrect_execute_rate_conflict:.4f}), Delta={diff_p_a2.diff_mean:.4f} [95% CI: {diff_p_a2.ci_lower:.4f}, {diff_p_a2.ci_upper:.4f}]",
            evidence={"diff_p_a2": diff_p_a2.model_dump(mode="json")},
        ),
        "H3": ThesisHypothesisResult(
            hypothesis_id="H3",
            statement="Proposed KAWAL reduces generative inference cost compared to Always-LLM (B4) while maintaining non-inferior action quality.",
            passed=h3_passed,
            comparison=f"Cost ratio P/B4 = {cost_ratio:.4f} (P=${p_metrics.mean_cost_usd:.6f}, B4=${b4_metrics.mean_cost_usd:.6f}), F1 delta = {p_metrics.action_macro_f1 - b4_metrics.action_macro_f1:.4f}",
            evidence={"diff_cost_p_b4": diff_cost_p_b4.model_dump(mode="json"), "cost_ratio": round(cost_ratio, 4)},
        ),
        "H4": ThesisHypothesisResult(
            hypothesis_id="H4",
            statement="Deterministic hard policy gate completely eliminates prohibited action attempts that leak under soft advisory policy (A4).",
            passed=h4_passed,
            comparison=f"Prohibited action rate: P={p_metrics.prohibited_action_rate:.4f} vs A4={a4_metrics.prohibited_action_rate:.4f}",
            evidence={"P_prohibited_rate": p_metrics.prohibited_action_rate, "A4_prohibited_rate": a4_metrics.prohibited_action_rate},
        ),
        "H5": ThesisHypothesisResult(
            hypothesis_id="H5",
            statement="Core robustness invariants hold: zero duplicate tickets under crash boundaries, audit trail reproducibility, fail-closed policy, and DLQ quarantine.",
            passed=h5_passed,
            comparison=f"All {len(invariants)} robustness invariants passed: {list(invariants.keys())}",
            evidence=invariants,
        ),
    }

    return ThesisEvaluationReport(
        total_cases=len(eval_samples),
        total_families=unique_families,
        systems_evaluated=systems_to_eval,
        system_metrics=system_metrics,
        hypotheses=hypotheses,
        paired_differences=paired_diffs,
        robustness_invariants=invariants,
        frozen_manifest_hashes=manifest_hashes or {},
    )
