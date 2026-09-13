from __future__ import annotations

from services.evaluation.contracts import (
    BootstrapCI,
    EvalSample,
    PairedDiffResult,
    SystemExecutionResult,
    SystemID,
    SystemMetrics,
    ThesisEvaluationReport,
    ThesisHypothesisResult,
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

__all__ = [
    "BootstrapCI",
    "EvalSample",
    "PairedDiffResult",
    "SystemExecutionResult",
    "SystemID",
    "SystemMetrics",
    "ThesisEvaluationReport",
    "ThesisHypothesisResult",
    "compute_action_macro_f1",
    "compute_paired_cluster_bootstrap",
    "compute_single_system_cluster_bootstrap",
    "compute_system_metrics",
    "execute_system",
    "generate_standard_thesis_eval_samples",
    "run_robustness_invariant_checks",
    "run_thesis_evaluation",
]
