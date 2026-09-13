from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.runner import (
    generate_standard_thesis_eval_samples,
    run_thesis_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL Thesis Benchmark & Evaluation Harness (M6)")
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/thesis_evaluation_report.json",
        help="Path to write the thesis evaluation JSON report",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Number of paired cluster bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--num-families",
        type=int,
        default=24,
        help="Number of scenario families to evaluate (default: 24)",
    )
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=8,
        help="Number of samples per scenario family (default: 8)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("KAWAL THESIS EVALUATION (M6) — BASELINES B0-B4, P, & ABLATIONS A1-A5")
    print(f"Scenario Families: {args.num_families} | Samples/Family: {args.samples_per_family}")
    print(f"Bootstrap Resamples: {args.bootstrap_resamples} | Seed: {args.seed}")
    print("=" * 70)

    samples = generate_standard_thesis_eval_samples(
        num_families=args.num_families,
        samples_per_family=args.samples_per_family,
    )
    print(f"Generated {len(samples)} evaluation cases across {args.num_families} families.")

    manifest_hashes = {
        "git_commit": "HEAD",
        "policy_bundle": "m1.v1",
        "model_architecture": "indobert-base-p1",
    }

    report = run_thesis_evaluation(
        samples=samples,
        n_bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        manifest_hashes=manifest_hashes,
    )

    print("\n" + "-" * 70)
    print("SYSTEM METRICS SUMMARY:")
    print(f"{'System':<8} {'Macro F1':<10} {'Accuracy':<10} {'Conflict Incr':<15} {'Prohibited Rate':<17} {'Cost ($)':<10} {'Lat (ms)':<10}")
    print("-" * 70)
    for sys_id, m in report.system_metrics.items():
        print(f"{sys_id:<8} {m.action_macro_f1:<10.4f} {m.action_accuracy:<10.4f} {m.incorrect_execute_rate_conflict:<15.4f} {m.prohibited_action_rate:<17.4f} {m.mean_cost_usd:<10.6f} {m.mean_latency_ms:<10.2f}")

    print("\n" + "-" * 70)
    print("THESIS HYPOTHESES VERIFICATION:")
    print("-" * 70)
    for hid, hyp in report.hypotheses.items():
        status = "PASSED [OK]" if hyp.passed else "FAILED [X]"
        print(f"[{status}] {hid}: {hyp.statement}")
        print(f"        Result: {hyp.comparison}")

    print("\n" + "-" * 70)
    print("ROBUSTNESS INVARIANTS:")
    print("-" * 70)
    for inv, res in report.robustness_invariants.items():
        status = "PASSED [OK]" if res else "FAILED [X]"
        print(f"[{status}] {inv}")

    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2)

    print(f"\nReport written to: {out_p.resolve()}")
    all_passed = all(h.passed for h in report.hypotheses.values()) and all(report.robustness_invariants.values())
    if all_passed:
        print("\nAll thesis hypotheses and robustness invariants PASSED successfully.")
        return 0
    else:
        print("\nSome hypotheses or invariants did not pass.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
