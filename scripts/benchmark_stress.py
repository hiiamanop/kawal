from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.stress import (
    STANDARD_LOAD_PROFILES,
    StressResult,
    run_full_stress_suite,
    run_stress_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="KAWAL Load, Concurrency & Stress Benchmark (PRD §35, §46)")
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/stress_benchmark_report.json",
        help="Path to output stress benchmark report JSON",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick smoke benchmark with scaled down case counts",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("KAWAL LOAD, CONCURRENCY & STRESS BENCHMARK (PRD §35, §46, NFR-04)")
    print("MODE: simulation-only internal pipeline; excludes live PostgreSQL, Redpanda, OpenWA, and ONNX inference.")
    print("Evaluating tiered load profiles: 25%, 50%, 75%, 100%, 125%, burst 2x")
    print("=" * 80)

    if args.quick:
        from services.evaluation.stress import LoadProfile
        profiles = (
            LoadProfile(name="25%", target_cases=10, concurrent_workers=2),
            LoadProfile(name="50%", target_cases=20, concurrent_workers=4),
            LoadProfile(name="100%", target_cases=40, concurrent_workers=8),
            LoadProfile(name="burst_2x", target_cases=60, concurrent_workers=12, burst_multiplier=2.0),
        )
    else:
        profiles = STANDARD_LOAD_PROFILES

    results = run_full_stress_suite(profiles)

    print("\n" + "-" * 80)
    print(f"{'Profile':<10} {'Cases':<8} {'Msgs/s':<10} {'Cases/s':<10} {'p50 (ms)':<10} {'p95 (ms)':<10} {'p99 (ms)':<10} {'Zero Dup':<10}")
    print("-" * 80)
    for r in results:
        dup_str = "PASS" if r.zero_duplicates_invariant else "FAIL"
        print(f"{r.profile_name:<10} {r.total_cases:<8} {r.messages_per_second:<10.1f} {r.cases_per_second:<10.1f} {r.latency.p50_ms:<10.1f} {r.latency.p95_ms:<10.1f} {r.latency.p99_ms:<10.1f} {dup_str:<10}")

    all_passed = all(r.zero_duplicates_invariant and r.all_cases_accounted for r in results)

    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump([r.model_dump(mode="json") for r in results], f, indent=2)

    print(f"\nReport written to: {out_p.resolve()}")
    if all_passed:
        print("\nAll stress benchmark profiles and zero-duplicate invariants PASSED.")
        return 0
    else:
        print("\nSome stress profiles failed invariants.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
