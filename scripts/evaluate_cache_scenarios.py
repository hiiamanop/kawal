#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.harness.cache_scenarios import (
    CacheScenarioSet,
    ScenarioReport,
    evaluate_scenarios,
)
from services.intelligence.embedder import (
    DeterministicFallbackEmbedder,
    IndoBertEmbedder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep semantic-cache thresholds on near-duplicate scenarios"
    )
    parser.add_argument(
        "--fixture",
        default=str(REPO_ROOT / "fixtures" / "x_cache_scenarios.json"),
    )
    parser.add_argument("--out", default="runs/m6-scenarios")
    parser.add_argument(
        "--thresholds",
        default="0.80,0.85,0.90,0.92,0.95",
        help="Comma-separated cosine thresholds",
    )
    parser.add_argument(
        "--embedder",
        choices=("deterministic", "indobert"),
        default="deterministic",
    )
    parser.add_argument(
        "--embedder-path",
        default=None,
        help="Local IndoBERT artifact directory with config.json",
    )
    parser.add_argument(
        "--allow-fallback-semantic",
        action="store_true",
        help="Allow semantic matching with the deterministic fallback embedder",
    )
    return parser


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError("thresholds must be values between 0 and 1")
    return thresholds


def _render_markdown(report: ScenarioReport) -> str:
    lines = [
        "# Cache Scenario Calibration",
        "",
        f"- Scenarios: {report.scenarios}",
        f"- Best threshold: {report.best_threshold}",
        "",
        "| Threshold | Precision | Recall | Accuracy | TP | FP | FN | TN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in report.thresholds:
        lines.append(
            f"| {metric.threshold:.2f} | {metric.precision:.3f} | "
            f"{metric.recall:.3f} | {metric.accuracy:.3f} | "
            f"{metric.true_positive} | {metric.false_positive} | "
            f"{metric.false_negative} | {metric.true_negative} |"
        )
    lines.extend(
        [
            "",
            "## Scenarios at best threshold",
            "",
            "| Scenario | Expected | Actual | Match | Similarity |",
            "|---|---|---|---|---:|",
        ]
    )
    best = next(
        (
            metric
            for metric in report.thresholds
            if metric.threshold == report.best_threshold
        ),
        None,
    )
    if best is not None:
        for result in best.results:
            lines.append(
                f"| {result.scenario_id} | {result.expected_reuse} | "
                f"{result.actual_reuse} | {result.match_type} | "
                f"{result.similarity:.3f} |"
            )
    return "\n".join(lines) + "\n"


def _build_embedder(args: argparse.Namespace):
    if args.embedder == "deterministic":
        return DeterministicFallbackEmbedder(dim=768)
    if args.embedder_path:
        return IndoBertEmbedder(model_path=args.embedder_path)
    return IndoBertEmbedder.get_instance()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        thresholds = _parse_thresholds(args.thresholds)
        payload = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
        scenarios = CacheScenarioSet.model_validate(payload).scenarios
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Invalid scenario input: {exc}", file=sys.stderr)
        return 2

    embedder = _build_embedder(args)
    if getattr(embedder, "is_fallback", False) and not args.allow_fallback_semantic:
        print(
            "The selected embedder is a fallback; semantic reuse is disabled. "
            "Use --allow-fallback-semantic only for routing tests.",
            file=sys.stderr,
        )
        return 2

    report = evaluate_scenarios(
        scenarios,
        embedder=embedder,
        thresholds=thresholds,
        allow_fallback_semantic=args.allow_fallback_semantic,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenario_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "scenario_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "scenarios": report.scenarios,
                "best_threshold": report.best_threshold,
                "embedder": getattr(embedder, "model_id", "unknown"),
                "fallback": getattr(embedder, "is_fallback", False),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
