#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.harness.compare import ComparisonReport
from services.harness.metrics import MetricsReport, load_labels, score_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a harness comparison against labels"
    )
    parser.add_argument("--comparison", required=True, help="comparison.json path")
    parser.add_argument(
        "--labels",
        default=str(REPO_ROOT / "fixtures" / "x_labels.json"),
        help="Ground-truth labels JSON",
    )
    parser.add_argument("--out", default=None, help="Output directory")
    return parser


def _render_markdown(report: MetricsReport) -> str:
    lines = [
        f"# Harness Metrics {report.comparison_id}",
        "",
        f"- Labels: {report.labels_count}",
        "",
        "| Metric | Rule | LLM |",
        "|---|---:|---:|",
    ]
    fields = (
        ("Complaint precision", "complaint_precision"),
        ("Complaint recall", "complaint_recall"),
        ("Category accuracy", "category_accuracy"),
        ("Decision accuracy", "decision_accuracy"),
        ("Accepted precision", "accepted_precision"),
        ("Location accuracy", "location_accuracy"),
        ("Source intent accuracy", "source_intent_accuracy"),
        ("Review rate", "review_rate"),
        ("False accepts", "false_accepts"),
        ("False rejects", "false_rejects"),
        ("Cost USD", "total_cost_usd"),
        ("Cost per correct USD", "cost_per_correct_usd"),
    )
    for title, field in fields:
        rule_value = getattr(report.rule, field)
        llm_value = getattr(report.llm, field)
        if isinstance(rule_value, float):
            lines.append(f"| {title} | {rule_value:.3f} | {llm_value:.3f} |")
        else:
            lines.append(f"| {title} | {rule_value} | {llm_value} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison_path = Path(args.comparison)
    if not comparison_path.is_file():
        print(f"Comparison not found: {comparison_path}", file=sys.stderr)
        return 2

    report = ComparisonReport.model_validate(
        json.loads(comparison_path.read_text(encoding="utf-8"))
    )
    labels = load_labels(args.labels)
    metrics = score_comparison(report, labels)

    out_dir = Path(args.out) if args.out else comparison_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "metrics.md").write_text(_render_markdown(metrics), encoding="utf-8")

    summary = {
        "comparison_id": metrics.comparison_id,
        "labels": metrics.labels_count,
        "rule": {
            "decision_accuracy": metrics.rule.decision_accuracy,
            "accepted_precision": metrics.rule.accepted_precision,
            "complaint_precision": metrics.rule.complaint_precision,
            "complaint_recall": metrics.rule.complaint_recall,
            "category_accuracy": metrics.rule.category_accuracy,
            "location_accuracy": metrics.rule.location_accuracy,
            "source_intent_accuracy": metrics.rule.source_intent_accuracy,
            "false_accepts": metrics.rule.false_accepts,
            "cost_usd": metrics.rule.total_cost_usd,
        },
        "llm": {
            "decision_accuracy": metrics.llm.decision_accuracy,
            "accepted_precision": metrics.llm.accepted_precision,
            "complaint_precision": metrics.llm.complaint_precision,
            "complaint_recall": metrics.llm.complaint_recall,
            "category_accuracy": metrics.llm.category_accuracy,
            "location_accuracy": metrics.llm.location_accuracy,
            "source_intent_accuracy": metrics.llm.source_intent_accuracy,
            "false_accepts": metrics.llm.false_accepts,
            "cost_usd": metrics.llm.total_cost_usd,
        },
        "out_dir": str(out_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
