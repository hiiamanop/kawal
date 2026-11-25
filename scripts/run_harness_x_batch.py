#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.batch import HarnessBatchRunner
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sources import (
    AgentReachSourceError,
    AgentReachXSource,
    ReplayXSource,
    SourceError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KAWAL X harness batch runner")
    parser.add_argument("--fixture", help="Fixture file or directory")
    parser.add_argument("--live-source", action="store_true", help="Use Agent-Reach")
    parser.add_argument(
        "--query", default="jalan rusak bandung", help="Live X search query"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum posts to process"
    )
    parser.add_argument("--out", default="runs", help="Output directory")
    parser.add_argument(
        "--max-cost-usd", type=float, default=0.05, help="Budget per case"
    )
    parser.add_argument(
        "--save-fixtures", default=None, help="Directory for captured posts"
    )
    parser.add_argument(
        "--batch-id", default=None, help="Optional deterministic batch id"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.live_source:
            posts = AgentReachXSource().fetch(args.query, limit=args.limit)
            source = "live"
        else:
            if not args.fixture:
                print(
                    "--fixture is required unless --live-source is set", file=sys.stderr
                )
                return 2
            posts = ReplayXSource(args.fixture).fetch(limit=args.limit)
            source = "replay"
    except (SourceError, AgentReachSourceError) as exc:
        print(f"Source error [{exc.code}]: {exc}", file=sys.stderr)
        return 2

    if not posts:
        print("No X posts found", file=sys.stderr)
        return 2

    orchestrator = HarnessOrchestrator(
        triage=RuleTriageAgent(),
        extraction=RuleExtractionAgent(),
        verifier=RuleVerifierAgent(),
        arbiter=RuleArbiterAgent(),
        max_cost_usd=args.max_cost_usd,
    )
    report = HarnessBatchRunner(
        args.out,
        orchestrator,
        fixture_out=args.save_fixtures,
    ).run(
        posts,
        batch_id=args.batch_id,
        source=source,
        query=args.query if args.live_source else None,
    )

    summary = {
        "batch_id": report.batch_id,
        "output_dir": report.output_dir,
        "total_posts": report.total_posts,
        "accepted": report.accepted,
        "needs_review": report.needs_review,
        "rejected": report.rejected,
        "failed": report.failed,
        "budget_exceeded": report.budget_exceeded,
        "total_cost_usd": report.total_cost_usd,
        "categories": report.categories,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if report.total_posts == 0 or report.failed == report.total_posts:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
