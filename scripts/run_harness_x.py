#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sinks import FileRunSink
from services.harness.sources import (
    AgentReachSourceError,
    AgentReachXSource,
    ReplayXSource,
    SourceError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KAWAL X harness milestone 1")
    parser.add_argument("--fixture", help="Path to a recorded X fixture")
    parser.add_argument("--query", default="jalan rusak bandung", help="Live X search query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum live posts to fetch")
    parser.add_argument("--live-source", action="store_true", help="Use Agent-Reach instead of a fixture")
    parser.add_argument("--out", default="runs", help="Output directory for run artifacts")
    parser.add_argument("--max-cost-usd", type=float, default=0.05, help="Per-run budget")
    parser.add_argument("--run-id", default=None, help="Optional deterministic run id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.live_source:
            posts = AgentReachXSource().fetch(args.query, limit=args.limit)
        else:
            if not args.fixture:
                print("--fixture is required unless --live-source is set", file=sys.stderr)
                return 2
            posts = ReplayXSource(args.fixture).fetch(limit=1)
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
    run = orchestrator.run(posts[0], run_id=args.run_id)
    receipt = FileRunSink(args.out).write(run)

    summary = {
        "run_id": run.run_id,
        "case_id": run.case.case_id if run.case is not None else None,
        "status": run.status,
        "output_dir": receipt.output_dir,
        "cost_usd": run.cost.total_cost_usd,
        "budget_status": run.cost.budget_status,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if run.status in ("COMPLETED", "NEEDS_REVIEW", "REJECTED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
