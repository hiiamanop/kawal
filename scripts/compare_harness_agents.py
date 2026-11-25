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
from services.harness.compare import HarnessComparisonRunner
from services.harness.llm import (
    LLMArbiterAgent,
    LLMExtractionAgent,
    LLMTriageAgent,
    LLMVerifierAgent,
    OpenRouterChatClient,
)
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sources import ReplayXSource, SourceError
from services.llm_config import openrouter_api_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare rule-based and LLM harness agents"
    )
    parser.add_argument("--fixture", required=True, help="Fixture file or directory")
    parser.add_argument("--out", default="runs", help="Output directory")
    parser.add_argument("--limit", type=int, default=10, help="Maximum posts")
    parser.add_argument(
        "--max-cost-usd", type=float, default=0.05, help="Budget per case"
    )
    parser.add_argument("--model", default=None, help="OpenRouter model id")
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument("--comparison-id", default=None)
    parser.add_argument(
        "--llm-extraction",
        action="store_true",
        help="Use LLMExtractionAgent instead of the rule extractor",
    )
    parser.add_argument(
        "--llm-arbiter",
        action="store_true",
        help="Use LLMArbiterAgent instead of the rule arbiter",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable model reasoning (useful for reasoning models that emit no content)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not openrouter_api_key():
        print("OPENROUTER_API_KEY is required for the LLM comparison", file=sys.stderr)
        return 2

    try:
        posts = ReplayXSource(args.fixture).fetch(limit=args.limit)
    except SourceError as exc:
        print(f"Source error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    if not posts:
        print("No posts found", file=sys.stderr)
        return 2

    rule_orchestrator = HarnessOrchestrator(
        triage=RuleTriageAgent(),
        extraction=RuleExtractionAgent(),
        verifier=RuleVerifierAgent(),
        arbiter=RuleArbiterAgent(),
        max_cost_usd=args.max_cost_usd,
    )
    client = OpenRouterChatClient(
        model_id=args.model,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        reasoning_enabled=False if args.no_reasoning else None,
    )
    llm_orchestrator = HarnessOrchestrator(
        triage=LLMTriageAgent(client),
        extraction=(
            LLMExtractionAgent(client) if args.llm_extraction else RuleExtractionAgent()
        ),
        verifier=LLMVerifierAgent(client),
        arbiter=LLMArbiterAgent(client) if args.llm_arbiter else RuleArbiterAgent(),
        max_cost_usd=args.max_cost_usd,
    )
    report = HarnessComparisonRunner(args.out, rule_orchestrator, llm_orchestrator).run(
        posts, comparison_id=args.comparison_id, source="replay"
    )

    summary = {
        "comparison_id": report.comparison_id,
        "output_dir": report.output_dir,
        "total_posts": report.total_posts,
        "agreement_rate": report.agreement_rate,
        "rule_accepted": report.rule_accepted,
        "llm_accepted": report.llm_accepted,
        "rule_needs_review": report.rule_needs_review,
        "llm_needs_review": report.llm_needs_review,
        "rule_total_cost_usd": report.rule_total_cost_usd,
        "llm_total_cost_usd": report.llm_total_cost_usd,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
