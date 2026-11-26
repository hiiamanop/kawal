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

from scripts.harness_llm_cli import add_llm_arguments, build_chat_client
from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.cache import HarnessAnalysisCache, HarnessCacheStoreError
from services.harness.cache_metrics import CacheEvaluationReport, score_cache_evaluation
from services.harness.cache_router import CacheFirstRunner
from services.harness.cache_store import (
    InMemoryHarnessCacheStore,
    PgVectorHarnessCacheStore,
)
from services.harness.cassette import CassetteError
from services.harness.llm import LLMArbiterAgent, LLMVerifierAgent, LLMTriageAgent
from services.harness.metrics import LabelSet, load_labels
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sources import ReplayXSource, SourceError
from services.intelligence.embedder import (
    DeterministicFallbackEmbedder,
    IndoBertEmbedder,
)
from services.llm_config import openrouter_api_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the harness semantic cache on X fixtures"
    )
    parser.add_argument("--fixture", required=True, help="Fixture file or directory")
    parser.add_argument(
        "--labels",
        default=str(REPO_ROOT / "fixtures" / "x_labels.json"),
        help="Ground-truth labels JSON",
    )
    parser.add_argument("--out", default="runs/m6", help="Output directory")
    parser.add_argument("--limit", type=int, default=50, help="Maximum posts")
    parser.add_argument("--analysis-version", required=True)
    parser.add_argument(
        "--cache-store", choices=("memory", "postgres"), default="memory"
    )
    parser.add_argument(
        "--cache-dsn",
        default=None,
        help="Postgres DSN; also read from KAWAL_TEST_DATABASE_URL",
    )
    parser.add_argument("--cache-namespace", default="harness-x-eval")
    parser.add_argument(
        "--cache-reset",
        action="store_true",
        help="Delete the selected namespace/version before pass 1",
    )
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--inner", choices=("rule", "llm"), default="llm")
    parser.add_argument(
        "--repeat-accepted",
        type=int,
        default=0,
        help="Repeat accepted labeled posts to form a duplicate stream",
    )
    parser.add_argument(
        "--embedder",
        choices=("deterministic", "indobert"),
        default="deterministic",
    )
    parser.add_argument(
        "--allow-fallback-semantic",
        action="store_true",
        help="Allow semantic reuse with the deterministic fallback embedder",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=0.05, help="Budget per case"
    )
    add_llm_arguments(parser)
    return parser


def _build_stream(posts: list, labels: LabelSet, repeat_accepted: int) -> list:
    if repeat_accepted <= 0:
        return posts
    accepted = [
        post
        for post in posts
        if (label := labels.labels.get(post.post_id)) is not None
        and label.is_complaint
        and "COMPLETED" in label.acceptable_statuses
    ]
    if not accepted:
        return []
    others = [post for post in posts if post not in accepted]
    repeated = [post for _ in range(repeat_accepted) for post in accepted]
    stream: list = []
    for index, post in enumerate(repeated):
        stream.append(post)
        if index < len(others):
            stream.append(others[index])
    stream.extend(others[len(repeated) :])
    return stream


def _build_store(args: argparse.Namespace):
    if args.cache_store == "memory":
        return InMemoryHarnessCacheStore()
    dsn = (
        args.cache_dsn
        or os.getenv("KAWAL_TEST_DATABASE_URL")
        or os.getenv("KAWAL_DATABASE_URL")
    )
    if not dsn:
        raise ValueError(
            "--cache-dsn or KAWAL_TEST_DATABASE_URL is required for postgres"
        )
    import psycopg

    return PgVectorHarnessCacheStore(lambda: psycopg.connect(dsn))


def _build_inner(args: argparse.Namespace) -> HarnessOrchestrator:
    if args.inner == "rule":
        return HarnessOrchestrator(
            triage=RuleTriageAgent(),
            extraction=RuleExtractionAgent(),
            verifier=RuleVerifierAgent(),
            arbiter=RuleArbiterAgent(),
            max_cost_usd=args.max_cost_usd,
        )
    if not args.replay_cassette and not openrouter_api_key():
        raise ValueError(
            "OPENROUTER_API_KEY is required for --inner llm unless replaying a cassette"
        )
    client = build_chat_client(args)
    return HarnessOrchestrator(
        triage=LLMTriageAgent(client),
        extraction=RuleExtractionAgent(),
        verifier=LLMVerifierAgent(client),
        arbiter=RuleArbiterAgent(),
        max_cost_usd=args.max_cost_usd,
    )


def _render_markdown(report: CacheEvaluationReport) -> str:
    lines = [
        "# Harness Cache Evaluation",
        "",
        f"- Namespace: `{report.namespace}`",
        f"- Analysis version: `{report.analysis_version}`",
        f"- Threshold: {report.threshold:.2f}",
        f"- Posts in evaluation pass: {report.total_posts}",
        f"- Cache-eligible posts: {report.cache_eligible}",
        "",
        "## Cache",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Exact hits | {report.exact_hits} |",
        f"| Semantic hits | {report.semantic_hits} |",
        f"| Rejected hits | {report.rejected_hits} |",
        f"| Misses | {report.misses} |",
        f"| Overall hit rate | {report.overall_hit_rate:.3f} |",
        f"| Cache-eligible hit rate | {report.cache_eligible_hit_rate:.3f} |",
        f"| False reuse | {report.false_reuse} |",
        f"| False reuse rate | {report.false_reuse_rate:.3f} |",
        "",
        "## Baseline vs evaluation pass",
        "",
        "The baseline is a separate no-cache pass. Warmup cost is not included in",
        "the saved values below.",
        "",
        "| Metric | Baseline | Cached |",
        "|---|---:|---:|",
        f"| Accepted | {report.baseline.accepted} | {report.cached.accepted} |",
        f"| Accepted precision | {report.baseline.accepted_precision:.3f} | {report.cached.accepted_precision:.3f} |",
        f"| Decision accuracy | {report.baseline.decision_accuracy:.3f} | {report.cached.decision_accuracy:.3f} |",
        f"| False accepts | {report.baseline.false_accepts} | {report.cached.false_accepts} |",
        f"| LLM calls | {report.baseline.llm_calls} | {report.cached.llm_calls} |",
        f"| Cost USD | {report.baseline.total_cost_usd:.6f} | {report.cached.total_cost_usd:.6f} |",
        f"| Latency ms | {report.baseline.total_latency_ms:.1f} | {report.cached.total_latency_ms:.1f} |",
        "",
        f"- LLM calls saved: {report.llm_calls_saved}",
        f"- Cost saved USD: {report.cost_saved_usd:.6f}",
        f"- Latency saved ms: {report.latency_saved_ms:.1f}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.passes < 2:
        print("--passes must be at least 2", file=sys.stderr)
        return 2
    if args.repeat_accepted < 0:
        print("--repeat-accepted must be non-negative", file=sys.stderr)
        return 2
    if not args.replay_cassette and args.inner == "llm" and not openrouter_api_key():
        print("OPENROUTER_API_KEY is required for --inner llm", file=sys.stderr)
        return 2

    try:
        posts = ReplayXSource(args.fixture).fetch(limit=args.limit)
    except SourceError as exc:
        print(f"Source error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    labels = load_labels(args.labels)
    stream = _build_stream(posts, labels, args.repeat_accepted)
    if not stream:
        print("No posts found for the requested evaluation", file=sys.stderr)
        return 2

    embedder = (
        DeterministicFallbackEmbedder(dim=768)
        if args.embedder == "deterministic"
        else IndoBertEmbedder.get_instance()
    )
    try:
        store = _build_store(args)
    except (ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    cache = HarnessAnalysisCache(
        store=store,
        embedder=embedder,
        namespace=args.cache_namespace,
        analysis_version=args.analysis_version,
        semantic_threshold=args.threshold,
        allow_fallback_semantic=args.allow_fallback_semantic,
    )
    if args.cache_reset:
        try:
            cache.reset()
        except HarnessCacheStoreError as exc:
            print(f"Cache error [{exc.code}]: {exc}", file=sys.stderr)
            return 2

    try:
        inner = _build_inner(args)
    except CassetteError as exc:
        print(f"Cassette error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    runner = CacheFirstRunner(inner=inner, cache=cache)
    baseline_runs = [inner.run(post) for post in stream]
    pass_runs = []
    for _ in range(args.passes):
        pass_runs.append([runner.run(post) for post in stream])

    warmup_runs = [run for pass_run in pass_runs[:-1] for run in pass_run]
    eval_runs = pass_runs[-1]
    report = score_cache_evaluation(
        warmup_runs,
        eval_runs,
        labels,
        baseline_runs=baseline_runs,
        namespace=args.cache_namespace,
        analysis_version=args.analysis_version,
        threshold=args.threshold,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "cache_report.md").write_text(_render_markdown(report), encoding="utf-8")

    summary = {
        "out_dir": str(out_dir),
        "namespace": report.namespace,
        "analysis_version": report.analysis_version,
        "threshold": report.threshold,
        "total_posts": report.total_posts,
        "cache_eligible": report.cache_eligible,
        "exact_hits": report.exact_hits,
        "semantic_hits": report.semantic_hits,
        "rejected_hits": report.rejected_hits,
        "misses": report.misses,
        "overall_hit_rate": report.overall_hit_rate,
        "cache_eligible_hit_rate": report.cache_eligible_hit_rate,
        "false_reuse": report.false_reuse,
        "false_reuse_rate": report.false_reuse_rate,
        "llm_calls_saved": report.llm_calls_saved,
        "cost_saved_usd": report.cost_saved_usd,
        "latency_saved_ms": report.latency_saved_ms,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
