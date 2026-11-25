from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from contracts.harness import HarnessRun, SocialPost
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sinks import FileRunSink


class BatchItemSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    post_id: str
    run_id: str
    case_id: str | None = None
    status: str
    trust_decision: str | None = None
    category: str | None = None
    is_complaint: bool | None = None
    location_quality: str | None = None
    source_intent: str | None = None
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None


class BatchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str
    output_dir: str
    source: str
    query: str | None = None
    total_posts: int = Field(ge=0)
    accepted: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    rejected: int = Field(ge=0)
    failed: int = Field(ge=0)
    budget_exceeded: int = Field(ge=0)
    categories: dict[str, int] = Field(default_factory=dict)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    items: tuple[BatchItemSummary, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")
    return (cleaned or "post")[:120]


class HarnessBatchRunner:
    def __init__(
        self,
        root: str | Path,
        orchestrator: HarnessOrchestrator,
        fixture_out: str | Path | None = None,
    ) -> None:
        self._root = Path(root)
        self._orchestrator = orchestrator
        self._fixture_out = Path(fixture_out) if fixture_out is not None else None

    def run(
        self,
        posts: Sequence[SocialPost],
        *,
        batch_id: str | None = None,
        source: str = "replay",
        query: str | None = None,
    ) -> BatchReport:
        resolved_batch_id = batch_id or uuid4().hex[:12]
        batch_dir = self._root / f"batch-{resolved_batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        sink = FileRunSink(batch_dir)

        if self._fixture_out is not None:
            self._save_fixtures(posts)

        items: list[BatchItemSummary] = []
        for post in posts:
            try:
                run = self._orchestrator.run(post)
            except Exception as exc:
                items.append(
                    BatchItemSummary(
                        post_id=post.post_id,
                        run_id=f"failed-{uuid4().hex[:8]}",
                        status="FAILED",
                        error=type(exc).__name__,
                    )
                )
                continue

            sink.write(run)
            items.append(self._summarize(post, run))

        categories = Counter(item.category for item in items if item.category)
        report = BatchReport(
            batch_id=resolved_batch_id,
            output_dir=str(batch_dir),
            source=source,
            query=query,
            total_posts=len(posts),
            accepted=sum(item.status == "COMPLETED" for item in items),
            needs_review=sum(item.status == "NEEDS_REVIEW" for item in items),
            rejected=sum(item.status == "REJECTED" for item in items),
            failed=sum(item.status == "FAILED" for item in items),
            budget_exceeded=sum(item.status == "BUDGET_EXCEEDED" for item in items),
            categories=dict(sorted(categories.items())),
            total_cost_usd=sum(item.cost_usd for item in items),
            total_latency_ms=sum(item.latency_ms for item in items),
            items=tuple(items),
        )
        (batch_dir / "summary.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (batch_dir / "summary.md").write_text(
            self._render_markdown(report), encoding="utf-8"
        )
        return report

    def _save_fixtures(self, posts: Sequence[SocialPost]) -> None:
        assert self._fixture_out is not None
        self._fixture_out.mkdir(parents=True, exist_ok=True)
        for post in posts:
            filename = self._fixture_out / f"x_{_safe_filename(post.post_id)}.json"
            filename.write_text(
                json.dumps(
                    {"post": post.model_dump(mode="json")}, indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )

    @staticmethod
    def _summarize(post: SocialPost, run: HarnessRun) -> BatchItemSummary:
        case = run.case
        return BatchItemSummary(
            post_id=post.post_id,
            run_id=run.run_id,
            case_id=case.case_id if case is not None else None,
            status=run.status,
            trust_decision=case.trust_decision if case is not None else None,
            category=case.triage.category.value if case is not None else None,
            is_complaint=case.triage.is_complaint if case is not None else None,
            location_quality=case.location_quality if case is not None else None,
            source_intent=case.source_intent if case is not None else None,
            cost_usd=run.cost.total_cost_usd,
            latency_ms=run.cost.total_latency_ms,
        )

    @staticmethod
    def _render_markdown(report: BatchReport) -> str:
        lines = [
            f"# Harness Batch {report.batch_id}",
            "",
            f"- Source: `{report.source}`",
            f"- Query: `{report.query or '-'}`",
            f"- Posts: {report.total_posts}",
            f"- Accepted: {report.accepted}",
            f"- Needs review: {report.needs_review}",
            f"- Rejected: {report.rejected}",
            f"- Failed: {report.failed}",
            f"- Budget exceeded: {report.budget_exceeded}",
            f"- Total cost: ${report.total_cost_usd:.6f}",
            f"- Total latency: {report.total_latency_ms:.1f} ms",
            "",
            "## Categories",
            "",
        ]
        if report.categories:
            lines.extend(
                f"- {name}: {count}" for name, count in report.categories.items()
            )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Items",
                "",
                "| Case | Status | Category | Cost |",
                "|---|---|---|---:|",
            ]
        )
        for item in report.items:
            lines.append(
                f"| {item.case_id or '-'} | {item.status} | {item.category or '-'} | ${item.cost_usd:.6f} |"
            )
        return "\n".join(lines) + "\n"
