from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from contracts.harness import SocialPost
from services.harness.batch import BatchItemSummary, HarnessBatchRunner
from services.harness.orchestrator import HarnessOrchestrator


class ComparisonItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    post_id: str
    rule_status: str
    rule_category: str | None = None
    llm_status: str
    llm_category: str | None = None
    rule_is_complaint: bool | None = None
    llm_is_complaint: bool | None = None
    rule_location_quality: str | None = None
    llm_location_quality: str | None = None
    agreement: bool
    rule_cost_usd: float = Field(default=0.0, ge=0.0)
    llm_cost_usd: float = Field(default=0.0, ge=0.0)
    rule_latency_ms: float = Field(default=0.0, ge=0.0)
    llm_latency_ms: float = Field(default=0.0, ge=0.0)


class ComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_id: str
    output_dir: str
    total_posts: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    agreement_rate: float = Field(ge=0.0, le=1.0)
    rule_accepted: int = Field(ge=0)
    llm_accepted: int = Field(ge=0)
    rule_needs_review: int = Field(ge=0)
    llm_needs_review: int = Field(ge=0)
    rule_failed: int = Field(ge=0)
    llm_failed: int = Field(ge=0)
    rule_total_cost_usd: float = Field(ge=0.0)
    llm_total_cost_usd: float = Field(ge=0.0)
    rule_total_latency_ms: float = Field(ge=0.0)
    llm_total_latency_ms: float = Field(ge=0.0)
    items: tuple[ComparisonItem, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HarnessComparisonRunner:
    def __init__(
        self,
        root: str | Path,
        rule_orchestrator: HarnessOrchestrator,
        llm_orchestrator: HarnessOrchestrator,
    ) -> None:
        self._root = Path(root)
        self._rule_orchestrator = rule_orchestrator
        self._llm_orchestrator = llm_orchestrator

    def run(
        self,
        posts: Sequence[SocialPost],
        *,
        comparison_id: str | None = None,
        source: str = "replay",
    ) -> ComparisonReport:
        resolved_id = comparison_id or uuid4().hex[:12]
        output_dir = self._root / f"comparison-{resolved_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        rule_report = HarnessBatchRunner(
            output_dir / "rule", self._rule_orchestrator
        ).run(posts, batch_id=f"{resolved_id}-rule", source=source)
        llm_report = HarnessBatchRunner(output_dir / "llm", self._llm_orchestrator).run(
            posts, batch_id=f"{resolved_id}-llm", source=source
        )

        rule_by_id = {item.post_id: item for item in rule_report.items}
        llm_by_id = {item.post_id: item for item in llm_report.items}
        items: list[ComparisonItem] = []
        for post in posts:
            rule_item = rule_by_id.get(post.post_id) or self._missing(post, "rule")
            llm_item = llm_by_id.get(post.post_id) or self._missing(post, "llm")
            items.append(
                ComparisonItem(
                    post_id=post.post_id,
                    rule_status=rule_item.status,
                    rule_category=rule_item.category,
                    llm_status=llm_item.status,
                    llm_category=llm_item.category,
                    rule_is_complaint=rule_item.is_complaint,
                    llm_is_complaint=llm_item.is_complaint,
                    rule_location_quality=rule_item.location_quality,
                    llm_location_quality=llm_item.location_quality,
                    agreement=(
                        rule_item.status == llm_item.status
                        and rule_item.category == llm_item.category
                    ),
                    rule_cost_usd=rule_item.cost_usd,
                    llm_cost_usd=llm_item.cost_usd,
                    rule_latency_ms=rule_item.latency_ms,
                    llm_latency_ms=llm_item.latency_ms,
                )
            )

        agreement_count = sum(item.agreement for item in items)
        report = ComparisonReport(
            comparison_id=resolved_id,
            output_dir=str(output_dir),
            total_posts=len(posts),
            agreement_count=agreement_count,
            agreement_rate=(agreement_count / len(posts)) if posts else 0.0,
            rule_accepted=rule_report.accepted,
            llm_accepted=llm_report.accepted,
            rule_needs_review=rule_report.needs_review,
            llm_needs_review=llm_report.needs_review,
            rule_failed=rule_report.failed,
            llm_failed=llm_report.failed,
            rule_total_cost_usd=rule_report.total_cost_usd,
            llm_total_cost_usd=llm_report.total_cost_usd,
            rule_total_latency_ms=rule_report.total_latency_ms,
            llm_total_latency_ms=llm_report.total_latency_ms,
            items=tuple(items),
        )
        (output_dir / "comparison.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "comparison.md").write_text(
            self._render_markdown(report), encoding="utf-8"
        )
        return report

    @staticmethod
    def _missing(post: SocialPost, mode: str) -> BatchItemSummary:
        return BatchItemSummary(
            post_id=post.post_id,
            run_id=f"missing-{mode}",
            status="FAILED",
            error="MISSING_RESULT",
        )

    @staticmethod
    def _render_markdown(report: ComparisonReport) -> str:
        lines = [
            f"# Rule vs LLM Comparison {report.comparison_id}",
            "",
            f"- Posts: {report.total_posts}",
            f"- Agreement: {report.agreement_count}/{report.total_posts} "
            f"({report.agreement_rate:.1%})",
            f"- Rule accepted / review / failed: "
            f"{report.rule_accepted} / {report.rule_needs_review} / {report.rule_failed}",
            f"- LLM accepted / review / failed: "
            f"{report.llm_accepted} / {report.llm_needs_review} / {report.llm_failed}",
            f"- Rule cost: ${report.rule_total_cost_usd:.6f}",
            f"- LLM cost: ${report.llm_total_cost_usd:.6f}",
            f"- Rule latency: {report.rule_total_latency_ms:.1f} ms",
            f"- LLM latency: {report.llm_total_latency_ms:.1f} ms",
            "",
            "## Items",
            "",
            "| Post | Rule | LLM | Agree |",
            "|---|---|---|---|",
        ]
        for item in report.items:
            lines.append(
                f"| {item.post_id} | {item.rule_status}/{item.rule_category or '-'} "
                f"| {item.llm_status}/{item.llm_category or '-'} | {item.agreement} |"
            )
        return "\n".join(lines) + "\n"
