from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from contracts.harness import SocialPost
from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.batch import HarnessBatchRunner
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sources import ReplayXSource


REAL_FIXTURES = Path(__file__).parent.parent / "fixtures" / "x_posts"


def _post(post_id: str, text: str) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform="x",
        url=f"https://x.com/example/status/{post_id}",
        author_handle="warga",
        text=text,
        collected_at=datetime(2026, 11, 25, tzinfo=timezone.utc),
    )


def _orchestrator() -> HarnessOrchestrator:
    return HarnessOrchestrator(
        triage=RuleTriageAgent(),
        extraction=RuleExtractionAgent(),
        verifier=RuleVerifierAgent(),
        arbiter=RuleArbiterAgent(),
        max_cost_usd=0.05,
    )


def test_batch_counts_and_writes_summary(tmp_path: Path) -> None:
    posts = [
        _post(
            "accepted-1",
            "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar. Tolong diperbaiki.",
        ),
        _post("rejected-1", "Selamat pagi min"),
        _post("review-1", "Lapor jalan rusak parah di Bandung, ada lubang besar"),
    ]
    runner = HarnessBatchRunner(
        tmp_path,
        _orchestrator(),
        fixture_out=tmp_path / "fixtures",
    )
    report = runner.run(
        posts,
        batch_id="batch-test-1",
        source="replay",
        query="jalan rusak bandung",
    )

    assert report.total_posts == 3
    assert report.accepted == 1
    assert report.rejected == 1
    assert report.needs_review == 1
    assert report.failed == 0
    assert len(report.items) == 3

    batch_dir = tmp_path / "batch-batch-test-1"
    assert (batch_dir / "summary.json").is_file()
    assert (batch_dir / "summary.md").is_file()
    summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted"] == 1
    assert summary["needs_review"] == 1
    assert len(list((tmp_path / "fixtures").glob("*.json"))) == 3


def test_batch_real_fixture_directory(tmp_path: Path) -> None:
    posts = ReplayXSource(REAL_FIXTURES).fetch(limit=100)
    assert len(posts) >= 2

    report = HarnessBatchRunner(tmp_path, _orchestrator()).run(
        posts,
        batch_id="batch-real-1",
        source="replay",
    )
    assert report.total_posts == len(posts)
    assert report.accepted >= 2
    assert report.failed == 0


def test_batch_isolates_orchestrator_exception(tmp_path: Path) -> None:
    class BrokenOrchestrator:
        def run(self, post: SocialPost) -> object:
            del post
            raise RuntimeError("boom")

    runner = HarnessBatchRunner(tmp_path, BrokenOrchestrator())  # type: ignore[arg-type]
    report = runner.run(
        [_post("broken-1", "Jalan rusak")],
        batch_id="batch-test-2",
    )

    assert report.total_posts == 1
    assert report.failed == 1
    assert report.items[0].error == "RuntimeError"
    assert (tmp_path / "batch-batch-test-2" / "summary.json").is_file()
