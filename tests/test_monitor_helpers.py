from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from contracts.harness import ComplaintCase, HarnessRun, SocialPost
from contracts.models import Category, RiskLevel
from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.orchestrator import HarnessOrchestrator
from services.monitor.models import MonitorCase, ScanRequest
from services.monitor.store import MonitorStore


NOW = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)


def _run(text: str) -> tuple[SocialPost, HarnessRun]:
    post = SocialPost(
        post_id=f"post-{abs(hash(text)) % 100000}",
        platform="x",
        url="https://x.com/example/status/1",
        author_handle="warga",
        text=text,
        posted_at=NOW - timedelta(hours=2),
        collected_at=NOW,
    )
    run = HarnessOrchestrator(
        triage=RuleTriageAgent(),
        extraction=RuleExtractionAgent(),
        verifier=RuleVerifierAgent(),
        arbiter=RuleArbiterAgent(),
    ).run(post, run_id="run-monitor-test")
    return post, run


def _monitor_case(
    post: SocialPost,
    run: HarnessRun,
    **overrides: object,
) -> MonitorCase:
    case = run.case
    base = {
        "case_id": case.case_id if case is not None else "case-x-1",
        "post": post,
        "run": run,
        "viral_score": 50 if case is not None else None,
        "department": "Dinas Sumber Daya Air dan Bina Marga (DSDABM)"
        if case is not None
        else None,
        "first_seen_at": NOW,
        "last_seen_at": NOW,
        "updated_at": NOW,
    }
    base.update(overrides)
    return MonitorCase(**base)


def _complaint_case(
    *,
    category: Category = Category.ROAD,
    risk: str = "MEDIUM",
    status: str = "ACCEPTED",
) -> ComplaintCase:
    _, run = _run("Min, jalan di Jl. Kopo rusak parah, ada lubang besar.")
    assert run.case is not None
    case = run.case
    return case.model_copy(
        update={
            "triage": case.triage.model_copy(
                update={"category": category, "risk": RiskLevel(risk)}
            ),
            "status": status,
        }
    )


def _store_with_case(tmp_path: Path, **overrides: object) -> MonitorStore:
    store = MonitorStore(tmp_path / "monitor.db")
    store.create_schema()
    post, run = _run("Min, jalan di Jl. Kopo rusak parah, ada lubang besar.")
    case = _monitor_case(post, run, case_id="case-x-1", **overrides)
    store.upsert_case(case, "jalan rusak")
    return store


def _write_fixture(directory: Path, post_id: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{post_id}.json").write_text(
        json.dumps(
            {
                "post": {
                    "post_id": post_id,
                    "platform": "x",
                    "url": f"https://x.com/example/status/{post_id}",
                    "author_handle": "warga",
                    "text": text,
                    "posted_at": NOW.isoformat(),
                    "collected_at": NOW.isoformat(),
                    "raw": {"likes": 100, "views": 1000},
                }
            }
        ),
        encoding="utf-8",
    )
    return directory


def _request(directory: Path, query: str) -> ScanRequest:
    return ScanRequest(
        query=query,
        limit=10,
        source_mode="replay",
        analysis_mode="rule",
        fixture_dir=str(directory),
    )


def _post(
    *,
    likes: int | None = 100,
    views: int | None = 1000,
    posted_at: datetime = NOW - timedelta(hours=1),
) -> SocialPost:
    return SocialPost(
        post_id="score-post",
        platform="x",
        url="https://x.com/example/status/1",
        author_handle="warga",
        text="Jalan rusak di Jl. Kopo.",
        posted_at=posted_at,
        collected_at=NOW,
        raw={"likes": likes, "views": views},
    )


def _monitor_case_with_note() -> MonitorCase:
    post, run = _run("Min, jalan di Jl. Kopo rusak parah, ada lubang besar.")
    return _monitor_case(post, run, notes="Sudah dicek")
