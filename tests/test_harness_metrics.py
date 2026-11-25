from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from contracts.harness import SocialPost
from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.compare import HarnessComparisonRunner
from services.harness.metrics import LabelSet, load_labels, score_comparison
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sources import ReplayXSource


FIXTURES = Path(__file__).parent.parent / "fixtures" / "x_posts"
LABELS = Path(__file__).parent.parent / "fixtures" / "x_labels.json"


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


def _labels() -> LabelSet:
    return LabelSet.model_validate(
        {
            "labels": {
                "metric-accepted": {
                    "is_complaint": True,
                    "category": "ROAD",
                    "location_quality": "SPECIFIC",
                    "acceptable_statuses": ["COMPLETED"],
                    "notes": "",
                },
                "metric-rejected": {
                    "is_complaint": False,
                    "category": None,
                    "location_quality": "MISSING",
                    "acceptable_statuses": ["REJECTED"],
                    "notes": "",
                },
            }
        }
    )


def _comparison(tmp_path: Path) -> object:
    posts = [
        _post(
            "metric-accepted",
            "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar.",
        ),
        _post("metric-rejected", "Selamat pagi min"),
    ]
    return HarnessComparisonRunner(tmp_path, _orchestrator(), _orchestrator()).run(
        posts, comparison_id="metric-comparison"
    )


def test_score_comparison_rewards_correct_decisions(tmp_path: Path) -> None:
    report = _comparison(tmp_path)
    metrics = score_comparison(report, _labels())  # type: ignore[arg-type]
    assert metrics.labels_count == 2
    assert metrics.rule.decision_accuracy == 1.0
    assert metrics.rule.complaint_precision == 1.0
    assert metrics.rule.complaint_recall == 1.0
    assert metrics.rule.category_accuracy == 1.0
    assert metrics.rule.location_accuracy == 1.0
    assert metrics.rule.false_accepts == 0


def test_score_comparison_penalizes_false_accept(tmp_path: Path) -> None:
    report = _comparison(tmp_path)
    labels = _labels()
    labels = labels.model_copy(
        update={
            "labels": {
                **labels.labels,
                "metric-accepted": labels.labels["metric-accepted"].model_copy(
                    update={"acceptable_statuses": ("NEEDS_REVIEW",)}
                ),
            }
        }
    )
    metrics = score_comparison(report, labels)  # type: ignore[arg-type]
    assert metrics.rule.false_accepts == 1
    assert metrics.rule.decision_accuracy == 0.5


def test_labels_cover_all_real_fixtures() -> None:
    posts = ReplayXSource(FIXTURES).fetch(limit=100)
    labels = load_labels(LABELS)
    assert {post.post_id for post in posts} == set(labels.labels)
