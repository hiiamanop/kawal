from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from contracts.harness import SocialPost, VerificationOutput
from contracts.models import Category
from scripts.compare_harness_agents import main
from services.harness.agents import (
    PricedStubAgent,
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.compare import HarnessComparisonRunner
from services.harness.orchestrator import HarnessOrchestrator


FIXTURE = (
    Path(__file__).parent / "fixtures" / "x_posts" / "synthetic_road_bandung_001.json"
)


def _post(post_id: str, text: str) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform="x",
        url=f"https://x.com/example/status/{post_id}",
        author_handle="warga",
        text=text,
        collected_at=datetime(2026, 11, 25, tzinfo=timezone.utc),
    )


def _orchestrator(verifier: object | None = None) -> HarnessOrchestrator:
    return HarnessOrchestrator(
        triage=RuleTriageAgent(),
        extraction=RuleExtractionAgent(),
        verifier=verifier or RuleVerifierAgent(),  # type: ignore[arg-type]
        arbiter=RuleArbiterAgent(),
        max_cost_usd=0.05,
    )


def test_comparison_agreement(tmp_path: Path) -> None:
    posts = [
        _post(
            "cmp-accepted",
            "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar.",
        ),
        _post("cmp-rejected", "Selamat pagi min"),
    ]
    report = HarnessComparisonRunner(tmp_path, _orchestrator(), _orchestrator()).run(
        posts, comparison_id="cmp-agree"
    )

    assert report.total_posts == 2
    assert report.agreement_count == 2
    assert report.agreement_rate == 1.0
    assert report.rule_accepted == 1
    assert report.llm_accepted == 1
    output_dir = tmp_path / "comparison-cmp-agree"
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "comparison.md").is_file()


def test_comparison_disagreement(tmp_path: Path) -> None:
    disagreeing_verifier = PricedStubAgent(
        role="verifier",
        output=VerificationOutput(
            is_complaint=True,
            category=Category.WASTE,
            evidence_supported=True,
            issues=(),
            confidence=0.8,
        ).model_dump(mode="json"),
    )
    posts = [
        _post(
            "cmp-disagree",
            "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar.",
        )
    ]
    report = HarnessComparisonRunner(
        tmp_path, _orchestrator(), _orchestrator(disagreeing_verifier)
    ).run(posts, comparison_id="cmp-disagree")

    assert report.total_posts == 1
    assert report.agreement_count == 0
    assert report.agreement_rate == 0.0
    assert report.rule_accepted == 1
    assert report.llm_needs_review == 1


def test_comparison_cli_requires_openrouter_key(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("KAWAL_OPENROUTER_API_KEY", raising=False)  # type: ignore[attr-defined]

    exit_code = main(["--fixture", str(FIXTURE), "--out", str(tmp_path)])
    assert exit_code == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err  # type: ignore[attr-defined]
