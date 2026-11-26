from __future__ import annotations

from datetime import datetime, timezone

from contracts.harness import (
    ArbitrationOutput,
    ExtractionOutput,
    SocialPost,
    TriageOutput,
    VerificationOutput,
)
from contracts.models import Category, RiskLevel
from services.harness.agents import (
    PricedStubAgent,
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.orchestrator import HarnessOrchestrator


TEXT = "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar. Tolong segera diperbaiki."


def _post(text: str = TEXT) -> SocialPost:
    return SocialPost(
        post_id="post-1",
        platform="x",
        url="https://x.com/example/status/1",
        author_handle="warga",
        text=text,
        collected_at=datetime(2026, 11, 25, tzinfo=timezone.utc),
    )


def _orchestrator(**kwargs: object) -> HarnessOrchestrator:
    triage = kwargs.pop("triage", RuleTriageAgent())
    extraction = kwargs.pop("extraction", RuleExtractionAgent())
    verifier = kwargs.pop("verifier", RuleVerifierAgent())
    arbiter = kwargs.pop("arbiter", RuleArbiterAgent())
    max_cost_usd = kwargs.pop("max_cost_usd", 0.05)
    return HarnessOrchestrator(
        triage=triage,
        extraction=extraction,
        verifier=verifier,
        arbiter=arbiter,
        max_cost_usd=max_cost_usd,
    )


def test_orchestrator_passes_all_categories_to_agents(monkeypatch) -> None:
    seen: list[list[str]] = []
    original_triage = RuleTriageAgent.run
    original_verifier = RuleVerifierAgent.run

    def capture_triage(self, task):
        seen.append(list(task.context.get("allowed_categories", [])))
        return original_triage(self, task)

    def capture_verifier(self, task):
        seen.append(list(task.context.get("allowed_categories", [])))
        return original_verifier(self, task)

    monkeypatch.setattr(RuleTriageAgent, "run", capture_triage)
    monkeypatch.setattr(RuleVerifierAgent, "run", capture_verifier)
    post = _post("Jl. Kopo rusak parah, tolong diperbaiki.")
    _orchestrator().run(post)
    assert set(seen[0]) == {category.value for category in Category}
    assert set(seen[1]) == {category.value for category in Category}


def test_unknown_category_requires_review() -> None:
    post = _post("Tolong tindak lanjut laporan warga di Bandung.")
    run = _orchestrator().run(post)
    assert run.status == "NEEDS_REVIEW"
    assert run.case is not None
    assert "CATEGORY_UNKNOWN" in run.case.reason_codes


def test_ambiguous_category_requires_review() -> None:
    post = _post("Pungli parkir liar di trotoar yang rusak dan jalan berlubang di Jl. Kopo.")
    run = _orchestrator().run(post)
    assert run.status == "NEEDS_REVIEW"
    assert run.case is not None
    assert "AMBIGUOUS_CATEGORY" in run.case.reason_codes


def test_agreement_produces_accepted_case() -> None:
    run = _orchestrator().run(_post())
    assert run.status == "COMPLETED"
    assert run.case is not None
    assert run.case.status == "ACCEPTED"
    assert run.case.trust_decision == "AGREED"
    assert run.trace
    assert run.cost.calls == 3


def test_non_complaint_produces_rejected_case() -> None:
    run = _orchestrator().run(_post("Selamat pagi min"))
    assert run.status == "REJECTED"
    assert run.case is not None
    assert run.case.status == "REJECTED"
    assert run.case.trust_decision == "REJECTED"


def test_verifier_disagreement_stays_needs_review() -> None:
    verifier = PricedStubAgent(
        role="verifier",
        output=VerificationOutput(
            is_complaint=True,
            category=Category.WASTE,
            evidence_supported=True,
            issues=(),
            confidence=0.8,
        ).model_dump(mode="json"),
    )
    arbiter = PricedStubAgent(
        role="arbiter",
        output=ArbitrationOutput(
            is_complaint=True,
            category=Category.ROAD,
            rationale="Triage is more likely correct.",
            confidence=0.7,
        ).model_dump(mode="json"),
    )
    run = _orchestrator(verifier=verifier, arbiter=arbiter).run(_post())
    assert run.status == "NEEDS_REVIEW"
    assert run.case is not None
    assert run.case.status == "NEEDS_REVIEW"
    assert run.case.trust_decision == "NEEDS_REVIEW"
    assert run.case.arbitration is not None
    assert run.cost.arbiter_calls == 1


def test_missing_location_stays_needs_review() -> None:
    extraction = PricedStubAgent(
        role="extraction",
        output=ExtractionOutput(
            location=None,
            object="lubang",
            time=None,
            evidence_spans=("lubang",),
            confidence=0.5,
        ).model_dump(mode="json"),
    )
    run = _orchestrator(extraction=extraction).run(_post())
    assert run.status == "NEEDS_REVIEW"
    assert run.case is not None
    assert run.case.status == "NEEDS_REVIEW"
    assert "LOCATION_MISSING" in run.case.reason_codes


def test_budget_exhaustion_before_triage_has_no_case() -> None:
    triage = PricedStubAgent(
        role="triage",
        output=TriageOutput(
            is_complaint=True,
            category=Category.ROAD,
            risk=RiskLevel.MEDIUM,
            confidence=0.9,
        ).model_dump(mode="json"),
        estimate_usd=1.0,
    )
    run = _orchestrator(triage=triage, max_cost_usd=0.01).run(_post())
    assert run.status == "BUDGET_EXCEEDED"
    assert run.case is None
    assert run.trace
    assert run.cost.budget_status == "EXCEEDED"


def test_actual_cost_overrun_is_recorded_and_stops() -> None:
    triage = PricedStubAgent(
        role="triage",
        output=TriageOutput(
            is_complaint=True,
            category=Category.ROAD,
            risk=RiskLevel.MEDIUM,
            confidence=0.9,
        ).model_dump(mode="json"),
        estimate_usd=0.0,
        cost_usd=0.02,
    )
    run = _orchestrator(triage=triage, max_cost_usd=0.01).run(_post())
    assert run.status == "BUDGET_EXCEEDED"
    assert run.case is None
    assert run.cost.total_cost_usd == 0.02
    assert run.cost.budget_status == "EXCEEDED"


def test_empty_post_fails_with_trace() -> None:
    run = _orchestrator().run(_post("   "))
    assert run.status == "FAILED"
    assert run.case is None
    assert run.trace


def test_verifier_unavailable_has_no_case() -> None:
    verifier = PricedStubAgent(role="verifier", status="UNAVAILABLE")
    run = _orchestrator(verifier=verifier).run(_post())
    assert run.status == "NEEDS_REVIEW"
    assert run.case is None
    assert run.trace


def test_arbiter_does_not_flip_disagreement() -> None:
    verifier = PricedStubAgent(
        role="verifier",
        output=VerificationOutput(
            is_complaint=True,
            category=Category.WASTE,
            evidence_supported=True,
            issues=(),
            confidence=0.8,
        ).model_dump(mode="json"),
    )
    arbiter = PricedStubAgent(
        role="arbiter",
        output=ArbitrationOutput(
            is_complaint=True,
            category=Category.ROAD,
            rationale="Arbiter agrees with triage.",
            confidence=0.99,
        ).model_dump(mode="json"),
    )
    run = _orchestrator(verifier=verifier, arbiter=arbiter).run(_post())
    assert run.case is not None
    assert run.case.status != "ACCEPTED"
    assert run.case.trust_decision == "NEEDS_REVIEW"


def test_news_report_is_rejected_as_non_actionable() -> None:
    run = _orchestrator().run(
        _post(
            "Perbaikan Tutup Jalan Utama Saguling KBB. Jalan rusak dan berlubang "
            "memaksa warga mencari jalur alternatif; pengendara diuji adrenalin."
        )
    )
    assert run.case is not None
    assert run.case.source_intent == "NEWS_REPORT"
    assert run.case.status == "REJECTED"
    assert "SOURCE_NOT_ACTIONABLE" in run.case.reason_codes


def test_opinion_is_rejected_as_non_actionable() -> None:
    run = _orchestrator().run(
        _post(
            "Menurut saya, Bandung gada perubahan signifikan sama sekali. "
            "Jalan rusak dibiarkan dan pembangunan tidak merata."
        )
    )
    assert run.case is not None
    assert run.case.source_intent == "OPINION"
    assert run.case.status == "REJECTED"
    assert "SOURCE_NOT_ACTIONABLE" in run.case.reason_codes
