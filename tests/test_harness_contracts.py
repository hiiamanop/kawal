from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from contracts.harness import (
    AgentResult,
    AgentUsage,
    ComplaintCase,
    CostReport,
    ExtractionOutput,
    HarnessRun,
    SocialPost,
    TriageOutput,
    VerificationOutput,
    deterministic_case_id,
)
from contracts.models import Category, RiskLevel


def _post(post_id: str = "post-1") -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform="x",
        url=f"https://x.com/example/status/{post_id}",
        author_handle="tester",
        text="Jalan rusak di Jl. Kopo Bandung.",
        collected_at=datetime(2026, 11, 25, tzinfo=timezone.utc),
    )


def _triage() -> TriageOutput:
    return TriageOutput(
        is_complaint=True,
        category=Category.ROAD,
        risk=RiskLevel.MEDIUM,
        confidence=0.9,
        reason_codes=("ROAD_DAMAGE",),
    )


def _extraction() -> ExtractionOutput:
    return ExtractionOutput(
        location="Jl. Kopo",
        object="jalan rusak",
        time=None,
        evidence_spans=("Jl. Kopo", "rusak"),
        confidence=0.8,
    )


def _verification() -> VerificationOutput:
    return VerificationOutput(
        is_complaint=True,
        category=Category.ROAD,
        evidence_supported=True,
        issues=(),
        confidence=0.85,
    )


def _usage() -> AgentUsage:
    return AgentUsage(
        model="rule",
        provider="local",
        cost_source="local",
    )


def test_case_id_is_deterministic() -> None:
    post = _post()
    assert deterministic_case_id(post) == deterministic_case_id(post)
    assert deterministic_case_id(post) != deterministic_case_id(_post("post-2"))


def test_complaint_case_requires_verification() -> None:
    with pytest.raises(ValidationError):
        ComplaintCase(
            case_id="case-x-1",
            post=_post(),
            triage=_triage(),
            extraction=_extraction(),
            trust_decision="AGREED",
            status="ACCEPTED",
        )


def test_harness_run_allows_missing_case() -> None:
    run = HarnessRun(
        run_id="run-1",
        status="BUDGET_EXCEEDED",
        cost=CostReport(budget_status="EXCEEDED"),
    )
    assert run.case is None


def test_agent_result_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            agent_id="agent-1",
            role="triage",
            status="BOGUS",
            output=None,
            usage=_usage(),
            prompt_version="rule-v1",
        )


def test_complaint_case_accepts_complete_outputs() -> None:
    case = ComplaintCase(
        case_id="case-x-1",
        post=_post(),
        triage=_triage(),
        extraction=_extraction(),
        verification=_verification(),
        trust_decision="AGREED",
        status="ACCEPTED",
    )
    assert case.case_id == "case-x-1"
    assert case.status == "ACCEPTED"
