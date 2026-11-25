from __future__ import annotations

from contracts.models import Category
from services.harness.agents import (
    PricedStubAgent,
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.ports import AgentTask


TEXT = "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar. Tolong segera diperbaiki."


def _task(text: str = TEXT, context: dict[str, object] | None = None) -> AgentTask:
    return AgentTask(role="test", source_text=text, context=context or {})


def test_triage_detects_road_complaint() -> None:
    result = RuleTriageAgent().run(_task())
    assert result.status == "OK"
    assert result.output is not None
    assert result.output["is_complaint"] is True
    assert result.output["category"] == Category.ROAD.value
    assert result.output["confidence"] >= 0.5


def test_triage_rejects_greeting() -> None:
    result = RuleTriageAgent().run(_task("Selamat pagi min"))
    assert result.output is not None
    assert result.output["is_complaint"] is False


def test_extraction_finds_street_and_object() -> None:
    result = RuleExtractionAgent().run(_task())
    assert result.output is not None
    assert "Kopo" in (result.output["location"] or "")
    assert result.output["object"] is not None
    assert result.output["evidence_spans"]


def test_verifier_agrees_with_clear_complaint() -> None:
    extraction = RuleExtractionAgent().run(_task())
    task = _task(context={"extraction": extraction.output})
    result = RuleVerifierAgent().run(task)
    assert result.output is not None
    assert result.output["is_complaint"] is True
    assert result.output["category"] == Category.ROAD.value
    assert result.output["evidence_supported"] is True


def test_verifier_flags_unsupported_location() -> None:
    task = _task(
        "Jalan rusak di Jl. Kopo",
        context={
            "extraction": {
                "location": "Jl. Merdeka",
                "object": "jalan rusak",
                "time": None,
                "evidence_spans": ["Jl. Merdeka", "rusak"],
                "confidence": 0.8,
            }
        },
    )
    result = RuleVerifierAgent().run(task)
    assert result.output is not None
    assert result.output["evidence_supported"] is False
    assert result.output["issues"]


def test_verifier_ignores_triage_category() -> None:
    extraction = RuleExtractionAgent().run(_task())
    task = _task(
        context={
            "extraction": extraction.output,
            "triage": {
                "is_complaint": True,
                "category": Category.WASTE.value,
                "risk": "LOW",
                "confidence": 0.9,
                "reason_codes": [],
            },
        }
    )
    result = RuleVerifierAgent().run(task)
    assert result.output is not None
    assert result.output["category"] == Category.ROAD.value


def test_arbiter_returns_explanation_only() -> None:
    result = RuleArbiterAgent().run(
        _task(
            context={
                "triage": {
                    "is_complaint": True,
                    "category": Category.ROAD.value,
                    "risk": "HIGH",
                    "confidence": 0.9,
                    "reason_codes": [],
                },
                "verification": {
                    "is_complaint": True,
                    "category": Category.WASTE.value,
                    "evidence_supported": True,
                    "issues": [],
                    "confidence": 0.8,
                },
            }
        )
    )
    assert result.status == "OK"
    assert result.output is not None
    assert "human review" in result.output["rationale"].lower()


def test_priced_stub_reports_usage() -> None:
    agent = PricedStubAgent(
        role="triage",
        output={},
        estimate_usd=0.02,
        cost_usd=0.015,
    )
    assert agent.estimate_cost(_task()) == 0.02
    result = agent.run(_task())
    assert result.usage.cost_usd == 0.015
    assert result.usage.model == "priced-stub"
