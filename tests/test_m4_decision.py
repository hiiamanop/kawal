from __future__ import annotations

import pytest

from contracts.models import DecisionMode
from services.core.decision import (
    BudgetState,
    ConflictInput,
    ConflictSeverity,
    ConflictType,
    DecisionInput,
    EgressDecision,
    EgressRequest,
    EscalationCandidate,
    ResolutionCapability,
    TaskResultCache,
    TrustContext,
    TrustObservation,
    decide,
    diagnose_conflict,
    estimate_contextual_trust,
    evaluate_egress,
    expected_value_of_information,
)


def _budget() -> BudgetState:
    return BudgetState(remaining_usd=1.0, remaining_latency_ms=5_000.0, remaining_egress_bytes=1_000_000)


def _input(**changes: object) -> DecisionInput:
    data: dict[str, object] = {
        "case_id": "case-1",
        "revision": 1,
        "required_fields_complete": True,
        "evidence_covered": True,
        "authority_valid": True,
        "policy_allowed": True,
        "is_complaint": True,
        "budget": _budget(),
        "clarification_available": True,
    }
    data.update(changes)
    return DecisionInput(**data)


def test_trust_recomputes_deterministically_and_is_conservative() -> None:
    context = TrustContext(agent_version="v1", task="category", category="ROAD", language_style="formal", risk_band="HIGH")
    observation = TrustObservation(context=context, successes=36, total=40, ece=0.05, drift_penalty=0.1)
    first = estimate_contextual_trust(observation, parent_successes=90, parent_total=100, evidence_quality=0.9)
    second = estimate_contextual_trust(observation, parent_successes=90, parent_total=100, evidence_quality=0.9)
    assert first == second
    assert first.lower_bound < first.posterior_mean
    assert first.reliability_weight < first.lower_bound


def test_cold_start_uses_parent_shrinkage_and_marks_it() -> None:
    result = estimate_contextual_trust(None, parent_successes=9, parent_total=10, evidence_quality=1.0)
    assert result.used_parent is True
    assert result.support == 0
    assert 0.0 < result.reliability_weight < 1.0


def test_authority_conflict_is_hard_and_needs_policy_correction() -> None:
    conflict = diagnose_conflict(ConflictInput(
        type=ConflictType.AUTHORITY,
        field="authority_unit_id",
        left_value="UNIT-A",
        right_value="UNIT-B",
        left_trust=0.1,
        right_trust=0.1,
        consequence_weight=0.1,
        evidence_refs=("directory", "model"),
    ))
    assert conflict is not None
    assert conflict.severity == ConflictSeverity.HARD
    assert conflict.resolvable_by == ResolutionCapability.POLICY_CORRECTION


@pytest.mark.parametrize("type_", list(ConflictType))
def test_all_conflict_types_preserve_both_evidence_sides(type_: ConflictType) -> None:
    conflict = diagnose_conflict(ConflictInput(
        type=type_, field="field", left_value="left", right_value="right",
        left_trust=0.8, right_trust=0.7, consequence_weight=0.8,
        evidence_refs=("result-left", "result-right"), visual_required=type_ == ConflictType.TEXT_IMAGE,
    ))
    assert conflict is not None
    assert conflict.evidence_refs == ("result-left", "result-right")


def test_non_complaint_is_rejected() -> None:
    assert decide(_input(is_complaint=False)).mode == DecisionMode.REJECT_IGNORE


def test_ready_case_executes() -> None:
    plan = decide(_input())
    assert plan.mode == DecisionMode.EXECUTE
    assert plan.next_state == "EXECUTING"


def test_missing_field_prefers_positive_voi_then_stable_task_id() -> None:
    candidates = (
        EscalationCandidate(task_id="z-task", expected_utility_gain=5.0, cost_usd=0.1, latency_ms=100.0, egress_bytes=0, queue_delay_ms=0.0, policy_allowed=True),
        EscalationCandidate(task_id="a-task", expected_utility_gain=5.0, cost_usd=0.1, latency_ms=100.0, egress_bytes=0, queue_delay_ms=0.0, policy_allowed=True),
    )
    plan = decide(_input(required_fields_complete=False, escalation_candidates=candidates))
    assert plan.mode == DecisionMode.RE_EVALUATE
    assert plan.selected_task_id == "a-task"
    assert plan.reserved_cost_usd == 0.1


def test_missing_field_requests_clarification_when_voi_is_not_positive() -> None:
    candidate = EscalationCandidate(task_id="low", expected_utility_gain=0.01, cost_usd=1.0, latency_ms=100.0, egress_bytes=0, queue_delay_ms=0.0, policy_allowed=True)
    plan = decide(_input(required_fields_complete=False, escalation_candidates=(candidate,)))
    assert plan.mode == DecisionMode.REQUEST_CLARIFICATION


def test_disallowed_candidate_has_negative_infinite_voi() -> None:
    candidate = EscalationCandidate(task_id="remote", expected_utility_gain=99.0, cost_usd=0.0, latency_ms=0.0, egress_bytes=0, queue_delay_ms=0.0, policy_allowed=False)
    assert expected_value_of_information(candidate) == float("-inf")


def test_decision_trace_reproducible_within_float_tolerance() -> None:
    context = TrustContext(agent_version="v1", task="category", category="ROAD", language_style="formal", risk_band="HIGH")
    observation = TrustObservation(context=context, successes=72, total=80, ece=0.04, drift_penalty=0.02)
    r1 = estimate_contextual_trust(observation, parent_successes=180, parent_total=200, evidence_quality=0.95)
    r2 = estimate_contextual_trust(observation, parent_successes=180, parent_total=200, evidence_quality=0.95)
    assert abs(r1.reliability_weight - r2.reliability_weight) < 1e-6
    plan_a = decide(_input())
    plan_b = decide(_input())
    assert plan_a == plan_b


def test_identical_task_ids_are_deduplicated() -> None:
    dup = EscalationCandidate(task_id="same-task", expected_utility_gain=5.0, cost_usd=0.1, latency_ms=100.0, egress_bytes=0, queue_delay_ms=0.0, policy_allowed=True)
    plan = decide(_input(required_fields_complete=False, escalation_candidates=(dup, dup)))
    assert plan.mode == DecisionMode.RE_EVALUATE
    assert plan.selected_task_id == "same-task"


def test_hard_conflict_blocks_execution() -> None:
    hard = diagnose_conflict(ConflictInput(
        type=ConflictType.AUTHORITY, field="authority_unit_id", left_value="A", right_value="B",
        left_trust=0.99, right_trust=0.99, consequence_weight=1.0, evidence_refs=("x", "y"),
    ))
    assert hard is not None
    plan = decide(_input(conflicts=(hard,)))
    assert plan.mode == DecisionMode.RE_EVALUATE


def test_egress_allowed_for_compliant_request() -> None:
    decision = evaluate_egress(EgressRequest(
        provider="openai", model_id="gpt-4o", data_class="public", cost_usd=0.01,
    ))
    assert decision.allowed is True


def test_egress_denied_unknown_provider() -> None:
    decision = evaluate_egress(EgressRequest(
        provider="malicious-corp", model_id="evil-model", data_class="public",
    ))
    assert decision.allowed is False
    assert "PROVIDER_NOT_ON_ALLOWLIST" in decision.reason_codes


def test_egress_denied_pii() -> None:
    decision = evaluate_egress(EgressRequest(
        provider="openai", model_id="gpt-4o", data_class="public", has_pii=True,
    ))
    assert decision.allowed is False
    assert "PII_EGRESS_BLOCKED" in decision.reason_codes


def test_egress_denied_sensitive_image() -> None:
    decision = evaluate_egress(EgressRequest(
        provider="openai", model_id="gpt-4o-vision", data_class="public",
        attachment_mime="image/jpeg", category_sensitive=True,
    ))
    assert decision.allowed is False
    assert "SENSITIVE_IMAGE_EGRESS_DENIED" in decision.reason_codes


def test_egress_denied_executable_mime() -> None:
    decision = evaluate_egress(EgressRequest(
        provider="openai", model_id="gpt-4o", data_class="public",
        attachment_mime="application/x-executable",
    ))
    assert decision.allowed is False
    assert "BLOCKED_MIME_TYPE" in decision.reason_codes


def test_egress_denied_budget_exceeded() -> None:
    decision = evaluate_egress(EgressRequest(
        provider="openai", model_id="gpt-4o", data_class="public",
        cost_usd=5.0, budget=BudgetState(remaining_usd=1.0, remaining_latency_ms=5000.0, remaining_egress_bytes=1_000_000),
    ))
    assert decision.allowed is False
    assert "BUDGET_EXCEEDED" in decision.reason_codes


def test_egress_denied_restricted_data_class() -> None:
    decision = evaluate_egress(EgressRequest(
        provider="openai", model_id="gpt-4o", data_class="restricted",
    ))
    assert decision.allowed is False
    assert "DATA_CLASS_NOT_ALLOWED" in decision.reason_codes


def test_no_conflict_when_values_identical() -> None:
    assert diagnose_conflict(ConflictInput(
        type=ConflictType.CLASSIFICATION, field="category", left_value="ROAD", right_value="ROAD",
        left_trust=0.9, right_trust=0.9, consequence_weight=0.5, evidence_refs=("a", "b"),
    )) is None


def test_budget_over_limit_blocks_escalation_candidate() -> None:
    expensive = EscalationCandidate(task_id="big", expected_utility_gain=50.0, cost_usd=100.0, latency_ms=100.0, egress_bytes=0, queue_delay_ms=0.0, policy_allowed=True)
    plan = decide(_input(required_fields_complete=False, escalation_candidates=(expensive,)))
    assert plan.mode == DecisionMode.REQUEST_CLARIFICATION


def test_conflict_same_values_no_record() -> None:
    assert diagnose_conflict(ConflictInput(
        type=ConflictType.RISK, field="risk", left_value="HIGH", right_value="high",
        left_trust=0.9, right_trust=0.9, consequence_weight=1.0, evidence_refs=("a", "b"),
    )) is None


def test_task_result_cache_tracks_and_prevents_duplicate_calls() -> None:
    cache = TaskResultCache()
    assert cache.has("task-1") is False

    cache.put("task-1", {"category": "ROAD"})
    assert cache.has("task-1") is True
    assert cache.get("task-1") == {"category": "ROAD"}
    assert "task-1" in cache.completed_task_ids

    candidate = EscalationCandidate(
        task_id="task-1", expected_utility_gain=10.0, cost_usd=0.1,
        latency_ms=100.0, egress_bytes=0, queue_delay_ms=0.0, policy_allowed=True,
    )
    plan = decide(_input(
        required_fields_complete=False,
        escalation_candidates=(candidate,),
        completed_task_ids=cache.completed_task_ids,
    ))
    assert plan.mode == DecisionMode.REQUEST_CLARIFICATION
    assert plan.selected_task_id is None

    cache.clear()
    assert cache.has("task-1") is False
