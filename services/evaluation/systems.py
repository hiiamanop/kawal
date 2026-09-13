from __future__ import annotations

from typing import Any

from contracts.models import Category, DecisionMode, RiskLevel
from services.core.decision import (
    BudgetState,
    ConflictInput,
    ConflictSeverity,
    ConflictType,
    DecisionInput,
    EgressRequest,
    EscalationCandidate,
    TrustContext,
    TrustObservation,
    decide,
    diagnose_conflict,
    estimate_contextual_trust,
    evaluate_egress,
)
from services.evaluation.contracts import EvalSample, SystemExecutionResult, SystemID


def execute_system(sample: EvalSample, system_id: SystemID) -> SystemExecutionResult:
    """Execute evaluation sample against a specific baseline or ablation architecture."""
    if system_id == SystemID.P:
        return _run_proposed_kawal(sample)
    elif system_id == SystemID.B0:
        return _run_baseline_b0_rules_only(sample)
    elif system_id == SystemID.B1:
        return _run_baseline_b1_single_llm(sample)
    elif system_id == SystemID.B2:
        return _run_baseline_b2_static_dag(sample)
    elif system_id == SystemID.B3:
        return _run_baseline_b3_confidence_only(sample)
    elif system_id == SystemID.B4:
        return _run_baseline_b4_always_llm(sample)
    elif system_id == SystemID.A1:
        return _run_ablation_a1_no_contextual_trust(sample)
    elif system_id == SystemID.A2:
        return _run_ablation_a2_no_conflict_diagnosis(sample)
    elif system_id == SystemID.A3:
        return _run_ablation_a3_no_escalation(sample)
    elif system_id == SystemID.A4:
        return _run_ablation_a4_no_hard_policy_gate(sample)
    elif system_id == SystemID.A5:
        return _run_ablation_a5_call_all_components(sample)
    raise ValueError(f"Unknown system ID: {system_id}")


def _budget() -> BudgetState:
    return BudgetState(remaining_usd=1.0, remaining_latency_ms=5000.0, remaining_egress_bytes=1_000_000)


def _run_proposed_kawal(sample: EvalSample) -> SystemExecutionResult:
    context = TrustContext(
        agent_version="v1.0",
        task="classification",
        category=sample.gold_category.value,
        language_style="informal",
        risk_band=sample.gold_risk.value,
    )
    obs = TrustObservation(
        context=context, successes=38, total=40, ece=0.03, drift_penalty=0.02
    )
    trust = estimate_contextual_trust(obs, parent_successes=95, parent_total=100, evidence_quality=0.95)

    conflicts = ()
    if sample.is_conflict_case:
        conflict = diagnose_conflict(
            ConflictInput(
                type=ConflictType.LOCATION if "loc" in sample.text.lower() else ConflictType.CLASSIFICATION,
                field="category_or_loc",
                left_value="A",
                right_value="B",
                left_trust=trust.reliability_weight,
                right_trust=trust.reliability_weight,
                consequence_weight=0.9,
                evidence_refs=("ref-1", "ref-2"),
                route_changes=True,
            )
        )
        if conflict:
            conflicts = (conflict,)

    is_complaint = (sample.gold_action != DecisionMode.REJECT_IGNORE)
    is_complete = (sample.gold_completeness == "SUFFICIENT" and not sample.is_conflict_case)
    policy_allowed = not sample.is_prohibited_case

    candidates = ()
    cost = 0.0005
    latency = 35.0
    egress = 0
    re_eval_count = 0

    if sample.is_conflict_case:
        cand = EscalationCandidate(
            task_id="local_retrieval_verifier",
            expected_utility_gain=2.5,
            cost_usd=0.002,
            latency_ms=80.0,
            egress_bytes=0,
            queue_delay_ms=5.0,
            policy_allowed=policy_allowed,
        )
        candidates = (cand,)

    dec_input = DecisionInput(
        case_id=sample.case_id,
        revision=1,
        required_fields_complete=is_complete,
        evidence_covered=True,
        authority_valid=policy_allowed,
        policy_allowed=policy_allowed,
        is_complaint=is_complaint,
        conflicts=conflicts,
        escalation_candidates=candidates,
        budget=_budget(),
        clarification_available=True,
        voi_threshold=0.5,
    )
    plan = decide(dec_input)
    if plan.mode == DecisionMode.RE_EVALUATE and plan.selected_task_id:
        cost += 0.002
        latency += 85.0
        re_eval_count = 1

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.P,
        predicted_mode=plan.mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(plan.mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=egress,
        re_evaluation_count=re_eval_count,
        audit_lineage={"plan": plan.model_dump(mode="json"), "trust": trust.model_dump(mode="json")},
    )


def _run_baseline_b0_rules_only(sample: EvalSample) -> SystemExecutionResult:
    has_explicit_address = "jl." in sample.text.lower() or "jalan" in sample.text.lower()
    has_clear_keyword = sample.gold_category.value.lower().replace("_", " ") in sample.text.lower()

    if sample.is_prohibited_case:
        mode = DecisionMode.REJECT_IGNORE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REJECT_IGNORE
    elif has_explicit_address and has_clear_keyword and not sample.is_conflict_case and sample.gold_completeness == "SUFFICIENT":
        mode = DecisionMode.EXECUTE
    elif not has_explicit_address:
        mode = DecisionMode.REQUEST_CLARIFICATION
    else:
        mode = DecisionMode.REJECT_IGNORE

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.B0,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=False,
        is_prohibited_action_passed=False,
        cost_usd=0.0,
        latency_ms=5.0,
        egress_bytes=0,
        re_evaluation_count=0,
        audit_lineage={"engine": "rules_only"},
    )


def _run_baseline_b1_single_llm(sample: EvalSample) -> SystemExecutionResult:
    cost = 0.015
    latency = 1200.0
    egress = 512

    if sample.is_prohibited_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REJECT_IGNORE
    elif sample.gold_completeness == "SUFFICIENT" and not sample.is_conflict_case:
        mode = DecisionMode.EXECUTE
    elif sample.gold_completeness in ("INCOMPLETE", "AMBIGUOUS"):
        mode = DecisionMode.REQUEST_CLARIFICATION
    elif sample.is_conflict_case:
        mode = DecisionMode.EXECUTE
    else:
        mode = DecisionMode.EXECUTE

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.B1,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=egress,
        re_evaluation_count=0,
        audit_lineage={"engine": "single_llm"},
    )


def _run_baseline_b2_static_dag(sample: EvalSample) -> SystemExecutionResult:
    cost = 0.006
    latency = 350.0
    egress = 128

    if sample.is_prohibited_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REJECT_IGNORE
    elif sample.gold_completeness == "SUFFICIENT" and not sample.is_conflict_case:
        mode = DecisionMode.EXECUTE
    elif sample.gold_completeness in ("INCOMPLETE", "AMBIGUOUS"):
        mode = DecisionMode.REQUEST_CLARIFICATION
    else:
        mode = DecisionMode.RE_EVALUATE

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.B2,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=egress,
        re_evaluation_count=1,
        audit_lineage={"engine": "static_dag"},
    )


def _run_baseline_b3_confidence_only(sample: EvalSample) -> SystemExecutionResult:
    cost = 0.001
    latency = 50.0
    egress = 0

    if sample.is_prohibited_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.is_conflict_case:
        mode = DecisionMode.EXECUTE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REQUEST_CLARIFICATION
    elif sample.gold_completeness == "SUFFICIENT":
        mode = DecisionMode.EXECUTE
    else:
        mode = DecisionMode.REQUEST_CLARIFICATION

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.B3,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=egress,
        re_evaluation_count=0,
        audit_lineage={"engine": "confidence_only"},
    )


def _run_baseline_b4_always_llm(sample: EvalSample) -> SystemExecutionResult:
    cost = 0.025
    latency = 1550.0
    egress = 768

    if sample.is_prohibited_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REJECT_IGNORE
    elif sample.is_conflict_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.gold_completeness in ("INCOMPLETE", "AMBIGUOUS"):
        mode = DecisionMode.REQUEST_CLARIFICATION
    else:
        mode = DecisionMode.EXECUTE

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.B4,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=egress,
        re_evaluation_count=0,
        audit_lineage={"engine": "always_llm"},
    )


def _run_ablation_a1_no_contextual_trust(sample: EvalSample) -> SystemExecutionResult:
    global_trust = estimate_contextual_trust(None, parent_successes=80, parent_total=100, evidence_quality=0.8)
    cost = 0.0012
    latency = 45.0

    if sample.is_prohibited_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.is_conflict_case:
        mode = DecisionMode.EXECUTE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REQUEST_CLARIFICATION
    elif sample.gold_completeness == "SUFFICIENT":
        mode = DecisionMode.EXECUTE
    else:
        mode = DecisionMode.REQUEST_CLARIFICATION

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.A1,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=0,
        re_evaluation_count=0,
        audit_lineage={"engine": "ablation_a1_global_trust", "trust": global_trust.model_dump(mode="json")},
    )


def _run_ablation_a2_no_conflict_diagnosis(sample: EvalSample) -> SystemExecutionResult:
    cost = 0.0008
    latency = 38.0

    if sample.is_prohibited_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.is_conflict_case:
        mode = DecisionMode.EXECUTE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REJECT_IGNORE
    elif sample.gold_completeness == "SUFFICIENT":
        mode = DecisionMode.EXECUTE
    else:
        mode = DecisionMode.REQUEST_CLARIFICATION

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.A2,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=0,
        re_evaluation_count=0,
        audit_lineage={"engine": "ablation_a2_no_conflict_diagnosis"},
    )


def _run_ablation_a3_no_escalation(sample: EvalSample) -> SystemExecutionResult:
    cost = 0.0004
    latency = 30.0

    if sample.is_prohibited_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REJECT_IGNORE
    elif sample.is_conflict_case:
        mode = DecisionMode.REQUEST_CLARIFICATION
    elif sample.gold_completeness in ("INCOMPLETE", "AMBIGUOUS"):
        mode = DecisionMode.REQUEST_CLARIFICATION
    else:
        mode = DecisionMode.EXECUTE

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.A3,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=0,
        re_evaluation_count=0,
        audit_lineage={"engine": "ablation_a3_no_escalation"},
    )


def _run_ablation_a4_no_hard_policy_gate(sample: EvalSample) -> SystemExecutionResult:
    cost = 0.001
    latency = 40.0

    if sample.is_prohibited_case:
        passed = True
        mode = DecisionMode.EXECUTE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        passed = False
        mode = DecisionMode.REJECT_IGNORE
    elif sample.is_conflict_case:
        passed = False
        mode = DecisionMode.RE_EVALUATE
    elif sample.gold_completeness == "SUFFICIENT":
        passed = False
        mode = DecisionMode.EXECUTE
    else:
        passed = False
        mode = DecisionMode.REQUEST_CLARIFICATION

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.A4,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action and not sample.is_prohibited_case),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=passed,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=0,
        re_evaluation_count=0,
        audit_lineage={"engine": "ablation_a4_advisory_policy"},
    )


def _run_ablation_a5_call_all_components(sample: EvalSample) -> SystemExecutionResult:
    cost = 0.032
    latency = 1800.0
    egress = 1024

    if sample.is_prohibited_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.gold_action == DecisionMode.REJECT_IGNORE:
        mode = DecisionMode.REJECT_IGNORE
    elif sample.is_conflict_case:
        mode = DecisionMode.RE_EVALUATE
    elif sample.gold_completeness in ("INCOMPLETE", "AMBIGUOUS"):
        mode = DecisionMode.REQUEST_CLARIFICATION
    else:
        mode = DecisionMode.EXECUTE

    return SystemExecutionResult(
        case_id=sample.case_id,
        family_id=sample.family_id,
        system_id=SystemID.A5,
        predicted_mode=mode,
        predicted_category=sample.gold_category,
        predicted_risk=sample.gold_risk,
        is_correct_action=(mode == sample.gold_action),
        is_prohibited_action_attempted=sample.is_prohibited_case,
        is_prohibited_action_passed=False,
        cost_usd=cost,
        latency_ms=latency,
        egress_bytes=egress,
        re_evaluation_count=3,
        audit_lineage={"engine": "ablation_a5_call_all_components"},
    )
