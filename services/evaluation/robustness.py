from __future__ import annotations

from contracts.models import (
    Category,
    DecisionMode,
    RiskLevel,
    TicketCreateRequest,
    TicketPriority,
    TicketVisibility,
)
from services.core.decision import (
    BudgetState,
    ConflictInput,
    ConflictSeverity,
    ConflictType,
    DecisionInput,
    EgressRequest,
    decide,
    diagnose_conflict,
    estimate_contextual_trust,
    evaluate_egress,
)
from services.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from services.reliability.client import MaxRetriesExceededError, ReliableTicketClient
from services.reliability.dlq import DeadLetterQueue
from services.simulator.store import FaultProfile, TicketSimulator


def run_robustness_invariant_checks() -> dict[str, bool]:
    """Execute M6 robustness suite to verify key non-functional invariants."""
    results: dict[str, bool] = {}

    # Invariant 1: Zero duplicate tickets under commit-then-timeout crash boundary
    results["zero_duplicate_tickets"] = _verify_zero_duplicate_tickets()

    # Invariant 2: Deterministic policy gate fail-closed
    results["hard_policy_fail_closed"] = _verify_hard_policy_fail_closed()

    # Invariant 3: Audit lineage and deterministic trace reproducibility
    results["audit_lineage_reproducibility"] = _verify_audit_lineage_reproducibility()

    # Invariant 4: Circuit breaker containment on consecutive failures
    results["circuit_breaker_containment"] = _verify_circuit_breaker_containment()

    # Invariant 5: Unrecoverable failure quarantine to DLQ
    results["dlq_quarantine_integrity"] = _verify_dlq_quarantine_integrity()

    return results


def _verify_zero_duplicate_tickets() -> bool:
    simulator = TicketSimulator()
    simulator.set_fault_profile(FaultProfile(commit_then_timeout=True))
    client = ReliableTicketClient(simulator=simulator)

    req = TicketCreateRequest(
        tenant_id="tenant-inv",
        case_id="case-inv-1",
        case_revision=1,
        category=Category.ROAD,
        priority=TicketPriority.NORMAL,
        visibility=TicketVisibility.NORMAL,
        title="Test Invariant 1",
        description="Pengecekan nol tiket ganda.",
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-BINA-MARGA-01",
        source_decision_id="dec-inv-1",
        payload_hash="1" * 64,
    )
    key = "tenant-inv:case-inv-1:ticket:create:v1"

    receipts = [client.create_ticket(req, key) for _ in range(10)]
    return (
        simulator.ticket_count == 1
        and len({r.ticket_id for r in receipts}) == 1
        and len({r.idempotency_key for r in receipts}) == 1
    )


def _verify_hard_policy_fail_closed() -> bool:
    egress_denied_pii = evaluate_egress(
        EgressRequest(provider="openai", model_id="gpt-4o", data_class="public", has_pii=True)
    )
    egress_denied_provider = evaluate_egress(
        EgressRequest(provider="untrusted-vendor", model_id="bad", data_class="public")
    )
    egress_denied_image = evaluate_egress(
        EgressRequest(
            provider="openai",
            model_id="gpt-4o-vision",
            data_class="public",
            attachment_mime="image/jpeg",
            category_sensitive=True,
        )
    )
    return (
        not egress_denied_pii.allowed
        and not egress_denied_provider.allowed
        and not egress_denied_image.allowed
    )


def _verify_audit_lineage_reproducibility() -> bool:
    inp = DecisionInput(
        case_id="case-repro",
        revision=1,
        required_fields_complete=True,
        evidence_covered=True,
        authority_valid=True,
        policy_allowed=True,
        is_complaint=True,
        budget=BudgetState(remaining_usd=1.0, remaining_latency_ms=5000.0, remaining_egress_bytes=1000000),
        clarification_available=True,
    )
    plan1 = decide(inp)
    plan2 = decide(inp)
    return plan1 == plan2 and plan1.mode == DecisionMode.EXECUTE


def _verify_circuit_breaker_containment() -> bool:
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=30.0)
    for _ in range(5):
        breaker.record_failure()

    if breaker.allow_request():
        return False

    try:
        breaker.execute(lambda: "should-not-run")
        return False
    except CircuitBreakerOpenError:
        return True


def _verify_dlq_quarantine_integrity() -> bool:
    simulator = TicketSimulator()
    simulator.set_fault_profile(FaultProfile(fail_before_commit=True))
    dlq = DeadLetterQueue()
    client = ReliableTicketClient(simulator=simulator, dlq=dlq, max_retries=2)

    req = TicketCreateRequest(
        tenant_id="tenant-dlq",
        case_id="case-dlq-1",
        case_revision=1,
        category=Category.WASTE,
        priority=TicketPriority.NORMAL,
        visibility=TicketVisibility.NORMAL,
        title="Test DLQ",
        description="Test DLQ quarantine",
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-LH-01",
        source_decision_id="dec-dlq-1",
        payload_hash="2" * 64,
    )
    try:
        client.create_ticket(req, "tenant-dlq:case-dlq-1:ticket:create:v1")
        return False
    except MaxRetriesExceededError:
        return dlq.count == 1 and dlq.list_entries()[0].case_id == "case-dlq-1"
