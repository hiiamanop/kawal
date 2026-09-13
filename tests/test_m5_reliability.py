from __future__ import annotations

import pytest

from contracts.models import (
    Category,
    TicketCreateRequest,
    TicketPriority,
    TicketVisibility,
)
from services.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from services.reliability.client import (
    MaxRetriesExceededError,
    ReliableTicketClient,
)
from services.reliability.dlq import DeadLetterQueue
from services.simulator.store import (
    FaultProfile,
    IdempotencyConflictError,
    TicketSimulator,
)


def _make_req(case_id: str = "case-rel-1", payload_hash: str = "a" * 64) -> TicketCreateRequest:
    return TicketCreateRequest(
        tenant_id="tenant-bdg",
        case_id=case_id,
        case_revision=1,
        category=Category.ROAD,
        priority=TicketPriority.NORMAL,
        visibility=TicketVisibility.NORMAL,
        title="Jalan Berlubang Cihampelas",
        description="Lubang aspal berdiameter 40cm.",
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-BINA-MARGA-01",
        source_decision_id="dec-rel-1",
        payload_hash=payload_hash,
    )


def test_circuit_breaker_transitions() -> None:
    current_time = 100.0

    def mock_time() -> float:
        return current_time

    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        time_fn=mock_time,
    )

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True

    # 4 failures: still CLOSED
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED

    # 5th failure: trips to OPEN
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False

    with pytest.raises(CircuitBreakerOpenError):
        breaker.execute(lambda: "ok")

    # Time advances 20s (less than 30s recovery timeout): still OPEN
    current_time += 20.0
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False

    # Time advances another 15s (total 35s >= 30s): transitions to HALF_OPEN
    current_time += 15.0
    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.allow_request() is True

    # Successful probe resets to CLOSED
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_half_open_failure_reopens() -> None:
    current_time = 100.0

    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout_seconds=10.0,
        time_fn=lambda: current_time,
    )

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Advance time past recovery timeout
    current_time += 15.0
    assert breaker.state == CircuitState.HALF_OPEN

    # Probe fails -> reopens immediately
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


def test_dead_letter_queue_quarantines_poison_entries() -> None:
    dlq = DeadLetterQueue()
    assert dlq.count == 0

    entry = dlq.enqueue(
        source_ref="tenant:case-x:ticket:create:v1",
        case_id="case-x",
        tenant_id="tenant-bdg",
        reason="MAX_RETRIES_EXCEEDED",
        error_type="TimeoutError",
        error_message="Gateway timeout",
        payload={"title": "Test Poison"},
        attempt_count=3,
    )
    assert dlq.count == 1
    assert entry.case_id == "case-x"
    assert dlq.get(entry.entry_id) is not None

    entries = dlq.list_entries()
    assert len(entries) == 1
    assert entries[0].entry_id == entry.entry_id

    dlq.clear()
    assert dlq.count == 0


def test_ten_retries_on_commit_then_timeout_boundary_produces_exactly_one_ticket() -> None:
    """M5 ACCEPTANCE TEST:

    10 retries on the commit-then-timeout crash boundary must produce EXACTLY ONE
    ticket in the simulator through stable idempotency and operation reconciliation.
    """
    simulator = TicketSimulator()
    # Configure fault: commit-then-timeout (ticket is committed, but request times out)
    simulator.set_fault_profile(FaultProfile(commit_then_timeout=True))

    breaker = CircuitBreaker(failure_threshold=20)
    dlq = DeadLetterQueue()
    client = ReliableTicketClient(
        simulator=simulator,
        circuit_breaker=breaker,
        dlq=dlq,
        max_retries=3,
    )

    req = _make_req(case_id="case-crash-10")
    key = "tenant-bdg:case-crash-10:ticket:create:v1"

    receipts = []
    # Simulate 10 retry cycles / worker retries on the crash boundary
    for i in range(10):
        receipt = client.create_ticket(req, key)
        receipts.append(receipt)

    # Invariant 1: Exactly one ticket was created in the simulator
    assert simulator.ticket_count == 1

    # Invariant 2: Exactly one operation was recorded in the simulator
    assert simulator.operation_count == 1

    # Invariant 3: All 10 attempts returned the exact same ticket ID and external ID
    first_ticket_id = receipts[0].ticket_id
    assert all(r.ticket_id == first_ticket_id for r in receipts)
    assert all(r.idempotency_key == key for r in receipts)

    # Invariant 4: No events quarantined to DLQ because reconciliation succeeded
    assert dlq.count == 0


def test_retry_with_mutated_payload_fails_closed_with_conflict() -> None:
    simulator = TicketSimulator()
    client = ReliableTicketClient(simulator=simulator)

    req = _make_req(case_id="case-conflict-1", payload_hash="a" * 64)
    key = "tenant-bdg:case-conflict-1:ticket:create:v1"

    receipt = client.create_ticket(req, key)
    assert simulator.ticket_count == 1

    # Second attempt with mutated payload
    mutated = _make_req(case_id="case-conflict-1", payload_hash="b" * 64)
    with pytest.raises(IdempotencyConflictError):
        client.create_ticket(mutated, key)

    # Simulator still has exactly 1 ticket
    assert simulator.ticket_count == 1


def test_client_exhausted_retries_quarantines_to_dlq() -> None:
    simulator = TicketSimulator()
    # Fail before commit: never writes ticket
    simulator.set_fault_profile(FaultProfile(fail_before_commit=True))

    dlq = DeadLetterQueue()
    client = ReliableTicketClient(
        simulator=simulator,
        circuit_breaker=CircuitBreaker(failure_threshold=10),
        dlq=dlq,
        max_retries=3,
    )

    req = _make_req(case_id="case-unrecoverable")
    key = "tenant-bdg:case-unrecoverable:ticket:create:v1"

    with pytest.raises(MaxRetriesExceededError):
        client.create_ticket(req, key)

    assert simulator.ticket_count == 0
    assert dlq.count == 1
    entry = dlq.list_entries()[0]
    assert entry.reason == "MAX_RETRIES_EXCEEDED"
    assert entry.case_id == "case-unrecoverable"
    assert entry.attempt_count == 3
