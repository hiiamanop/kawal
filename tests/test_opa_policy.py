from __future__ import annotations

import hashlib
import uuid

from contracts.models import Category, PolicyInput, TicketCreateRequest, TicketPriority, TicketVisibility
from services.core.opa import OpaPolicyClient


def _policy_input() -> PolicyInput:
    return PolicyInput(
        tenant_id="tenant-opa",
        case_id="case-opa-01",
        revision=1,
        category=Category.ROAD,
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-BINA-MARGA-01",
        missing_fields=(),
        has_mandatory_evidence=True,
        idempotency_key="tenant-opa:case-opa-01:ticket:create:v1",
    )


def test_opa_client_fails_closed_when_unreachable() -> None:
    result = OpaPolicyClient(base_url="http://127.0.0.1:1", timeout_seconds=0.01).evaluate_ticket_creation(
        _policy_input()
    )

    assert result.decision == "DENY"
    assert result.rule_ids == ("POL-OPA-FAIL-CLOSED",)
    assert result.reason_codes == ("OPA_UNAVAILABLE",)


def test_opa_client_uses_expected_data_api_path() -> None:
    client = OpaPolicyClient(base_url="http://localhost:8181/")
    assert client._url == "http://localhost:8181/v1/data/kawal/policy/result"


def test_gateway_ticket_denies_when_hard_gate_rejects() -> None:
    from services.core.gateway_worker import ToolGatewayWorker
    from services.intake.openwa import OpenWAConnector
    from services.outbox.broker import InMemoryEventBroker
    from services.outbox.consumer import AtomicInboxConsumer
    from services.reliability.client import ReliableTicketClient
    from services.simulator.store import TicketSimulator
    from tests.test_gateway_worker import FakeConnection, FakeRunner

    class DenyPolicyClient:
        def evaluate_ticket_creation(self, input_: PolicyInput):
            return OpaPolicyClient._deny("OPA_UNAVAILABLE")

    broker = InMemoryEventBroker()
    connection = FakeConnection()
    consumer = AtomicInboxConsumer(
        transaction_runner=FakeRunner(connection),  # type: ignore[arg-type]
        broker=broker,
        consumer_name="test-opa-deny",
    )
    simulator = TicketSimulator()
    worker = ToolGatewayWorker(
        consumer=consumer,
        ticket_client=ReliableTicketClient(simulator=simulator),
        messaging_connector=OpenWAConnector(),
        policy_client=DenyPolicyClient(),  # type: ignore[arg-type]
    )

    request = TicketCreateRequest(
        tenant_id="tenant-opa",
        case_id="case-opa-01",
        case_revision=1,
        category=Category.ROAD,
        priority=TicketPriority.NORMAL,
        visibility=TicketVisibility.NORMAL,
        title="Jalan rusak",
        description="Lubang jalan",
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-BINA-MARGA-01",
        source_decision_id=str(uuid.uuid4()),
        payload_hash=hashlib.sha256(b"jalan").hexdigest(),
    )
    key = "tenant-opa:case-opa-01:ticket:create:v1"
    broker.publish(
        topic="commands.ticket.v1",
        key="tenant-opa:case-opa-01",
        payload={"request": request.model_dump(mode="json"), "idempotency_key": key},
        event_id=str(uuid.uuid4()),
    )

    assert worker.consume_ticket_once() == 1
    assert simulator.ticket_count == 0
    executed_queries = " ".join(query for query, _ in connection.queries)
    assert "'BLOCKED'" in executed_queries
    assert "ticket.policy_denied.v1" in str(connection.queries)
