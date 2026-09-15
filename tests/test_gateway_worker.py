from __future__ import annotations

import hashlib
from typing import Any
import uuid

from contracts.models import (
    Category,
    OutboundMessageCommand,
    OutboundPurpose,
    TicketCreateRequest,
    TicketPriority,
    TicketVisibility,
)
from services.core.escalation import build_anonymized_escalation_command
from services.core.gateway_worker import ToolGatewayWorker
from services.core.model_gateway import EscalationResult, ModelGateway
from services.intake.openwa import OpenWAConnector
from services.outbox.broker import InMemoryEventBroker
from services.outbox.consumer import AtomicInboxConsumer
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


class FakeConnection:
    def __init__(self) -> None:
        self.queries: list[tuple[str, Any]] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class FakeCursor:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def execute(self, query: str, params: Any = None) -> None:
        self.conn.queries.append((query, params))

    def fetchone(self) -> Any:
        return (1,)


class FakeRunner:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    def run(self, fn: Any) -> Any:
        return fn(self.conn)


def _worker(broker: InMemoryEventBroker, conn: FakeConnection, **kwargs: Any) -> ToolGatewayWorker:
    return ToolGatewayWorker(
        consumer=AtomicInboxConsumer("test-gateway", FakeRunner(conn), broker),  # type: ignore[arg-type]
        ticket_client=ReliableTicketClient(simulator=kwargs.pop("simulator", TicketSimulator())),
        messaging_connector=OpenWAConnector(),
        **kwargs,
    )


def test_tool_gateway_worker_ticket_execution() -> None:
    broker = InMemoryEventBroker()
    conn = FakeConnection()
    simulator = TicketSimulator()
    worker = _worker(broker, conn, simulator=simulator)
    request = TicketCreateRequest(
        tenant_id="tenant-gw",
        case_id="case-gw-01",
        case_revision=1,
        category=Category.ROAD,
        priority=TicketPriority.NORMAL,
        visibility=TicketVisibility.NORMAL,
        title="Jalan Rusak",
        description="Lubang jalan di Jl Merdeka",
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-BINA-MARGA-01",
        source_decision_id=str(uuid.uuid4()),
        payload_hash=hashlib.sha256(b"jalan rusak").hexdigest(),
    )
    key = "tenant-gw:case-gw-01:ticket:create:v1"
    broker.publish("commands.ticket.v1", "tenant-gw:case-gw-01", {"request": request.model_dump(mode="json"), "idempotency_key": key}, event_id=str(uuid.uuid4()))

    assert worker.consume_ticket_once() == 1
    assert simulator.ticket_count == 1
    assert "ticket.executed.v1" in str(conn.queries)


def test_tool_gateway_worker_message_execution() -> None:
    broker = InMemoryEventBroker()
    conn = FakeConnection()
    worker = _worker(broker, conn)
    command = OutboundMessageCommand(
        tenant_id="tenant-gw",
        conversation_id="628111@c.us",
        case_id="case-gw-02",
        recipient_phone="628111@c.us",
        text="Mohon lengkapi lokasi",
        idempotency_key="tenant-gw:628111@c.us:clarification:round_1",
        purpose=OutboundPurpose.CLARIFICATION,
        payload_hash=hashlib.sha256(b"clarify").hexdigest(),
    )
    broker.publish("commands.message.v1", "tenant-gw:case-gw-02", command.model_dump(mode="json"), event_id=str(uuid.uuid4()))

    assert worker.consume_message_once() == 1
    assert "message.executed.v1" in str(conn.queries)


class FakeEscalationAdapter:
    provider = "local"
    model_id = "qwen2.5:7b"

    def run(self, prompt: str) -> EscalationResult:
        return EscalationResult(status="SUCCEEDED", provider=self.provider, model_id=self.model_id, content="KLARIFIKASI")


def test_tool_gateway_worker_executes_anonymized_escalation() -> None:
    broker = InMemoryEventBroker()
    conn = FakeConnection()
    worker = _worker(broker, conn, model_gateway=ModelGateway(FakeEscalationAdapter()))
    command = build_anonymized_escalation_command(
        tenant_id="tenant-escalation",
        case_id="case-escalation",
        revision=1,
        task_id="local-clarification-check",
        category="ROAD",
        risk="HIGH",
        completeness="INCOMPLETE",
    )
    broker.publish("commands.escalation.v1", "tenant-escalation:case-escalation", command.model_dump(mode="json"), event_id=str(uuid.uuid4()))

    assert worker.consume_escalation_once() == 1
    assert "model.escalation.executed.v1" in str(conn.queries)
