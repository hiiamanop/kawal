from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any
import uuid

import pytest

from contracts.models import (
    CaseSnapshot,
    Category,
    OutboundMessageCommand,
    OutboundPurpose,
    ProcessingState,
    RawMessage,
    TicketCreateRequest,
    TicketPriority,
    TicketVisibility,
)
from services.core.gateway_worker import ToolGatewayWorker
from services.core.pipeline import CaseProcessingPipeline
from services.core.worker import CaseReadyWorker
from services.intake.openwa import OpenWAConnector
from services.outbox.broker import BrokerMessage, InMemoryEventBroker
from services.outbox.consumer import AtomicInboxConsumer
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


class FakeConnection:
    """Mock psycopg connection for unit tests."""
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


def test_tool_gateway_worker_ticket_execution() -> None:
    broker = InMemoryEventBroker()
    conn = FakeConnection()
    runner = FakeRunner(conn)
    consumer = AtomicInboxConsumer(
        transaction_runner=runner,  # type: ignore
        broker=broker,
        consumer_name="test-gateway-ticket",
    )
    simulator = TicketSimulator()
    ticket_client = ReliableTicketClient(simulator=simulator)
    worker = ToolGatewayWorker(
        consumer=consumer,
        ticket_client=ticket_client,
        messaging_connector=OpenWAConnector(),
    )

    req = TicketCreateRequest(
        tenant_id="tenant-gw",
        case_id="case-gw-01",
        case_revision=1,
        category=Category.ROAD,
        priority=TicketPriority.NORMAL,
        visibility=TicketVisibility.NORMAL,
        title="Jalan Rusak",
        description="Lubang jalan di Jl Merdeka",
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-HIGHWAY-01",
        source_decision_id=str(uuid.uuid4()),
        payload_hash=hashlib.sha256(b"jalan rusak").hexdigest(),
    )
    key = f"{req.tenant_id}:{req.case_id}:ticket:create:v1"

    broker.publish(
        topic="commands.ticket.v1",
        key=f"{req.tenant_id}:{req.case_id}",
        payload={"request": req.model_dump(mode="json"), "idempotency_key": key},
        event_id=str(uuid.uuid4()),
    )

    count = worker.consume_ticket_once()
    assert count == 1
    assert simulator.ticket_count == 1
    queries_str = " ".join(q[0] for q in conn.queries)
    assert "UPDATE cases" in queries_str
    assert "INSERT INTO audit_traces" in queries_str


def test_tool_gateway_worker_message_execution() -> None:
    broker = InMemoryEventBroker()
    conn = FakeConnection()
    runner = FakeRunner(conn)
    consumer = AtomicInboxConsumer(
        transaction_runner=runner,  # type: ignore
        broker=broker,
        consumer_name="test-gateway-msg",
    )
    worker = ToolGatewayWorker(
        consumer=consumer,
        ticket_client=ReliableTicketClient(simulator=TicketSimulator()),
        messaging_connector=OpenWAConnector(),
    )

    cmd = OutboundMessageCommand(
        tenant_id="tenant-gw",
        conversation_id="628111@c.us",
        case_id="case-gw-02",
        recipient_phone="628111@c.us",
        text="Mohon lengkapi lokasi",
        idempotency_key="tenant-gw:628111@c.us:clarification:round_1",
        purpose=OutboundPurpose.CLARIFICATION,
        payload_hash=hashlib.sha256(b"clarify").hexdigest(),
    )

    broker.publish(
        topic="commands.message.v1",
        key=f"{cmd.tenant_id}:{cmd.case_id}",
        payload=cmd.model_dump(mode="json"),
        event_id=str(uuid.uuid4()),
    )

    count = worker.consume_message_once()
    assert count == 1
    queries_str = " ".join(q[0] for q in conn.queries)
    assert "INSERT INTO audit_traces" in queries_str
    assert "message.executed.v1" in str(conn.queries)


def test_deferred_command_end_to_end() -> None:
    """Verifies that pipeline produces TicketCommand and worker persists to outbox."""
    pipeline = CaseProcessingPipeline()
    now = datetime.now(timezone.utc)
    text = "Lapor jalan berlubang parah di Jl. Asia Afrika No. 10, RT 01 RW 02, Kelurahan Braga, Kecamatan Sumur Bandung, Kota Bandung."
    snapshot = CaseSnapshot(
        case_id="case-deferred-01",
        tenant_id="tenant-deferred",
        conversation_id="628999@c.us",
        revision=1,
        state=ProcessingState.READY,
        messages=(
            RawMessage(
                message_id="msg-1",
                tenant_id="tenant-deferred",
                conversation_id="628999@c.us",
                source_message_id="src-1",
                text=text,
                received_at=now,
            ),
        ),
        evidence_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        created_at=now,
    )

    result = pipeline.process_snapshot(snapshot)
    assert result.decision_mode.value == "EXECUTE"
    assert result.ticket_command is not None
    assert result.ticket_command.request.category == Category.ROAD
    assert result.ticket_command.idempotency_key == "tenant-deferred:case-deferred-01:ticket:create:v1"

    # Now verify CaseReadyWorker persists this to outbox table
    conn = FakeConnection()
    broker = InMemoryEventBroker()
    runner = FakeRunner(conn)
    case_worker = CaseReadyWorker(
        consumer=AtomicInboxConsumer(runner, broker, "test-case-worker"),  # type: ignore
        pipeline=pipeline,
    )

    case_worker._persist_result(
        conn,  # type: ignore
        snapshot,
        BrokerMessage(
            event_id=str(uuid.uuid4()),
            topic="cases.ready.v1",
            partition_key="tenant:case",
            payload={"case_id": snapshot.case_id},
        ),
        result,
    )

    outbox_inserts = [q for q in conn.queries if "INSERT INTO outbox" in q[0]]
    assert len(outbox_inserts) == 1
    query, params = outbox_inserts[0]
    assert "ticket.create.requested" in query
    assert params[0] == snapshot.tenant_id
    assert params[1] == snapshot.case_id
