from __future__ import annotations

import os
from pathlib import Path
import time
from uuid import UUID, uuid4
import psycopg
import pytest

from infra.db import TransactionRunner
from services.core.persistence import M1Store
from services.outbox.broker import (
    BrokerMessage,
    InMemoryEventBroker,
    RedpandaBrokerUnavailableError,
    RedpandaEventBroker,
)
from services.outbox.consumer import AtomicInboxConsumer
from services.outbox.relay import (
    EVENT_TYPE_TO_TOPIC,
    OutboxRelayDaemon,
)

DATABASE_URL = os.getenv("KAWAL_TEST_DATABASE_URL")
MIGRATIONS = tuple(sorted((Path(__file__).parents[1] / "supabase" / "migrations").glob("*.sql")))


def test_in_memory_event_broker_publish_poll_commit() -> None:
    broker = InMemoryEventBroker()

    # Publish 3 messages to topic
    m1 = broker.publish(
        topic="intake.messages.v1",
        key="conv-1",
        payload={"text": "message 1"},
    )
    m2 = broker.publish(
        topic="intake.messages.v1",
        key="conv-1",
        payload={"text": "message 2"},
    )
    m3 = broker.publish(
        topic="intake.messages.v1",
        key="conv-2",
        payload={"text": "message 3"},
    )

    assert broker.message_count("intake.messages.v1") == 3

    # Group 1 polls first 2
    polled = broker.poll("intake.messages.v1", group_id="group-assembly", max_records=2)
    assert len(polled) == 2
    assert polled[0].payload["text"] == "message 1"
    assert polled[1].payload["text"] == "message 2"

    # Commit offset for group 1 up to m2 (offset 1)
    broker.commit_offset("intake.messages.v1", group_id="group-assembly", offset=1)

    # Next poll for group 1 returns only remaining message 3
    polled_next = broker.poll("intake.messages.v1", group_id="group-assembly", max_records=10)
    assert len(polled_next) == 1
    assert polled_next[0].payload["text"] == "message 3"

    # Group 2 has independent offset and sees all 3 messages from start
    polled_g2 = broker.poll("intake.messages.v1", group_id="group-indexer", max_records=10)
    assert len(polled_g2) == 3


def test_redpanda_adapter_fails_explicitly_when_dependency_missing_or_broker_down() -> None:
    broker = RedpandaEventBroker(bootstrap_servers="127.0.0.1:1", request_timeout_ms=100)
    with pytest.raises(RedpandaBrokerUnavailableError):
        broker.ensure_topics(["kawal.unavailable.test"])


@pytest.mark.integration
def test_redpanda_live_publish_consume_smoke() -> None:
    broker = RedpandaEventBroker(bootstrap_servers="127.0.0.1:19092")
    topic = "kawal.redpanda.integration.v1"
    group_id = f"kawal-integration-{uuid4()}"
    try:
        broker.ensure_topics([topic])
        published = broker.publish(topic, "case-live", {"status": "READY"})
        received: list[BrokerMessage] = []
        for _ in range(5):
            received = broker.poll(topic, group_id, max_records=10)
            if any(message.event_id == published.event_id for message in received):
                break
            time.sleep(0.5)
        assert any(message.event_id == published.event_id for message in received)
        for message in received:
            broker.commit_offset(topic, group_id, message.offset)
        assert broker.poll(topic, group_id, max_records=10) == []
    except RedpandaBrokerUnavailableError:
        pytest.skip("Redpanda live broker is not running at 127.0.0.1:19092")
    finally:
        broker.close()


def test_topic_routing_map_matches_prd_section_32() -> None:
    assert EVENT_TYPE_TO_TOPIC["message.received.v1"] == "intake.messages.v1"
    assert EVENT_TYPE_TO_TOPIC["case.ready.v1"] == "cases.ready.v1"
    assert EVENT_TYPE_TO_TOPIC["ticket.create.requested"] == "commands.ticket.v1"
    assert EVENT_TYPE_TO_TOPIC["whatsapp.send.requested.v1"] == "commands.message.v1"
    assert EVENT_TYPE_TO_TOPIC["ticket.status.updated.v1"] == "tickets.status.v1"
    assert EVENT_TYPE_TO_TOPIC["failure.quarantined.v1"] == "failures.dlq.v1"


class MockCursor:
    def __init__(self, inbox_set: set[tuple[str, str]]) -> None:
        self.inbox_set = inbox_set
        self._last_result: tuple[object, ...] | None = None

    def __enter__(self) -> MockCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, query: str, params: tuple[object, ...]) -> None:
        if "INSERT INTO inbox" in query:
            consumer_name, event_uuid, _ = params
            key = (str(consumer_name), str(event_uuid))
            if key in self.inbox_set:
                self._last_result = None
            else:
                self.inbox_set.add(key)
                self._last_result = (event_uuid,)

    def fetchone(self) -> tuple[object, ...] | None:
        return self._last_result


class MockConnection:
    def __init__(self, inbox_set: set[tuple[str, str]]) -> None:
        self.inbox_set = inbox_set

    def cursor(self) -> MockCursor:
        return MockCursor(self.inbox_set)


class MockTransactionRunner:
    def __init__(self) -> None:
        self.inbox_set: set[tuple[str, str]] = set()

    def run(self, operation: object) -> object:
        conn = MockConnection(self.inbox_set)
        return operation(conn)  # type: ignore[operator]


def test_atomic_inbox_consumer_deduplication_invariant_in_memory() -> None:
    broker = InMemoryEventBroker()
    runner = MockTransactionRunner()
    consumer = AtomicInboxConsumer(
        consumer_name="test-consumer",
        transaction_runner=runner,  # type: ignore[arg-type]
        broker=broker,
    )

    msg = broker.publish(
        topic="cases.ready.v1",
        key="case-100",
        payload={"action": "ANALYZE"},
    )

    handler_calls = 0

    def handler(conn: object, event: BrokerMessage) -> None:
        nonlocal handler_calls
        handler_calls += 1

    # First delivery
    assert consumer.process_message(msg, handler) is True
    assert handler_calls == 1

    # Redelivery (simulated network replay / crash)
    assert consumer.process_message(msg, handler) is False
    assert handler_calls == 1  # Invariant: Handler was NOT called again!


@pytest.mark.skipif(DATABASE_URL is None, reason="set KAWAL_TEST_DATABASE_URL for PostgreSQL test")
def test_outbox_relay_daemon_and_atomic_inbox_consumer() -> None:
    psycopg = pytest.importorskip("psycopg")

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
            for migration in MIGRATIONS:
                cursor.execute(migration.read_text(encoding="utf-8"))
            cursor.execute("INSERT INTO tenants (tenant_id, name) VALUES ('tenant-bdg', 'Bandung') ON CONFLICT DO NOTHING")

            # Insert an unpublished outbox event
            event_id = uuid4()
            cursor.execute(
                """
                INSERT INTO outbox (
                    event_id, tenant_id, aggregate_type, aggregate_id, revision,
                    event_type, partition_key, payload
                ) VALUES (
                    %s, 'tenant-bdg', 'case', 'case-outbox-1', 1,
                    'case.ready.v1', 'tenant-bdg:case-outbox-1', '{"status": "READY"}'::jsonb
                )
                """,
                (event_id,),
            )

    runner = TransactionRunner(DATABASE_URL)
    store = M1Store()
    broker = InMemoryEventBroker()
    daemon = OutboxRelayDaemon(transaction_runner=runner, store=store, broker=broker)

    # Relay next event from outbox to broker
    relayed_msg = daemon.relay_next()
    assert relayed_msg is not None
    assert relayed_msg.topic == "cases.ready.v1"
    assert relayed_msg.partition_key == "tenant-bdg:case-outbox-1"
    assert broker.message_count("cases.ready.v1") == 1

    # Second relay call finds no more unpublished events
    assert daemon.relay_next() is None

    # Far-future unpublished lease from a legacy/corrupted relay can be released,
    # while a published event must remain untouched.
    cursor_connection = psycopg.connect(DATABASE_URL)
    with cursor_connection:
        with cursor_connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO outbox (
                    tenant_id, aggregate_type, aggregate_id, revision,
                    event_type, partition_key, payload, lease_expires_at
                ) VALUES (
                    'tenant-bdg', 'case', 'case-stuck-1', 1,
                    'case.ready.v1', 'tenant-bdg:case-stuck-1',
                    '{\"status\": \"READY\"}'::jsonb, '9999-12-31T23:59:59Z'
                ) RETURNING event_id
                """
            )
            stuck_id = cursor.fetchone()[0]
    cursor_connection.close()

    released = runner.run(lambda conn: store.reconcile_stuck_outbox_leases(conn))
    assert stuck_id in released

    # Now test AtomicInboxConsumer
    consumer = AtomicInboxConsumer(
        consumer_name="orchestrator-consumer",
        transaction_runner=runner,
        broker=broker,
    )

    handler_calls = 0

    def mock_handler(conn: psycopg.Connection, msg: BrokerMessage) -> None:
        nonlocal handler_calls
        handler_calls += 1

    # First consumption: newly processed
    processed_count = consumer.consume_batch("cases.ready.v1", mock_handler, max_records=10)
    assert processed_count == 1
    assert handler_calls == 1

    # Simulate redelivery of the exact same message (crash boundary test)
    duplicate_processed = consumer.process_message(relayed_msg, mock_handler)
    # Invariant: Duplicate event skipped via inbox table deduplication!
    assert duplicate_processed is False
    assert handler_calls == 1  # Handler was NOT called a second time!


@pytest.mark.integration
def test_assembly_to_case_ready_outbox_integration() -> None:
    if DATABASE_URL is None:
        pytest.skip("set KAWAL_TEST_DATABASE_URL to run PostgreSQL integration tests")

    from datetime import datetime, timezone, timedelta
    from contracts.models import RawMessage
    from services.intake.service import IntakeService
    from services.intake.assembly import ConversationAssembler
    from services.intake.worker import IntakeAssemblyWorker

    runner = TransactionRunner(DATABASE_URL)
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)

    tenant_id = f"tenant-test-{uuid4().hex[:6]}"
    conv_id = f"{uuid4().hex[:8]}@c.us"

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO tenants (tenant_id, name) VALUES (%s, %s)", (tenant_id, "Test Tenant"))

    now = datetime.now(timezone.utc)
    msg = RawMessage(
        message_id=f"test-msg-{uuid4().hex[:8]}",
        tenant_id=tenant_id,
        conversation_id=conv_id,
        source_message_id=f"src-{uuid4().hex[:8]}",
        text="Lapor jalan rusak di Jl. Merdeka",
        received_at=now,
    )
    assert intake.accept(msg, connector_id="openwa", account_id="bot") is True

    # Assemble using IntakeAssemblyWorker
    worker = IntakeAssemblyWorker(transaction_runner=runner, assembler=assembler)
    assembled = worker.process_due_timers(now=now + timedelta(seconds=10))
    assert conv_id in assembled

    # Verify that case_snapshots and outbox event case.ready.v1 exist
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT case_id, processing_state FROM cases WHERE tenant_id = %s", (tenant_id,))
            case_row = cur.fetchone()
            assert case_row is not None
            assert case_row[1] == "READY"
            case_id = case_row[0]

            cur.execute("SELECT state, evidence_hash FROM case_snapshots WHERE case_id = %s", (case_id,))
            snap_row = cur.fetchone()
            assert snap_row is not None
            assert snap_row[0] == "READY"

            cur.execute("SELECT event_type FROM outbox WHERE tenant_id = %s AND aggregate_id = %s", (tenant_id, case_id))
            outbox_row = cur.fetchone()
            assert outbox_row is not None
            assert outbox_row[0] == "case.ready.v1"
