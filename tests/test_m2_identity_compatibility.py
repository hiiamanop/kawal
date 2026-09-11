import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from contracts.models import RawMessage
from infra.db import TransactionRunner
from services.core.persistence import M1Store
from services.intake.assembly import ConversationAssembler
from services.intake.replay import (
    load_fixture,
    replay_to_durable_ticket,
)
from services.outbox.relay import OutboxRelay
from services.simulator.store import TicketSimulator


DATABASE_URL = os.getenv("KAWAL_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set KAWAL_TEST_DATABASE_URL to run PostgreSQL integration tests",
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "three_bubble_road_complaint.json"
MIGRATIONS = tuple(sorted((Path(__file__).parents[1] / "supabase" / "migrations").glob("*.sql")))


@pytest.fixture
def runner() -> TransactionRunner:
    assert DATABASE_URL is not None
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
            cursor.execute(
                "DROP POLICY IF EXISTS storage_attachments_service_role_only ON storage.objects"
            )
            for migration in MIGRATIONS:
                cursor.execute(migration.read_text(encoding="utf-8"))
    return TransactionRunner(DATABASE_URL)


def test_two_connector_account_same_source_id_can_persist(
    runner: TransactionRunner,
) -> None:
    assembler = ConversationAssembler()
    now = datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc)
    shared_source_id = "shared-msg-001"

    msg1 = RawMessage(
        message_id="research:wa:acc1:shared-msg-001",
        tenant_id="research",
        conversation_id="conv-wa-1",
        source_message_id=shared_source_id,
        text="Laporan via WhatsApp account 1",
        received_at=now,
    )
    accepted1 = runner.run(
        lambda conn: assembler.ingest(
            conn,
            msg1,
            connector_id="whatsapp",
            account_id="acc-1",
        )
    )
    assert accepted1 is True

    msg2 = RawMessage(
        message_id="research:tg:acc1:shared-msg-001",
        tenant_id="research",
        conversation_id="conv-tg-1",
        source_message_id=shared_source_id,
        text="Laporan via Telegram account 1",
        received_at=now,
    )
    accepted2 = runner.run(
        lambda conn: assembler.ingest(
            conn,
            msg2,
            connector_id="telegram",
            account_id="acc-1",
        )
    )
    assert accepted2 is True

    msg3 = RawMessage(
        message_id="research:wa:acc2:shared-msg-001",
        tenant_id="research",
        conversation_id="conv-wa-2",
        source_message_id=shared_source_id,
        text="Laporan via WhatsApp account 2",
        received_at=now,
    )
    accepted3 = runner.run(
        lambda conn: assembler.ingest(
            conn,
            msg3,
            connector_id="whatsapp",
            account_id="acc-2",
        )
    )
    assert accepted3 is True

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT connector_id, account_id, source_message_id, message_id
                FROM raw_messages
                WHERE tenant_id = 'research' AND source_message_id = %s
                ORDER BY connector_id, account_id
                """,
                (shared_source_id,),
            )
            rows = cursor.fetchall()
            assert len(rows) == 3
            assert rows[0] == ("telegram", "acc-1", shared_source_id, msg2.message_id)
            assert rows[1] == ("whatsapp", "acc-1", shared_source_id, msg1.message_id)
            assert rows[2] == ("whatsapp", "acc-2", shared_source_id, msg3.message_id)

            cursor.execute(
                """
                SELECT conname FROM pg_constraint
                WHERE conrelid = 'raw_messages'::regclass AND conname = 'uq_raw_messages_source'
                """
            )
            assert cursor.fetchone() is None

            cursor.execute(
                """
                SELECT conname FROM pg_constraint
                WHERE conrelid = 'raw_messages'::regclass AND conname = 'uq_raw_messages_source_identity'
                """
            )
            assert cursor.fetchone() is not None


def test_m1_durable_replay_still_works(runner: TransactionRunner) -> None:
    simulator = TicketSimulator()
    relay = OutboxRelay(runner, M1Store(), simulator)
    fixture = load_fixture(FIXTURE_PATH)

    first = replay_to_durable_ticket(fixture, runner, M1Store(), relay)
    second = replay_to_durable_ticket(fixture, runner, M1Store(), relay)

    assert second.ticket_id == first.ticket_id
    assert simulator.ticket_count == 1
    assert first.status == "SUBMITTED"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            for table, expected in (
                ("raw_messages", 3),
                ("cases", 1),
                ("case_snapshots", 1),
                ("decisions", 1),
                ("ticket_commands", 1),
                ("inbox", 1),
                ("audit_traces", 1),
            ):
                cursor.execute(f"SELECT count(*) FROM {table}")
                assert cursor.fetchone()[0] == expected
            cursor.execute(
                "SELECT count(*) FROM outbox WHERE event_type = 'ticket.create.requested' AND published_at IS NOT NULL"
            )
            assert cursor.fetchone()[0] == 1


def test_intake_outbox_does_not_poison_ticket_relay(runner: TransactionRunner) -> None:
    assembler = ConversationAssembler()
    now = datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc)
    intake_msg = RawMessage(
        message_id="research:wa:intake-poison-test",
        tenant_id="research",
        conversation_id="conv-poison-1",
        source_message_id="intake-poison-test",
        text="Pesan intake sebelum ticket relay dispatch",
        received_at=now,
    )
    accepted = runner.run(
        lambda conn: assembler.ingest(
            conn,
            intake_msg,
            connector_id="whatsapp",
            account_id="acc-poison",
        )
    )
    assert accepted is True

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE outbox
                SET lease_expires_at = NULL,
                    created_at = clock_timestamp() - interval '1 hour'
                WHERE event_type = 'message.received.v1'
                """
            )

    simulator = TicketSimulator()
    relay = OutboxRelay(runner, M1Store(), simulator)

    unclaimed = relay.dispatch_next()
    assert unclaimed is None

    fixture = load_fixture(FIXTURE_PATH)
    receipt = replay_to_durable_ticket(fixture, runner, M1Store(), relay)

    assert receipt is not None
    assert receipt.status == "SUBMITTED"
    assert simulator.ticket_count == 1

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT published_at, attempts
                FROM outbox
                WHERE event_type = 'message.received.v1'
                """
            )
            row = cursor.fetchone()
            assert row is not None
            published_at, attempts = row
            assert published_at is None
            assert attempts == 0

            cursor.execute(
                """
                SELECT published_at, attempts
                FROM outbox
                WHERE event_type = 'ticket.create.requested'
                """
            )
            row = cursor.fetchone()
            assert row is not None
            ticket_published_at, ticket_attempts = row
            assert ticket_published_at is not None
            assert ticket_attempts == 1

    assert relay.dispatch_next() is None
