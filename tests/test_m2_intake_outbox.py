import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid5

import pytest

psycopg = pytest.importorskip("psycopg")

from contracts.models import RawMessage
from infra.db import TransactionRunner
from services.core.persistence import M1Store
from services.intake.assembly import ConversationAssembler
from services.intake.replay import (
    EVENT_NAMESPACE,
    build_replay_case,
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
OUTBOX_NAMESPACE = UUID("00000000-0000-0000-0000-000000000000")


@pytest.fixture
def runner() -> TransactionRunner:
    assert DATABASE_URL is not None
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
            cursor.execute("DROP POLICY IF EXISTS storage_attachments_service_role_only ON storage.objects")
            for migration in MIGRATIONS:
                cursor.execute(migration.read_text(encoding="utf-8"))
    return TransactionRunner(DATABASE_URL)


def test_atomic_intake_outbox_emission_on_ingest(runner: TransactionRunner) -> None:
    assembler = ConversationAssembler()
    now = datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc)
    msg = RawMessage(
        message_id="research:wa-outbox-test-1",
        tenant_id="research",
        conversation_id="conv-outbox-1",
        source_message_id="wa-outbox-test-1",
        text="Halo mau lapor jalan rusak",
        received_at=now,
    )

    accepted = runner.run(
        lambda conn: assembler.ingest(
            conn,
            msg,
            quoted_source_message_id=None,
            case_key="road-1",
            connector_id="openwa",
            account_id="research-acc",
        )
    )
    assert accepted is True

    expected_event_id = uuid5(
        OUTBOX_NAMESPACE,
        "research:openwa:research-acc:wa-outbox-test-1:message.received.v1",
    )

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event_id, tenant_id, aggregate_type, aggregate_id, revision,
                       event_type, partition_key, payload, published_at
                FROM outbox
                WHERE event_id = %s
                """,
                (expected_event_id,),
            )
            row = cursor.fetchone()
            assert row is not None
            (
                event_id,
                tenant_id,
                aggregate_type,
                aggregate_id,
                revision,
                event_type,
                partition_key,
                payload,
                published_at,
            ) = row
            assert event_id == expected_event_id
            assert tenant_id == "research"
            assert aggregate_type == "raw_message"
            assert aggregate_id == "research:wa-outbox-test-1"
            assert revision == 1
            assert event_type == "message.received.v1"
            assert partition_key == "conv-outbox-1"
            assert payload["source_message_id"] == "wa-outbox-test-1"
            assert payload["case_key"] == "road-1"
            assert payload["connector_id"] == "openwa"
            assert payload["account_id"] == "research-acc"
            assert published_at is None


def test_duplicate_ingest_does_not_duplicate_outbox_or_fail(runner: TransactionRunner) -> None:
    assembler = ConversationAssembler()
    now = datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc)
    msg = RawMessage(
        message_id="research:wa-outbox-test-2",
        tenant_id="research",
        conversation_id="conv-outbox-2",
        source_message_id="wa-outbox-test-2",
        text="Laporan kedua",
        received_at=now,
    )

    accepted1 = runner.run(
        lambda conn: assembler.ingest(
            conn,
            msg,
            quoted_source_message_id=None,
            case_key=None,
            connector_id="openwa",
            account_id="research-acc",
        )
    )
    assert accepted1 is True

    accepted2 = runner.run(
        lambda conn: assembler.ingest(
            conn,
            msg,
            quoted_source_message_id=None,
            case_key=None,
            connector_id="openwa",
            account_id="research-acc",
        )
    )
    assert accepted2 is False

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM outbox WHERE event_type = 'message.received.v1' AND partition_key = 'conv-outbox-2'"
            )
            assert cursor.fetchone()[0] == 1


def test_m1_outbox_relay_does_not_break_with_message_received_in_outbox(
    runner: TransactionRunner,
) -> None:
    assembler = ConversationAssembler()
    now = datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc)
    msg = RawMessage(
        message_id="research:wa-outbox-test-3",
        tenant_id="research",
        conversation_id="conv-outbox-3",
        source_message_id="wa-outbox-test-3",
        text="Pesan intake sebelum m1 relay",
        received_at=now,
    )
    runner.run(lambda conn: assembler.ingest(conn, msg))

    simulator = TicketSimulator()
    relay = OutboxRelay(runner, M1Store(), simulator)
    fixture = load_fixture(FIXTURE_PATH)
    receipt = replay_to_durable_ticket(fixture, runner, M1Store(), relay)

    assert receipt is not None
    assert receipt.status == "SUBMITTED"
    assert simulator.ticket_count == 1
