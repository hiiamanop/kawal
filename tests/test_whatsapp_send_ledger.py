from __future__ import annotations

import os
from pathlib import Path
import pytest

from contracts.models import (
    DeliveryStatus,
    OutboundMessageCommand,
    OutboundPurpose,
)
from services.intake.send_ledger import (
    IdempotencyConflictError,
    InMemorySendLedger,
    SendLedgerStore,
)

DATABASE_URL = os.getenv("KAWAL_TEST_DATABASE_URL")
MIGRATIONS = tuple(sorted((Path(__file__).parents[1] / "supabase" / "migrations").glob("*.sql")))


def _sample_command(
    key: str = "tenant-bdg:conv-1:clarification:round_1",
    payload_hash: str = "a" * 64,
) -> OutboundMessageCommand:
    return OutboundMessageCommand(
        tenant_id="tenant-bdg",
        conversation_id="conv-1",
        case_id="case-1",
        recipient_phone="6281234567890@c.us",
        text="Mohon informasikan nama jalan lokasi kejadian.",
        quoted_source_message_id="src-msg-1",
        idempotency_key=key,
        purpose=OutboundPurpose.CLARIFICATION,
        payload_hash=payload_hash,
    )


def test_in_memory_send_ledger_idempotency_and_lifecycle() -> None:
    ledger = InMemorySendLedger()
    cmd = _sample_command()

    # 1. Record pending send
    send_id, is_new = ledger.record_pending_send(None, cmd)
    assert is_new is True
    assert send_id is not None

    entry = ledger.lookup_by_idempotency_key(None, cmd.idempotency_key)
    assert entry is not None
    assert entry["status"] == DeliveryStatus.PENDING.value
    assert entry["attempts"] == 0

    # 2. Idempotent replay returns same send_id without creating new entry
    send_id_replay, is_new_replay = ledger.record_pending_send(None, cmd)
    assert is_new_replay is False
    assert send_id_replay == send_id

    # 3. Conflict on mutated payload
    mutated = cmd.model_copy(update={"payload_hash": "b" * 64})
    with pytest.raises(IdempotencyConflictError):
        ledger.record_pending_send(None, mutated)

    # 4. Mark unknown outcome on network timeout (no blind resend)
    ledger.mark_unknown_outcome(None, send_id, "Gateway timeout after 5000ms")
    entry_unknown = ledger.lookup_by_idempotency_key(None, cmd.idempotency_key)
    assert entry_unknown["status"] == DeliveryStatus.DELIVERY_UNKNOWN.value
    assert entry_unknown["attempts"] == 1
    assert "timeout" in entry_unknown["last_error"].lower()

    # 5. Mark sent after delivery confirmed
    ledger.mark_sent(None, send_id, "false_6281234567890@c.us_ABC123")
    entry_sent = ledger.lookup_by_idempotency_key(None, cmd.idempotency_key)
    assert entry_sent["status"] == DeliveryStatus.SENT.value
    assert entry_sent["source_message_id"] == "false_6281234567890@c.us_ABC123"
    assert entry_sent["attempts"] == 2


@pytest.mark.skipif(DATABASE_URL is None, reason="set KAWAL_TEST_DATABASE_URL for PostgreSQL test")
def test_postgresql_send_ledger_integration() -> None:
    psycopg = pytest.importorskip("psycopg")
    from infra.db import TransactionRunner

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
            for migration in MIGRATIONS:
                cursor.execute(migration.read_text(encoding="utf-8"))
            cursor.execute("INSERT INTO tenants (tenant_id, name) VALUES ('tenant-bdg', 'Bandung') ON CONFLICT DO NOTHING")
            cursor.execute(
                """
                INSERT INTO cases (case_id, tenant_id, conversation_id, processing_state)
                VALUES ('case-1', 'tenant-bdg', 'conv-1', 'WAITING_CLARIFICATION')
                """
            )

    runner = TransactionRunner(DATABASE_URL)
    store = SendLedgerStore()
    cmd = _sample_command()

    # Record pending send with outbox
    send_id, is_new = runner.run(lambda conn: store.record_pending_send(conn, cmd))
    assert is_new is True

    # Replay idempotent
    send_id2, is_new2 = runner.run(lambda conn: store.record_pending_send(conn, cmd))
    assert is_new2 is False
    assert send_id2 == send_id

    # Conflict
    mutated = cmd.model_copy(update={"payload_hash": "c" * 64})
    with pytest.raises(IdempotencyConflictError):
        runner.run(lambda conn: store.record_pending_send(conn, mutated))

    # Mark sent
    runner.run(lambda conn: store.mark_sent(conn, send_id, "wa-msg-id-999"))
    row = runner.run(lambda conn: store.lookup_by_idempotency_key(conn, cmd.idempotency_key))
    assert row["status"] == DeliveryStatus.SENT.value
    assert row["source_message_id"] == "wa-msg-id-999"
