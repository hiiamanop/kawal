import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from uuid import uuid4

from contracts.models import RawMessage
from infra.db import TransactionRunner
from services.intake.assembly import ASSEMBLY_LEASE, ConversationAssembler, IntakeConflictError
from services.intake.service import AssemblyWorker, IntakeService


DATABASE_URL = os.getenv("KAWAL_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set KAWAL_TEST_DATABASE_URL to run PostgreSQL integration tests",
)
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
            cursor.execute("DROP POLICY IF EXISTS storage_attachments_service_role_only ON storage.objects")
            for migration in MIGRATIONS:
                cursor.execute(migration.read_text(encoding="utf-8"))
    return TransactionRunner(DATABASE_URL)


def message(message_id: str, received_at: datetime, conversation_id: str = "conv-1") -> RawMessage:
    return RawMessage(
        message_id=f"research:{message_id}",
        tenant_id="research",
        conversation_id=conversation_id,
        source_message_id=message_id,
        text=message_id,
        received_at=received_at,
    )


def test_worker_crash_is_reclaimed_and_batch_is_incremental(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    intake = IntakeService(runner, ConversationAssembler())
    worker = AssemblyWorker(runner, ConversationAssembler())
    assert intake.accept(message("first", started_at), case_key="first")

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE assembly_timers
                    SET lease_token = gen_random_uuid(), lease_expires_at = %s
                    WHERE conversation_id = 'conv-1'
                    """,
                    (started_at + ASSEMBLY_LEASE, ),
                )

    assert worker.process_due(started_at + ASSEMBLY_LEASE - timedelta(microseconds=1)) is None
    assert worker.process_due(started_at + ASSEMBLY_LEASE) == "conv-1"
    assert intake.accept(message("second", started_at + timedelta(seconds=40)), case_key="second")
    assert worker.process_due(started_at + timedelta(seconds=45)) == "conv-1"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM message_case_links")
            assert cursor.fetchone()[0] == 2
            cursor.execute("SELECT count(*) FROM cases")
            assert cursor.fetchone()[0] == 2


def test_two_workers_claim_different_conversations(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    intake = IntakeService(runner, ConversationAssembler())
    worker_a = AssemblyWorker(runner, ConversationAssembler())
    worker_b = AssemblyWorker(runner, ConversationAssembler())
    assert intake.accept(message("a", started_at, "conv-a"), case_key="a")
    assert intake.accept(message("b", started_at, "conv-b"), case_key="b")

    results = {
        worker_a.process_due(started_at + timedelta(seconds=5)),
        worker_b.process_due(started_at + timedelta(seconds=5)),
    }
    assert results == {"conv-a", "conv-b"}

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM message_case_links")
            assert cursor.fetchone()[0] == 2


def test_duplicate_delivery_after_assembly_is_safe(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    intake = IntakeService(runner, ConversationAssembler())
    worker = AssemblyWorker(runner, ConversationAssembler())
    assert intake.accept(message("same", started_at), case_key="stable")
    assert worker.process_due(started_at + timedelta(seconds=5)) == "conv-1"
    assert not intake.accept(message("same", started_at), case_key="stable")
    assert worker.process_due(started_at + timedelta(seconds=6)) is None

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM raw_messages")
            assert cursor.fetchone()[0] == 1
            cursor.execute("SELECT count(*) FROM message_case_links")
            assert cursor.fetchone()[0] == 1


def test_two_workers_safety_same_conversation_and_stale_worker_rejected(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker_b = AssemblyWorker(runner, assembler)

    assert intake.accept(message("first", started_at), case_key="case-1")

    claim_now = started_at + timedelta(seconds=5)
    stale_token = uuid4()
    with psycopg.connect(DATABASE_URL) as conn_a:
        with conn_a.transaction():
            with conn_a.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE assembly_timers
                    SET lease_token = %s, lease_expires_at = %s
                    WHERE conversation_id = 'conv-1'
                    """,
                    (stale_token, claim_now + ASSEMBLY_LEASE),
                )

    assert worker_b.process_due(claim_now + timedelta(seconds=10)) is None

    reclaim_now = claim_now + ASSEMBLY_LEASE
    assert worker_b.process_due(reclaim_now) == "conv-1"

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE assembly_timers
                SET fired_at = %s, lease_token = NULL, lease_expires_at = NULL
                WHERE conversation_id = 'conv-1' AND lease_token = %s AND generation = 1
                """,
                (reclaim_now, stale_token),
            )
            assert cursor.rowcount == 0


def test_quote_precedence_in_same_connector_account_chat(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)

    assert intake.accept(
        message("base-msg", started_at, "conv-1"),
        case_key="explicit-key-1",
        connector_id="replay",
        account_id="research",
    )
    assert worker.process_due(started_at + timedelta(seconds=5)) == "conv-1"

    reply_time = started_at + timedelta(seconds=10)
    assert intake.accept(
        RawMessage(
            message_id="research:cross-chat-reply",
            tenant_id="research",
            conversation_id="conv-2",
            source_message_id="cross-chat-reply",
            text="replying across chat boundary",
            received_at=reply_time,
        ),
        quoted_source_message_id="base-msg",
        connector_id="replay",
        account_id="research",
    )
    assert worker.process_due(reply_time + timedelta(seconds=5)) == "conv-2"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT base_link.case_id = reply_link.case_id
                FROM message_case_links AS base_link
                JOIN raw_messages AS base_m ON base_link.message_id = base_m.message_id AND base_m.source_message_id = 'base-msg'
                JOIN raw_messages AS reply_m ON reply_m.source_message_id = 'cross-chat-reply'
                JOIN message_case_links AS reply_link ON reply_link.message_id = reply_m.message_id
                """
            )
            assert cursor.fetchone()[0] is False


def test_three_phase_claim_commits_before_assembly_and_finalizes_conditionally(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)

    assert intake.accept(message("m-1", started_at), case_key="k-1")

    claim_time = started_at + timedelta(seconds=5)
    claim = worker.claim(claim_time)
    assert claim is not None
    assert claim.conversation_id == "conv-1"
    assert claim.generation == 2

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT lease_token, generation, fired_at
                FROM assembly_timers
                WHERE conversation_id = 'conv-1'
                """
            )
            row = cursor.fetchone()
            assert row[0] == claim.lease_token
            assert row[1] == claim.generation
            assert row[2] is None
            cursor.execute("SELECT count(*) FROM message_case_links")
            assert cursor.fetchone()[0] == 0

    last_received_at, last_message_id = worker.assemble(claim, claim_time)
    assert last_message_id == "research:m-1"
    assert last_received_at == started_at

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM message_case_links")
            assert cursor.fetchone()[0] == 1
            cursor.execute("SELECT fired_at, lease_token FROM assembly_timers WHERE conversation_id = 'conv-1'")
            row = cursor.fetchone()
            assert row[0] is None
            assert row[1] == claim.lease_token

    assert worker.finalize(claim, claim_time, last_received_at, last_message_id) is True

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT fired_at, lease_token, lease_expires_at FROM assembly_timers WHERE conversation_id = 'conv-1'")
            row = cursor.fetchone()
            assert row[0] == claim_time
            assert row[1] is None
            assert row[2] is None


def test_new_message_during_active_lease_bumps_generation_and_leaves_timer_pending(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)

    assert intake.accept(message("first", started_at), case_key="case-1")

    claim_time = started_at + timedelta(seconds=5)
    claim = worker.claim(claim_time)
    assert claim is not None
    assert claim.generation == 2

    assert intake.accept(message("second", started_at + timedelta(seconds=6)))

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT generation, fired_at, lease_token
                FROM assembly_timers
                WHERE conversation_id = 'conv-1'
                """
            )
            row = cursor.fetchone()
            assert row[0] == 3
            assert row[1] is None
            assert row[2] == claim.lease_token

    last_received_at, last_message_id = worker.assemble(claim, claim_time)

    with pytest.raises(IntakeConflictError):
        runner.run(
            lambda conn: assembler.finalize(
                conn, claim, claim_time, last_received_at, last_message_id
            )
        )

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT fired_at FROM assembly_timers WHERE conversation_id = 'conv-1'")
            assert cursor.fetchone()[0] is None

    reclaim_time = claim_time + ASSEMBLY_LEASE
    assert worker.process_due(reclaim_time) == "conv-1"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM message_case_links")
            assert cursor.fetchone()[0] == 2
            cursor.execute("SELECT count(DISTINCT case_id) FROM message_case_links")
            assert cursor.fetchone()[0] == 1


def test_delayed_and_out_of_order_messages_not_skipped_forever(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)

    assert intake.accept(message("msg-10", started_at + timedelta(seconds=10)), case_key="base")
    assert worker.process_due(started_at + timedelta(seconds=15)) == "conv-1"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT assembled_through FROM assembly_timers WHERE conversation_id = 'conv-1'")
            assert cursor.fetchone()[0] == started_at + timedelta(seconds=10)

    assert intake.accept(message("delayed-05", started_at + timedelta(seconds=5)))
    assert worker.process_due(started_at + timedelta(seconds=20)) == "conv-1"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM message_case_links")
            assert cursor.fetchone()[0] == 2
            cursor.execute("SELECT count(DISTINCT case_id) FROM message_case_links")
            assert cursor.fetchone()[0] == 1


def test_case_uuid_includes_connector_and_account(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)

    assert intake.accept(
        RawMessage(
            message_id="wa:m-1",
            tenant_id="research",
            conversation_id="conv-dup",
            source_message_id="src-1",
            text="hello",
            received_at=started_at,
        ),
        connector_id="whatsapp",
        account_id="acc-wa",
    )
    assert intake.accept(
        RawMessage(
            message_id="tg:m-1",
            tenant_id="research",
            conversation_id="conv-dup",
            source_message_id="src-1",
            text="hello",
            received_at=started_at,
        ),
        connector_id="telegram",
        account_id="acc-tg",
    )

    assert worker.process_due(started_at + timedelta(seconds=5)) == "conv-dup"
    assert worker.process_due(started_at + timedelta(seconds=5)) == "conv-dup"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT case_id FROM message_case_links")
            rows = cursor.fetchall()
            assert len(rows) == 2
            assert rows[0][0] != rows[1][0]
