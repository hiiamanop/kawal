import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from contracts.models import RawMessage
from infra.db import TransactionRunner
from services.intake.assembly import ConversationAssembler
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


def message(message_id: str, received_at: datetime) -> RawMessage:
    return RawMessage(
        message_id=f"research:{message_id}",
        tenant_id="research",
        conversation_id="conv-1",
        source_message_id=message_id,
        text=message_id,
        received_at=received_at,
    )


def test_interleaved_cases_and_quoted_reply_are_durably_associated(runner: TransactionRunner) -> None:
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)

    assert intake.accept(message("a-1", started_at), case_key="a")
    assert intake.accept(message("b-1", started_at + timedelta(seconds=1)), case_key="b")
    assert intake.accept(message("a-2", started_at + timedelta(seconds=2)), case_key="a")
    assert intake.accept(message("b-2", started_at + timedelta(seconds=3)), case_key="b")
    assert intake.accept(message("a-reply", started_at + timedelta(seconds=4)), "a-1")
    assert worker.process_due(started_at + timedelta(seconds=9)) == "conv-1"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(DISTINCT case_id) FROM message_case_links")
            assert cursor.fetchone()[0] == 2
            cursor.execute(
                """
                SELECT target.case_id = source.case_id
                FROM message_case_links AS target
                JOIN raw_messages AS target_message ON target.message_id = target_message.message_id
                JOIN raw_messages AS source_message ON source_message.source_message_id = 'a-1'
                JOIN message_case_links AS source ON source.message_id = source_message.message_id
                WHERE target_message.source_message_id = 'a-reply'
                """
            )
            assert cursor.fetchone()[0] is True


def test_durable_timer_is_processed_after_worker_restart(runner: TransactionRunner) -> None:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    intake = IntakeService(runner, ConversationAssembler())
    assert intake.accept(message("a-1", started_at), case_key="a")

    restarted_worker = AssemblyWorker(TransactionRunner(DATABASE_URL), ConversationAssembler())
    assert restarted_worker.process_due(started_at + timedelta(seconds=5)) == "conv-1"
    assert restarted_worker.process_due(started_at + timedelta(seconds=6)) is None


def test_duplicate_delivery_does_not_reset_timer_or_create_new_case(runner: TransactionRunner) -> None:
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)

    assert intake.accept(message("a-1", started_at), case_key="a")
    assert not intake.accept(message("a-1", started_at), case_key="a")
    worker.process_due(started_at + timedelta(seconds=5))

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM raw_messages")
            assert cursor.fetchone()[0] == 1
            cursor.execute("SELECT count(*) FROM cases")
            assert cursor.fetchone()[0] == 1


def test_continuous_burst_cap_enforces_hard_limit(runner: TransactionRunner) -> None:
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)

    offsets = [0, 3, 7, 11, 15, 18]
    for offset in offsets:
        assert intake.accept(message(f"burst-{offset}", started_at + timedelta(seconds=offset)), case_key="burst")

    assert worker.process_due(started_at + timedelta(seconds=19)) is None
    assert worker.process_due(started_at + timedelta(seconds=20)) == "conv-1"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM message_case_links")
            assert cursor.fetchone()[0] == len(offsets)
            cursor.execute("SELECT count(*) FROM cases")
            assert cursor.fetchone()[0] == 1


def test_candidate_fallback_within_48_hours_and_expiration(runner: TransactionRunner) -> None:
    assembler = ConversationAssembler()
    intake = IntakeService(runner, assembler)
    worker = AssemblyWorker(runner, assembler)
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)

    assert intake.accept(message("c1-1", started_at), case_key="case-one")
    assert worker.process_due(started_at + timedelta(seconds=5)) == "conv-1"

    followup_time = started_at + timedelta(hours=24)
    assert intake.accept(message("c1-followup", followup_time))
    assert worker.process_due(followup_time + timedelta(seconds=5)) == "conv-1"

    late_time = followup_time + timedelta(hours=49)
    assert intake.accept(message("c2-new", late_time))
    assert worker.process_due(late_time + timedelta(seconds=5)) == "conv-1"

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(DISTINCT case_id) FROM message_case_links")
            assert cursor.fetchone()[0] == 2

            cursor.execute(
                """
                SELECT c1.case_id = cf.case_id
                FROM message_case_links AS c1
                JOIN raw_messages AS m1 ON c1.message_id = m1.message_id AND m1.source_message_id = 'c1-1'
                JOIN raw_messages AS mf ON mf.source_message_id = 'c1-followup'
                JOIN message_case_links AS cf ON cf.message_id = mf.message_id
                """
            )
            assert cursor.fetchone()[0] is True

            cursor.execute(
                """
                SELECT c1.case_id != c2.case_id
                FROM message_case_links AS c1
                JOIN raw_messages AS m1 ON c1.message_id = m1.message_id AND m1.source_message_id = 'c1-1'
                JOIN raw_messages AS m2 ON m2.source_message_id = 'c2-new'
                JOIN message_case_links AS c2 ON c2.message_id = m2.message_id
                """
            )
            assert cursor.fetchone()[0] is True
