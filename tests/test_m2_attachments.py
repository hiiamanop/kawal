import concurrent.futures
import os
import threading
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from contracts.models import AttachmentMetadata, RawMessage
from infra.db import TransactionRunner
from services.intake.assembly import ConversationAssembler
from services.intake.attachments import AttachmentAssociationError, AttachmentStore
from services.intake.service import AssemblyWorker, IntakeService


DATABASE_URL = os.getenv("KAWAL_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(DATABASE_URL is None, reason="set KAWAL_TEST_DATABASE_URL")
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


def test_late_attachment_is_idempotent_within_association_horizon(runner: TransactionRunner) -> None:
    received_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    message = RawMessage(
        message_id="research:message-1", tenant_id="research", conversation_id="chat-1",
        source_message_id="message-1", text="jalan rusak", received_at=received_at,
    )
    intake = IntakeService(runner, ConversationAssembler())
    intake.accept(message, case_key="road")
    AssemblyWorker(runner, ConversationAssembler()).process_due(received_at + timedelta(seconds=5))
    metadata = AttachmentMetadata(
        object_key="research/message-1/photo.webp", mime_type="image/webp", size_bytes=1024,
        content_hash=sha256(b"photo").hexdigest(), received_at=received_at + timedelta(hours=47),
        retention_class="evidence",
    )
    store = AttachmentStore()

    assert runner.run(lambda connection: store.persist(connection, message_id=message.message_id, metadata=metadata, now=metadata.received_at))
    assert not runner.run(lambda connection: store.persist(connection, message_id=message.message_id, metadata=metadata, now=metadata.received_at))
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM attachments")
            assert cursor.fetchone()[0] == 1


def test_attachment_after_horizon_is_rejected(runner: TransactionRunner) -> None:
    received_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    message = RawMessage(
        message_id="research:message-1", tenant_id="research", conversation_id="chat-1",
        source_message_id="message-1", text="jalan rusak", received_at=received_at,
    )
    intake = IntakeService(runner, ConversationAssembler())
    intake.accept(message, case_key="road")
    AssemblyWorker(runner, ConversationAssembler()).process_due(received_at + timedelta(seconds=5))
    metadata = AttachmentMetadata(
        object_key="research/message-1/late.jpg", mime_type="image/jpeg", size_bytes=1024,
        content_hash=sha256(b"late").hexdigest(), received_at=received_at + timedelta(hours=49),
        retention_class="evidence",
    )

    with pytest.raises(AttachmentAssociationError, match="association window has expired"):
        runner.run(
            lambda connection: AttachmentStore().persist(
                connection, message_id=message.message_id, metadata=metadata, now=metadata.received_at
            )
        )


def test_concurrent_attachment_persistence_is_idempotent(runner: TransactionRunner) -> None:
    received_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    message = RawMessage(
        message_id="research:message-1", tenant_id="research", conversation_id="chat-1",
        source_message_id="message-1", text="jalan rusak", received_at=received_at,
    )
    intake = IntakeService(runner, ConversationAssembler())
    intake.accept(message, case_key="road")
    AssemblyWorker(runner, ConversationAssembler()).process_due(received_at + timedelta(seconds=5))

    metadata = AttachmentMetadata(
        object_key="research/message-1/concurrent.webp", mime_type="image/webp", size_bytes=2048,
        content_hash=sha256(b"concurrent").hexdigest(), received_at=received_at + timedelta(hours=2),
        retention_class="evidence",
    )
    store = AttachmentStore()

    barrier = threading.Barrier(2)

    def attempt() -> bool:
        barrier.wait()
        return runner.run(
            lambda connection: store.persist(
                connection, message_id=message.message_id, metadata=metadata, now=metadata.received_at
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(attempt)
        f2 = executor.submit(attempt)
        results = [f1.result(), f2.result()]

    assert sorted(results) == [False, True]
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM attachments WHERE object_key = %s", (metadata.object_key,))
            assert cursor.fetchone()[0] == 1


def test_attachment_persistence_rejects_different_metadata(runner: TransactionRunner) -> None:
    received_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    message = RawMessage(
        message_id="research:message-1", tenant_id="research", conversation_id="chat-1",
        source_message_id="message-1", text="jalan rusak", received_at=received_at,
    )
    intake = IntakeService(runner, ConversationAssembler())
    intake.accept(message, case_key="road")
    AssemblyWorker(runner, ConversationAssembler()).process_due(received_at + timedelta(seconds=5))

    meta1 = AttachmentMetadata(
        object_key="research/message-1/diff.webp", mime_type="image/webp", size_bytes=1024,
        content_hash=sha256(b"meta1").hexdigest(), received_at=received_at + timedelta(hours=1),
        retention_class="evidence",
    )
    meta2 = AttachmentMetadata(
        object_key="research/message-1/diff.webp", mime_type="image/webp", size_bytes=2048,
        content_hash=sha256(b"meta2").hexdigest(), received_at=received_at + timedelta(hours=1),
        retention_class="evidence",
    )
    store = AttachmentStore()

    assert runner.run(
        lambda connection: store.persist(
            connection, message_id=message.message_id, metadata=meta1, now=meta1.received_at
        )
    )
    with pytest.raises(AttachmentAssociationError, match="different metadata"):
        runner.run(
            lambda connection: store.persist(
                connection, message_id=message.message_id, metadata=meta2, now=meta2.received_at
            )
        )


def test_concurrent_attachment_with_different_metadata_is_rejected(runner: TransactionRunner) -> None:
    received_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    message = RawMessage(
        message_id="research:message-1", tenant_id="research", conversation_id="chat-1",
        source_message_id="message-1", text="jalan rusak", received_at=received_at,
    )
    intake = IntakeService(runner, ConversationAssembler())
    intake.accept(message, case_key="road")
    AssemblyWorker(runner, ConversationAssembler()).process_due(received_at + timedelta(seconds=5))

    meta1 = AttachmentMetadata(
        object_key="research/message-1/conflict.webp", mime_type="image/webp", size_bytes=1024,
        content_hash=sha256(b"meta1").hexdigest(), received_at=received_at + timedelta(hours=1),
        retention_class="evidence",
    )
    meta2 = AttachmentMetadata(
        object_key="research/message-1/conflict.webp", mime_type="image/webp", size_bytes=2048,
        content_hash=sha256(b"meta2").hexdigest(), received_at=received_at + timedelta(hours=1),
        retention_class="evidence",
    )
    store = AttachmentStore()

    barrier = threading.Barrier(2)

    def attempt(meta: AttachmentMetadata) -> bool:
        barrier.wait()
        return runner.run(
            lambda connection: store.persist(
                connection, message_id=message.message_id, metadata=meta, now=meta.received_at
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(attempt, meta1)
        f2 = executor.submit(attempt, meta2)

        results = []
        errors = []
        for f in (f1, f2):
            try:
                results.append(f.result())
            except AttachmentAssociationError as e:
                errors.append(e)

    assert results == [True]
    assert len(errors) == 1
    assert "different metadata" in str(errors[0])
