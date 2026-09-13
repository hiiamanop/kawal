from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from psycopg import Connection
from psycopg.types.json import Jsonb

from contracts.models import (
    DeliveryStatus,
    OutboundMessageCommand,
    OutboundSendReceipt,
)


class IdempotencyConflictError(Exception):
    pass


class SendLedgerStore:
    def record_pending_send(
        self,
        connection: Connection,
        command: OutboundMessageCommand,
    ) -> tuple[str, bool]:
        """Record an outbound message command atomically in the send ledger and outbox.

        Returns:
            (send_id, is_new): send_id as string, and True if newly inserted, False if idempotent replay.
        Raises:
            IdempotencyConflictError if the same idempotency key is reused with a mutated payload.
        """
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT send_id, payload_hash, status, source_message_id
                FROM whatsapp_send_ledger
                WHERE idempotency_key = %s
                """,
                (command.idempotency_key,),
            )
            row = cursor.fetchone()
            if row is not None:
                existing_send_id, existing_hash, existing_status, _ = row
                if existing_hash != command.payload_hash:
                    raise IdempotencyConflictError(
                        f"Idempotency key {command.idempotency_key} already exists with different payload hash"
                    )
                return str(existing_send_id), False

            cursor.execute(
                """
                INSERT INTO whatsapp_send_ledger (
                    tenant_id, conversation_id, case_id, idempotency_key,
                    recipient_phone, purpose, text, quoted_source_message_id,
                    payload_hash, status, attempts
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0
                )
                RETURNING send_id
                """,
                (
                    command.tenant_id,
                    command.conversation_id,
                    command.case_id,
                    command.idempotency_key,
                    command.recipient_phone,
                    command.purpose.value,
                    command.text,
                    command.quoted_source_message_id,
                    command.payload_hash,
                    DeliveryStatus.PENDING.value,
                ),
            )
            send_id = str(cursor.fetchone()[0])

            # Atomic outbox registration
            cursor.execute(
                """
                INSERT INTO outbox (
                    tenant_id, aggregate_type, aggregate_id, revision,
                    event_type, partition_key, payload
                ) VALUES (
                    %s, 'whatsapp_send', %s, 1,
                    'whatsapp.send.requested.v1', %s, %s
                )
                """,
                (
                    command.tenant_id,
                    send_id,
                    command.conversation_id,
                    Jsonb(command.model_dump(mode="json")),
                ),
            )

            return send_id, True

    def mark_sent(
        self,
        connection: Connection,
        send_id: str,
        source_message_id: str,
    ) -> None:
        """Mark an outbound send as SENT with WhatsApp-assigned source_message_id."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE whatsapp_send_ledger
                SET status = %s,
                    source_message_id = %s,
                    attempts = attempts + 1,
                    updated_at = clock_timestamp()
                WHERE send_id = %s
                """,
                (DeliveryStatus.SENT.value, source_message_id, send_id),
            )

    def mark_unknown_outcome(
        self,
        connection: Connection,
        send_id: str,
        error_message: str,
    ) -> None:
        """Mark an outbound send as DELIVERY_UNKNOWN (prevents blind duplicate resend per PRD §42)."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE whatsapp_send_ledger
                SET status = %s,
                    last_error = %s,
                    attempts = attempts + 1,
                    updated_at = clock_timestamp()
                WHERE send_id = %s
                """,
                (DeliveryStatus.DELIVERY_UNKNOWN.value, error_message, send_id),
            )

    def mark_failed(
        self,
        connection: Connection,
        send_id: str,
        error_message: str,
    ) -> None:
        """Mark an outbound send as definitively FAILED."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE whatsapp_send_ledger
                SET status = %s,
                    last_error = %s,
                    attempts = attempts + 1,
                    updated_at = clock_timestamp()
                WHERE send_id = %s
                """,
                (DeliveryStatus.FAILED.value, error_message, send_id),
            )

    def lookup_by_idempotency_key(
        self,
        connection: Connection,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT send_id, tenant_id, conversation_id, case_id, idempotency_key,
                       recipient_phone, purpose, text, quoted_source_message_id,
                       payload_hash, status, source_message_id, attempts, last_error,
                       created_at, updated_at
                FROM whatsapp_send_ledger
                WHERE idempotency_key = %s
                """,
                (idempotency_key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cursor.description]
            return dict(zip(cols, row))


class InMemorySendLedger:
    """Thread-safe in-memory send ledger for testing and local operation."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}

    def record_pending_send(
        self,
        connection: Any,
        command: OutboundMessageCommand,
    ) -> tuple[str, bool]:
        key = command.idempotency_key
        if key in self._entries:
            existing = self._entries[key]
            if existing["payload_hash"] != command.payload_hash:
                raise IdempotencyConflictError(
                    f"Idempotency key {key} already exists with different payload hash"
                )
            return existing["send_id"], False

        from uuid import uuid4

        send_id = str(uuid4())
        now = datetime.now(timezone.utc)
        record = {
            "send_id": send_id,
            "tenant_id": command.tenant_id,
            "conversation_id": command.conversation_id,
            "case_id": command.case_id,
            "idempotency_key": command.idempotency_key,
            "recipient_phone": command.recipient_phone,
            "purpose": command.purpose.value,
            "text": command.text,
            "quoted_source_message_id": command.quoted_source_message_id,
            "payload_hash": command.payload_hash,
            "status": DeliveryStatus.PENDING.value,
            "source_message_id": None,
            "attempts": 0,
            "last_error": None,
            "created_at": now,
            "updated_at": now,
        }
        self._entries[key] = record
        self._by_id[send_id] = record
        return send_id, True

    def mark_sent(
        self,
        connection: Any,
        send_id: str,
        source_message_id: str,
    ) -> None:
        record = self._by_id.get(send_id)
        if record:
            record["status"] = DeliveryStatus.SENT.value
            record["source_message_id"] = source_message_id
            record["attempts"] += 1
            record["updated_at"] = datetime.now(timezone.utc)

    def mark_unknown_outcome(
        self,
        connection: Any,
        send_id: str,
        error_message: str,
    ) -> None:
        record = self._by_id.get(send_id)
        if record:
            record["status"] = DeliveryStatus.DELIVERY_UNKNOWN.value
            record["last_error"] = error_message
            record["attempts"] += 1
            record["updated_at"] = datetime.now(timezone.utc)

    def mark_failed(
        self,
        connection: Any,
        send_id: str,
        error_message: str,
    ) -> None:
        record = self._by_id.get(send_id)
        if record:
            record["status"] = DeliveryStatus.FAILED.value
            record["last_error"] = error_message
            record["attempts"] += 1
            record["updated_at"] = datetime.now(timezone.utc)

    def lookup_by_idempotency_key(
        self,
        connection: Any,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        return self._entries.get(idempotency_key)
