from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid5

from psycopg import Connection
from psycopg.types.json import Jsonb

from contracts.models import CaseSnapshot, PolicyResult, RawMessage, TicketCommand, TicketReceipt


OUTBOX_NAMESPACE = UUID("00000000-0000-0000-0000-000000000000")
LEASE_DURATION = timedelta(minutes=1)


class IdempotencyConflictError(Exception):
    pass


class M1Store:
    def persist_case(
        self,
        connection: Connection,
        *,
        event_id: UUID,
        snapshot: CaseSnapshot,
        messages: tuple[RawMessage, ...],
        command: TicketCommand,
        policy: PolicyResult,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO inbox (consumer_name, event_id, aggregate_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (consumer_name, event_id) DO NOTHING
                RETURNING event_id
                """,
                ("m1-replay", event_id, snapshot.case_id),
            )
            if cursor.fetchone() is None:
                return
            cursor.execute(
                """
                INSERT INTO cases (
                    case_id, tenant_id, conversation_id, revision, processing_state,
                    category, risk, sensitivity
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (case_id) DO NOTHING
                """,
                (
                    snapshot.case_id,
                    snapshot.tenant_id,
                    snapshot.conversation_id,
                    snapshot.revision,
                    "EXECUTING",
                    command.request.category.value,
                    "MEDIUM",
                    command.request.visibility.value,
                ),
            )
            for message in messages:
                cursor.execute(
                    """
                    INSERT INTO raw_messages (
                        message_id, tenant_id, conversation_id, source_message_id, text, received_at,
                        connector_id, account_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT ON CONSTRAINT uq_raw_messages_source_identity DO NOTHING
                    """,
                    (
                        message.message_id,
                        message.tenant_id,
                        message.conversation_id,
                        message.source_message_id,
                        message.text,
                        message.received_at,
                        getattr(message, "connector_id", None) or "replay",
                        getattr(message, "account_id", None) or "research",
                    ),
                )
            cursor.execute(
                """
                INSERT INTO case_snapshots (
                    case_id, revision, tenant_id, conversation_id, state, evidence_hash,
                    messages_payload, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (case_id, revision) DO NOTHING
                """,
                (
                    snapshot.case_id,
                    snapshot.revision,
                    snapshot.tenant_id,
                    snapshot.conversation_id,
                    snapshot.state.value,
                    snapshot.evidence_hash,
                    Jsonb([message.model_dump(mode="json") for message in messages]),
                    snapshot.created_at,
                ),
            )
            cursor.execute(
                """
                INSERT INTO decisions (
                    decision_id, case_id, input_revision, mode, reason_codes, policy_result, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (case_id, input_revision, sequence)
                DO UPDATE SET decision_id = decisions.decision_id
                RETURNING decision_id
                """,
                (
                    UUID(command.decision.decision_id),
                    command.decision.case_id,
                    command.decision.revision,
                    command.decision.mode.value,
                    list(command.decision.reason_codes),
                    Jsonb(policy.model_dump(mode="json")),
                    command.decision.created_at,
                ),
            )
            decision_id = cursor.fetchone()[0]
            cursor.execute(
                """
                INSERT INTO ticket_commands (
                    tenant_id, case_id, decision_id, idempotency_key, payload_hash, request_payload
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, idempotency_key)
                DO UPDATE SET payload_hash = ticket_commands.payload_hash
                WHERE ticket_commands.payload_hash = EXCLUDED.payload_hash
                RETURNING command_id
                """,
                (
                    command.request.tenant_id,
                    command.request.case_id,
                    decision_id,
                    command.idempotency_key,
                    command.request.payload_hash,
                    Jsonb(command.request.model_dump(mode="json")),
                ),
            )
            if cursor.fetchone() is None:
                raise IdempotencyConflictError(command.idempotency_key)
            cursor.execute(
                """
                INSERT INTO audit_traces (
                    tenant_id, case_id, decision_id, event_type, payload
                ) SELECT %s, %s, %s, %s, %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM audit_traces
                    WHERE decision_id = %s AND event_type = %s
                )
                """,
                (
                    snapshot.tenant_id,
                    snapshot.case_id,
                    decision_id,
                    "ticket.command.created",
                    Jsonb({"reason_codes": list(command.decision.reason_codes)}),
                    decision_id,
                    "ticket.command.created",
                ),
            )
            cursor.execute(
                """
                INSERT INTO outbox (
                    event_id, tenant_id, aggregate_type, aggregate_id, revision,
                    event_type, partition_key, payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (
                    uuid5(OUTBOX_NAMESPACE, command.idempotency_key),
                    command.request.tenant_id,
                    "case",
                    command.request.case_id,
                    command.request.case_revision,
                    "ticket.create.requested",
                    command.request.case_id,
                    Jsonb(
                        {
                            "idempotency_key": command.idempotency_key,
                            "request": command.request.model_dump(mode="json"),
                        }
                    ),
                ),
            )

    def claim_next_outbox_event(
        self, connection: Connection, event_type: str = "ticket.create.requested"
    ) -> tuple[UUID, dict[str, object]] | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH candidate AS (
                    SELECT event_id
                    FROM outbox
                    WHERE event_type = %s
                      AND published_at IS NULL
                      AND (lease_expires_at IS NULL OR lease_expires_at <= clock_timestamp())
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE outbox
                SET attempts = attempts + 1,
                    lease_expires_at = clock_timestamp() + %s::interval
                FROM candidate
                WHERE outbox.event_id = candidate.event_id
                RETURNING outbox.event_id, outbox.payload
                """,
                (event_type, f"{LEASE_DURATION.total_seconds()} seconds"),
            )
            row = cursor.fetchone()
            return None if row is None else (row[0], row[1])

    def mark_ticket_confirmed(
        self, connection: Connection, event_id: UUID, receipt: TicketReceipt
    ) -> None:
        now = datetime.now(timezone.utc)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE outbox
                SET published_at = %s, lease_expires_at = NULL
                WHERE event_id = %s AND published_at IS NULL
                """,
                (now, event_id),
            )
            cursor.execute(
                """
                UPDATE ticket_commands
                SET status = 'CONFIRMED', attempts = attempts + 1, updated_at = %s
                WHERE idempotency_key = %s
                """,
                (now, receipt.idempotency_key),
            )
            cursor.execute(
                """
                UPDATE cases SET processing_state = 'TICKETED', ticket_id = %s, updated_at = %s
                WHERE case_id = %s
                """,
                (UUID(receipt.ticket_id), now, receipt.idempotency_key.split(":")[1]),
            )

    def claim_next_any_outbox_event(
        self, connection: Connection
    ) -> tuple[UUID, str, str, dict[str, object], int] | None:
        """Claim next unpublished outbox event of any type."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH candidate AS (
                    SELECT event_id
                    FROM outbox
                    WHERE published_at IS NULL
                      AND (lease_expires_at IS NULL OR lease_expires_at <= clock_timestamp())
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE outbox
                SET attempts = attempts + 1,
                    lease_expires_at = clock_timestamp() + %s::interval
                FROM candidate
                WHERE outbox.event_id = candidate.event_id
                RETURNING outbox.event_id, outbox.event_type, outbox.partition_key, outbox.payload, outbox.revision
                """,
                (f"{LEASE_DURATION.total_seconds()} seconds",),
            )
            row = cursor.fetchone()
            return None if row is None else (row[0], row[1], row[2], row[3], row[4])

    def mark_event_published(
        self, connection: Connection, event_id: UUID
    ) -> None:
        """Mark an outbox event as published."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE outbox
                SET published_at = clock_timestamp(),
                    lease_expires_at = NULL
                WHERE event_id = %s
                """,
                (event_id,),
            )

    def reconcile_stuck_outbox_leases(
        self,
        connection: Connection,
        max_lease_duration: timedelta = LEASE_DURATION * 10,
    ) -> list[UUID]:
        """Release unpublished leases beyond a bounded duration for safe retry.

        A normal expired lease is already eligible for claim. This is exclusively an
        operator reconciliation for invalid/far-future lease values left by legacy
        code or a corrupted clock. Published events are never modified.
        """
        if max_lease_duration <= timedelta(0):
            raise ValueError("max_lease_duration must be positive")
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE outbox
                SET lease_expires_at = NULL
                WHERE published_at IS NULL
                  AND lease_expires_at > clock_timestamp() + %s::interval
                RETURNING event_id
                """,
                (f"{max_lease_duration.total_seconds()} seconds",),
            )
            return [row[0] for row in cursor.fetchall()]
