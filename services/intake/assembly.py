from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4, uuid5

from psycopg import Connection
from psycopg.types.json import Jsonb

from contracts.models import RawMessage


QUIET_WINDOW = timedelta(seconds=5)
BURST_LIMIT = timedelta(seconds=20)
ASSEMBLY_LEASE = timedelta(seconds=30)
ASSOCIATION_HORIZON = timedelta(hours=48)
OUTBOX_NAMESPACE = UUID("00000000-0000-0000-0000-000000000000")
CASE_NAMESPACE = UUID("00000000-0000-0000-0000-000000000002")
INTAKE_OUTBOX_LEASE_FAR_FUTURE = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


class IntakeConflictError(Exception):
    pass


@dataclass(frozen=True)
class AssemblyClaim:
    tenant_id: str
    connector_id: str
    account_id: str
    conversation_id: str
    generation: int
    lease_token: UUID
    assembled_through: datetime | None = None
    assembled_through_message_id: str | None = None


class ConversationAssembler:
    def ingest(
        self,
        connection: Connection,
        message: RawMessage,
        quoted_source_message_id: str | None = None,
        case_key: str | None = None,
        connector_id: str = "replay",
        account_id: str = "research",
    ) -> bool:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO raw_messages (
                    message_id, tenant_id, conversation_id, source_message_id, text, received_at,
                    quoted_source_message_id, case_key, connector_id, account_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, connector_id, account_id, source_message_id)
                DO NOTHING
                RETURNING message_id
                """,
                (
                    message.message_id,
                    message.tenant_id,
                    message.conversation_id,
                    message.source_message_id,
                    message.text,
                    message.received_at,
                    quoted_source_message_id,
                    case_key,
                    connector_id,
                    account_id,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """
                    SELECT conversation_id, text, received_at, quoted_source_message_id, case_key
                    FROM raw_messages
                    WHERE tenant_id = %s AND connector_id = %s AND account_id = %s
                      AND source_message_id = %s
                    """,
                    (message.tenant_id, connector_id, account_id, message.source_message_id),
                )
                existing = cursor.fetchone()
                if existing != (
                    message.conversation_id,
                    message.text,
                    message.received_at,
                    quoted_source_message_id,
                    case_key,
                ):
                    raise IntakeConflictError(message.source_message_id)
                return False
            cursor.execute(
                """
                INSERT INTO assembly_timers (
                    tenant_id, connector_id, account_id, conversation_id, due_at, hard_due_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, connector_id, account_id, conversation_id) DO UPDATE
                SET due_at = CASE
                        WHEN assembly_timers.fired_at IS NOT NULL THEN EXCLUDED.due_at
                        WHEN assembly_timers.lease_token IS NOT NULL THEN EXCLUDED.due_at
                        ELSE LEAST(assembly_timers.hard_due_at, EXCLUDED.due_at)
                    END,
                    hard_due_at = CASE
                        WHEN assembly_timers.fired_at IS NOT NULL THEN EXCLUDED.hard_due_at
                        WHEN assembly_timers.lease_token IS NOT NULL THEN EXCLUDED.hard_due_at
                        ELSE assembly_timers.hard_due_at
                    END,
                    generation = CASE
                        WHEN assembly_timers.fired_at IS NOT NULL THEN assembly_timers.generation + 1
                        WHEN assembly_timers.lease_token IS NOT NULL THEN assembly_timers.generation + 1
                        ELSE assembly_timers.generation
                    END,
                    fired_at = NULL,
                    lease_token = CASE
                        WHEN assembly_timers.fired_at IS NOT NULL THEN NULL
                        ELSE assembly_timers.lease_token
                    END,
                    lease_expires_at = CASE
                        WHEN assembly_timers.fired_at IS NOT NULL THEN NULL
                        ELSE assembly_timers.lease_expires_at
                    END
                """,
                (
                    message.tenant_id,
                    connector_id,
                    account_id,
                    message.conversation_id,
                    message.received_at + QUIET_WINDOW,
                    message.received_at + BURST_LIMIT,
                ),
            )
            event_id = uuid5(
                OUTBOX_NAMESPACE,
                f"{message.tenant_id}:{connector_id}:{account_id}:{message.source_message_id}:message.received.v1",
            )
            payload = {
                **message.model_dump(mode="json"),
                "quoted_source_message_id": quoted_source_message_id,
                "case_key": case_key,
                "connector_id": connector_id,
                "account_id": account_id,
            }
            cursor.execute(
                """
                INSERT INTO outbox (
                    event_id, tenant_id, aggregate_type, aggregate_id, revision,
                    event_type, partition_key, payload, lease_expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (
                    event_id,
                    message.tenant_id,
                    "raw_message",
                    message.message_id,
                    1,
                    "message.received.v1",
                    message.conversation_id,
                    Jsonb(payload),
                    INTAKE_OUTBOX_LEASE_FAR_FUTURE,
                ),
            )
            return True

    def claim(self, connection: Connection, now: datetime) -> AssemblyClaim | None:
        with connection.cursor() as cursor:
            claim_id = uuid4()
            cursor.execute(
                """
                WITH candidate AS (
                    SELECT tenant_id, connector_id, account_id, conversation_id
                    FROM assembly_timers
                    WHERE fired_at IS NULL
                      AND due_at <= %s AND (lease_expires_at IS NULL OR lease_expires_at <= %s)
                    ORDER BY due_at
                    FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE assembly_timers AS timer
                SET lease_token = %s,
                    lease_expires_at = %s,
                    generation = timer.generation + 1
                FROM candidate
                WHERE timer.tenant_id = candidate.tenant_id
                  AND timer.connector_id = candidate.connector_id
                  AND timer.account_id = candidate.account_id
                  AND timer.conversation_id = candidate.conversation_id
                RETURNING timer.tenant_id, timer.connector_id, timer.account_id,
                          timer.conversation_id, timer.generation, timer.lease_token,
                          timer.assembled_through, timer.assembled_through_message_id
                """,
                (now, now, claim_id, now + ASSEMBLY_LEASE),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return AssemblyClaim(
                tenant_id=row[0],
                connector_id=row[1],
                account_id=row[2],
                conversation_id=row[3],
                generation=row[4],
                lease_token=row[5],
                assembled_through=row[6],
                assembled_through_message_id=row[7],
            )

    def assemble(
        self, connection: Connection, claim: AssemblyClaim, now: datetime
    ) -> tuple[datetime | None, str | None]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT message.message_id, message.source_message_id,
                       message.quoted_source_message_id, message.case_key, message.received_at
                FROM raw_messages AS message
                LEFT JOIN message_case_links AS link ON link.message_id = message.message_id
                WHERE message.tenant_id = %s AND message.connector_id = %s AND message.account_id = %s
                  AND message.conversation_id = %s
                  AND message.received_at <= %s
                  AND link.message_id IS NULL
                ORDER BY message.received_at, message.message_id
                """,
                (
                    claim.tenant_id, claim.connector_id, claim.account_id,
                    claim.conversation_id, now,
                ),
            )
            messages = cursor.fetchall()
            for message_id, source_message_id, quoted_source_message_id, case_key, _received_at in messages:
                case_id = self._case_id(
                    cursor, claim.tenant_id, claim.connector_id, claim.account_id, claim.conversation_id,
                    source_message_id, quoted_source_message_id, case_key, now,
                )
                cursor.execute(
                    """
                    INSERT INTO cases (case_id, tenant_id, conversation_id, processing_state)
                    VALUES (%s, %s, %s, 'READY') ON CONFLICT (case_id) DO NOTHING
                    """,
                    (case_id, claim.tenant_id, claim.conversation_id),
                )
                cursor.execute(
                    """
                    INSERT INTO message_case_links (message_id, case_id)
                    VALUES (%s, %s) ON CONFLICT (message_id) DO NOTHING
                    """,
                    (message_id, case_id),
                )
            if messages:
                max_received_at = max(m[4] for m in messages)
                last_message = messages[-1]
                return max_received_at, last_message[0]
            return None, None

    def finalize(
        self,
        connection: Connection,
        claim: AssemblyClaim,
        now: datetime,
        last_received_at: datetime | None = None,
        last_message_id: str | None = None,
    ) -> bool:
        with connection.cursor() as cursor:
            if last_received_at is not None:
                cursor.execute(
                    """
                    UPDATE assembly_timers
                    SET assembled_through = GREATEST(COALESCE(assembly_timers.assembled_through, '-infinity'::timestamptz), %s),
                        assembled_through_message_id = COALESCE(%s, assembly_timers.assembled_through_message_id),
                        fired_at = %s,
                        lease_token = NULL,
                        lease_expires_at = NULL
                    WHERE tenant_id = %s AND connector_id = %s AND account_id = %s
                      AND conversation_id = %s AND lease_token = %s AND generation = %s
                    """,
                    (
                        last_received_at, last_message_id, now,
                        claim.tenant_id, claim.connector_id, claim.account_id,
                        claim.conversation_id, claim.lease_token, claim.generation,
                    ),
                )
            else:
                cursor.execute(
                    """
                    UPDATE assembly_timers
                    SET fired_at = %s,
                        lease_token = NULL,
                        lease_expires_at = NULL
                    WHERE tenant_id = %s AND connector_id = %s AND account_id = %s
                      AND conversation_id = %s AND lease_token = %s AND generation = %s
                    """,
                    (
                        now,
                        claim.tenant_id, claim.connector_id, claim.account_id,
                        claim.conversation_id, claim.lease_token, claim.generation,
                    ),
                )
            if cursor.rowcount == 0:
                raise IntakeConflictError(f"Lease lost or generation conflict for {claim.conversation_id}")
            return True

    def assemble_next(self, connection: Connection, now: datetime) -> str | None:
        claim = self.claim(connection, now)
        if claim is None:
            return None
        last_received_at, last_message_id = self.assemble(connection, claim, now)
        self.finalize(connection, claim, now, last_received_at, last_message_id)
        return claim.conversation_id

    def _case_id(
        self,
        cursor,
        tenant_id: str,
        connector_id: str,
        account_id: str,
        conversation_id: str,
        source_message_id: str,
        quoted_source_message_id: str | None,
        case_key: str | None,
        now: datetime,
    ) -> str:
        if quoted_source_message_id is not None:
            cursor.execute(
                """
                SELECT link.case_id FROM message_case_links AS link
                JOIN raw_messages AS message ON message.message_id = link.message_id
                WHERE message.tenant_id = %s AND message.connector_id = %s AND message.account_id = %s
                  AND message.conversation_id = %s AND message.source_message_id = %s
                """,
                (tenant_id, connector_id, account_id, conversation_id, quoted_source_message_id),
            )
            linked = cursor.fetchone()
            if linked is not None:
                return linked[0]
        if case_key is not None:
            return f"case-{uuid5(CASE_NAMESPACE, f'{tenant_id}:{connector_id}:{account_id}:{conversation_id}:{case_key}')}"
        cursor.execute(
            """
            SELECT link.case_id FROM message_case_links AS link
            JOIN raw_messages AS message ON message.message_id = link.message_id
            WHERE message.tenant_id = %s AND message.connector_id = %s AND message.account_id = %s
              AND message.conversation_id = %s AND message.received_at >= %s
            ORDER BY message.received_at DESC, message.message_id DESC LIMIT 1
            """,
            (tenant_id, connector_id, account_id, conversation_id, now - ASSOCIATION_HORIZON),
        )
        linked = cursor.fetchone()
        if linked is not None:
            return linked[0]
        return f"case-{uuid5(CASE_NAMESPACE, f'{tenant_id}:{connector_id}:{account_id}:{conversation_id}:{source_message_id}')}"
