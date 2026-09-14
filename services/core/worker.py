from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from contracts.models import CaseSnapshot, ProcessingState, RawMessage
from services.core.pipeline import CaseProcessingPipeline, PipelineResult
from services.outbox.broker import BrokerMessage, EventBroker
from services.outbox.consumer import AtomicInboxConsumer


class MissingCaseSnapshotError(ValueError):
    pass


class CaseReadyWorker:
    """Live worker for cases.ready.v1 events with inbox dedup and revision guard."""

    def __init__(
        self,
        consumer: AtomicInboxConsumer,
        pipeline: CaseProcessingPipeline,
    ) -> None:
        self._consumer = consumer
        self._pipeline = pipeline

    def consume_once(self, max_records: int = 10) -> int:
        return self._consumer.consume_batch(
            topic="cases.ready.v1",
            handler=self._handle_case_ready,
            max_records=max_records,
            on_error=self._handle_unprocessable_event,
        )

    @staticmethod
    def _handle_unprocessable_event(event: BrokerMessage, error: Exception) -> bool:
        # A structurally valid broker event for a deleted/nonexistent snapshot cannot
        # become processable through transport retry; consume it without mutating a case.
        return isinstance(error, MissingCaseSnapshotError)

    def _handle_case_ready(self, connection: Connection, event: BrokerMessage) -> None:
        payload = event.payload
        case_id = str(payload.get("case_id") or payload.get("aggregate_id") or "")
        if not case_id:
            raise ValueError("cases.ready.v1 requires case_id")

        snapshot = self._load_current_snapshot(connection, case_id)
        if snapshot is None:
            raise MissingCaseSnapshotError(f"No snapshot found for case {case_id}")

        # Stale events are retained in inbox/audit, but cannot overwrite a newer case revision.
        expected_revision = int(event.headers.get("revision", snapshot.revision))
        if expected_revision != snapshot.revision:
            self._write_audit(
                connection,
                snapshot,
                "case.ready.stale.v1",
                {"event_id": event.event_id, "expected_revision": expected_revision, "current_revision": snapshot.revision},
            )
            return

        self._mark_analyzing(connection, snapshot.case_id, snapshot.revision)
        result = self._pipeline.process_snapshot(snapshot)
        self._persist_result(connection, snapshot, event, result)

    @staticmethod
    def _load_current_snapshot(connection: Connection, case_id: str) -> CaseSnapshot | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT case_id, revision, tenant_id, conversation_id, state, evidence_hash,
                       messages_payload, created_at
                FROM case_snapshots
                WHERE case_id = %s
                ORDER BY revision DESC
                LIMIT 1
                """,
                (case_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        raw_messages = row[6]
        messages = tuple(RawMessage.model_validate(item) for item in raw_messages)
        return CaseSnapshot(
            case_id=row[0],
            revision=row[1],
            tenant_id=row[2],
            conversation_id=row[3],
            state=ProcessingState(row[4]),
            evidence_hash=row[5],
            messages=messages,
            created_at=row[7],
        )

    @staticmethod
    def _mark_analyzing(connection: Connection, case_id: str, revision: int) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE cases
                SET processing_state = 'ANALYZING', updated_at = clock_timestamp()
                WHERE case_id = %s AND revision = %s
                """,
                (case_id, revision),
            )

    def _persist_result(
        self,
        connection: Connection,
        snapshot: CaseSnapshot,
        event: BrokerMessage,
        result: PipelineResult,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE cases
                SET processing_state = %s,
                    category = %s,
                    risk = %s,
                    updated_at = clock_timestamp()
                WHERE case_id = %s AND revision = %s
                """,
                (
                    result.processing_state.value,
                    result.category.value,
                    result.risk.value,
                    snapshot.case_id,
                    snapshot.revision,
                ),
            )

            # Deferred ticket command outbox insertion
            if result.ticket_command is not None:
                cmd = result.ticket_command
                cursor.execute(
                    """
                    INSERT INTO outbox (
                        tenant_id, aggregate_type, aggregate_id, revision,
                        event_type, partition_key, payload
                    ) VALUES (%s, 'case', %s, %s, 'ticket.create.requested', %s, %s)
                    """,
                    (
                        snapshot.tenant_id,
                        snapshot.case_id,
                        snapshot.revision,
                        f"{snapshot.tenant_id}:{snapshot.case_id}",
                        Jsonb({
                            "request": cmd.request.model_dump(mode="json"),
                            "idempotency_key": cmd.idempotency_key,
                        }),
                    ),
                )

            # Deferred clarification message outbox insertion
            if result.outbound_command is not None:
                cmd_out = result.outbound_command
                cursor.execute(
                    """
                    INSERT INTO outbox (
                        tenant_id, aggregate_type, aggregate_id, revision,
                        event_type, partition_key, payload
                    ) VALUES (%s, 'case', %s, %s, 'whatsapp.send.requested.v1', %s, %s)
                    """,
                    (
                        snapshot.tenant_id,
                        snapshot.case_id,
                        snapshot.revision,
                        f"{snapshot.tenant_id}:{snapshot.case_id}",
                        Jsonb(cmd_out.model_dump(mode="json")),
                    ),
                )

        self._write_audit(
            connection,
            snapshot,
            "case.processed.v1",
            {
                "event_id": event.event_id,
                "decision_mode": result.decision_mode.value,
                "processing_state": result.processing_state.value,
                "category": result.category.value,
                "risk": result.risk.value,
                "completeness": result.completeness,
                "has_ticket_command": result.ticket_command is not None,
                "has_outbound_command": result.outbound_command is not None,
                "audit_trace": result.audit_trace,
            },
        )

    @staticmethod
    def _write_audit(
        connection: Connection,
        snapshot: CaseSnapshot,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_traces (tenant_id, case_id, event_type, payload)
                VALUES (%s, %s, %s, %s)
                """,
                (snapshot.tenant_id, snapshot.case_id, event_type, Jsonb(payload)),
            )
