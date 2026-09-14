from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from contracts.models import OutboundMessageCommand, TicketCreateRequest
from services.intake.openwa import OpenWAConnector
from services.outbox.broker import BrokerMessage
from services.outbox.consumer import AtomicInboxConsumer
from services.reliability.client import ReliableTicketClient


class ToolGatewayWorker:
    """Consumes validated command topics; no model or planner can call providers directly."""

    def __init__(
        self,
        consumer: AtomicInboxConsumer,
        ticket_client: ReliableTicketClient,
        messaging_connector: OpenWAConnector,
    ) -> None:
        self._consumer = consumer
        self._ticket_client = ticket_client
        self._messaging_connector = messaging_connector

    def consume_ticket_once(self, max_records: int = 10) -> int:
        return self._consumer.consume_batch(
            topic="commands.ticket.v1",
            handler=self._handle_ticket_command,
            max_records=max_records,
        )

    def consume_message_once(self, max_records: int = 10) -> int:
        return self._consumer.consume_batch(
            topic="commands.message.v1",
            handler=self._handle_message_command,
            max_records=max_records,
        )

    def _handle_ticket_command(self, connection: Connection, event: BrokerMessage) -> None:
        payload = event.payload
        raw_request = payload.get("request")
        idempotency_key = payload.get("idempotency_key")
        if not isinstance(raw_request, dict) or not isinstance(idempotency_key, str):
            raise ValueError("commands.ticket.v1 requires request and idempotency_key")
        request = TicketCreateRequest.model_validate(raw_request)
        receipt = self._ticket_client.create_ticket(request, idempotency_key)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE cases
                SET processing_state = 'TICKETED', ticket_id = %s, updated_at = clock_timestamp()
                WHERE case_id = %s AND tenant_id = %s
                """,
                (receipt.ticket_id, request.case_id, request.tenant_id),
            )
            cursor.execute(
                """
                INSERT INTO audit_traces (tenant_id, case_id, event_type, payload)
                VALUES (%s, %s, 'ticket.executed.v1', %s)
                """,
                (
                    request.tenant_id,
                    request.case_id,
                    Jsonb({
                        "event_id": event.event_id,
                        "ticket_id": receipt.ticket_id,
                        "external_id": receipt.external_id,
                        "idempotency_key": receipt.idempotency_key,
                    }),
                ),
            )

    def _handle_message_command(self, connection: Connection, event: BrokerMessage) -> None:
        command = OutboundMessageCommand.model_validate(event.payload)
        receipt = self._messaging_connector.send_text(command, connection=connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_traces (tenant_id, case_id, event_type, payload)
                VALUES (%s, %s, 'message.executed.v1', %s)
                """,
                (
                    command.tenant_id,
                    command.case_id,
                    Jsonb({
                        "event_id": event.event_id,
                        "send_id": receipt.send_id,
                        "status": receipt.status.value,
                        "source_message_id": receipt.source_message_id,
                        "idempotency_key": receipt.idempotency_key,
                    }),
                ),
            )
