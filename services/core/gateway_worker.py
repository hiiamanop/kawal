from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from contracts.models import OutboundMessageCommand, PolicyInput, TicketCreateRequest
from services.core.escalation import EscalationCommand
from services.core.model_gateway import ModelGateway
from services.core.opa import OpaPolicyClient
from services.core.policy import evaluate_ticket_creation
from services.intake.openwa import OpenWAConnector
from services.outbox.broker import BrokerMessage
from services.observability import metrics
from services.outbox.consumer import AtomicInboxConsumer
from services.reliability.client import ReliableTicketClient


class UnprocessableCommandError(ValueError):
    pass


class ToolGatewayWorker:
    """Consumes validated command topics; no model or planner can call providers directly."""

    def __init__(
        self,
        consumer: AtomicInboxConsumer,
        ticket_client: ReliableTicketClient,
        messaging_connector: OpenWAConnector,
        policy_client: OpaPolicyClient | None = None,
        model_gateway: ModelGateway | None = None,
    ) -> None:
        self._consumer = consumer
        self._ticket_client = ticket_client
        self._messaging_connector = messaging_connector
        self._policy_client = policy_client
        self._model_gateway = model_gateway

    def consume_ticket_once(self, max_records: int = 10) -> int:
        return self._consumer.consume_batch(
            topic="commands.ticket.v1",
            handler=self._handle_ticket_command,
            max_records=max_records,
            on_error=self._handle_unprocessable_command,
        )

    def consume_message_once(self, max_records: int = 10) -> int:
        return self._consumer.consume_batch(
            topic="commands.message.v1",
            handler=self._handle_message_command,
            max_records=max_records,
            on_error=self._handle_unprocessable_command,
        )

    def consume_escalation_once(self, max_records: int = 10) -> int:
        return self._consumer.consume_batch(
            topic="commands.escalation.v1",
            handler=self._handle_escalation_command,
            max_records=max_records,
            on_error=self._handle_unprocessable_command,
        )

    @staticmethod
    def _handle_unprocessable_command(event: BrokerMessage, error: Exception) -> bool:
        # If the case/tenant was deleted from DB, consuming the stale command is terminal.
        import psycopg
        if isinstance(error, psycopg.errors.ForeignKeyViolation):
            return True
        return isinstance(error, UnprocessableCommandError)

    def _handle_ticket_command(self, connection: Connection, event: BrokerMessage) -> None:
        payload = event.payload
        raw_request = payload.get("request")
        idempotency_key = payload.get("idempotency_key")
        if not isinstance(raw_request, dict) or not isinstance(idempotency_key, str):
            raise ValueError("commands.ticket.v1 requires request and idempotency_key")
        request = TicketCreateRequest.model_validate(raw_request)
        policy_input = PolicyInput(
            tenant_id=request.tenant_id,
            case_id=request.case_id,
            revision=request.case_revision,
            category=request.category,
            jurisdiction_id=request.jurisdiction_id,
            authority_unit_id=request.authority_unit_id,
            missing_fields=(),
            has_mandatory_evidence=True,
            idempotency_key=idempotency_key,
        )
        policy = (
            self._policy_client.evaluate_ticket_creation(policy_input)
            if self._policy_client is not None
            else evaluate_ticket_creation(policy_input)
        )
        if policy.decision != "ALLOW":
            metrics.increment("kawal_ticket_commands_total", labels={"outcome": "policy_denied"})
            self._write_ticket_policy_denial(connection, request, event, policy.model_dump(mode="json"))
            return

        receipt = self._ticket_client.create_ticket(request, idempotency_key)
        metrics.increment("kawal_ticket_commands_total", labels={"outcome": "executed"})
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

    def _handle_escalation_command(self, connection: Connection, event: BrokerMessage) -> None:
        command = EscalationCommand.model_validate(event.payload)
        if self._model_gateway is None:
            result_payload = {"status": "UNAVAILABLE", "reason": "MODEL_GATEWAY_NOT_CONFIGURED"}
        else:
            gate, result = self._model_gateway.escalate(command.prompt, command.egress_request)
            result_payload = {
                "status": result.status,
                "reason": result.reason,
                "provider": result.provider,
                "model_id": result.model_id,
                "policy_allowed": gate.allowed,
                "rule_ids": list(gate.rule_ids),
            }
        metrics.increment("kawal_escalation_commands_total", labels={"status": result_payload["status"]})
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_traces (tenant_id, case_id, event_type, payload)
                VALUES (%s, %s, 'model.escalation.executed.v1', %s)
                """,
                (
                    command.tenant_id,
                    command.case_id,
                    Jsonb({"event_id": event.event_id, "task_id": command.task_id, "result": result_payload}),
                ),
            )

    @staticmethod
    def _write_ticket_policy_denial(
        connection: Connection,
        request: TicketCreateRequest,
        event: BrokerMessage,
        policy: dict[str, Any],
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE cases
                SET processing_state = 'BLOCKED', updated_at = clock_timestamp()
                WHERE case_id = %s AND tenant_id = %s
                """,
                (request.case_id, request.tenant_id),
            )
            cursor.execute(
                """
                INSERT INTO audit_traces (tenant_id, case_id, event_type, payload)
                VALUES (%s, %s, 'ticket.policy_denied.v1', %s)
                """,
                (
                    request.tenant_id,
                    request.case_id,
                    Jsonb({"event_id": event.event_id, "policy": policy}),
                ),
            )

    def _handle_message_command(self, connection: Connection, event: BrokerMessage) -> None:
        command = OutboundMessageCommand.model_validate(event.payload)
        receipt = self._messaging_connector.send_text(command, connection=connection)
        metrics.increment("kawal_message_commands_total", labels={"status": receipt.status.value})
        with connection.cursor() as cursor:
            # Delete any duplicate outbox item produced by record_pending_send to prevent loop
            cursor.execute(
                """
                DELETE FROM outbox
                WHERE aggregate_type = 'whatsapp_send'
                  AND aggregate_id = %s
                  AND published_at IS NULL
                """,
                (receipt.send_id,),
            )
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
