from __future__ import annotations

from typing import Any
from uuid import UUID

from contracts.models import TicketCreateRequest, TicketReceipt
from infra.db import TransactionRunner
from services.core.persistence import M1Store
from services.outbox.broker import BrokerMessage, EventBroker, InMemoryEventBroker
from services.simulator.store import TicketSimulator

EVENT_TYPE_TO_TOPIC: dict[str, str] = {
    "message.received.v1": "intake.messages.v1",
    "case.ready.v1": "cases.ready.v1",
    "ticket.create.requested": "commands.ticket.v1",
    "whatsapp.send.requested.v1": "commands.message.v1",
    "model.escalation.requested.v1": "commands.escalation.v1",
    "ticket.status.updated.v1": "tickets.status.v1",
    "failure.quarantined.v1": "failures.dlq.v1",
}


class OutboxRelay:
    """M1 vertical slice relay for direct ticket dispatch."""

    def __init__(
        self, transaction_runner: TransactionRunner, store: M1Store, simulator: TicketSimulator
    ) -> None:
        self._transaction_runner = transaction_runner
        self._store = store
        self._simulator = simulator

    def lookup_operation(self, idempotency_key: str) -> TicketReceipt | None:
        operation = self._simulator.lookup_operation(idempotency_key)
        return operation.receipt if operation is not None else None

    def dispatch_next(self) -> TicketReceipt | None:
        event = self._transaction_runner.run(
            lambda connection: self._store.claim_next_outbox_event(
                connection, event_type="ticket.create.requested"
            )
        )
        if event is None:
            return None
        event_id, payload = event
        request = TicketCreateRequest.model_validate(payload["request"])
        receipt = self._simulator.create_ticket(request, str(payload["idempotency_key"]))
        self._transaction_runner.run(
            lambda connection: self._store.mark_ticket_confirmed(connection, event_id, receipt)
        )
        return receipt


class OutboxRelayDaemon:
    """PRD §32 transactional outbox relay daemon publishing events to Kafka/Redpanda."""

    def __init__(
        self,
        transaction_runner: TransactionRunner,
        store: M1Store | None = None,
        broker: EventBroker | None = None,
    ) -> None:
        self._transaction_runner = transaction_runner
        self._store = store or M1Store()
        self._broker = broker or InMemoryEventBroker()

    @property
    def broker(self) -> EventBroker:
        return self._broker

    def relay_next(self) -> BrokerMessage | None:
        """Claim next unpublished event and publish it to the designated broker topic."""
        claimed = self._transaction_runner.run(
            lambda conn: self._store.claim_next_any_outbox_event(conn)
        )
        if claimed is None:
            return None

        event_id, event_type, partition_key, payload, revision = claimed
        topic = EVENT_TYPE_TO_TOPIC.get(event_type, f"{event_type}.events")

        # Publish to broker
        msg = self._broker.publish(
            topic=topic,
            key=partition_key,
            payload=payload,
            headers={
                "event_id": str(event_id),
                "event_type": event_type,
                "revision": str(revision),
            },
            event_id=str(event_id),
        )

        # Mark as published in DB
        self._transaction_runner.run(
            lambda conn: self._store.mark_event_published(conn, event_id)
        )
        return msg

    def relay_batch(self, max_batch: int = 50) -> int:
        count = 0
        for _ in range(max_batch):
            msg = self.relay_next()
            if msg is None:
                break
            count += 1
        return count
