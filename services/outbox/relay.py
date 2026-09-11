from contracts.models import TicketCreateRequest, TicketReceipt
from infra.db import TransactionRunner
from services.core.persistence import M1Store
from services.simulator.store import TicketSimulator


class OutboxRelay:
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
