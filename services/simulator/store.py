from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from contracts.models import OperationReceipt, TicketCreateRequest, TicketReceipt, TicketStatus


class IdempotencyConflictError(Exception):
    pass


class TicketSimulator:
    def __init__(self) -> None:
        self._operations: dict[str, OperationReceipt] = {}

    def create_ticket(
        self, request: TicketCreateRequest, idempotency_key: str
    ) -> TicketReceipt:
        existing = self._operations.get(idempotency_key)
        if existing is not None:
            if existing.receipt.payload_hash != request.payload_hash:
                raise IdempotencyConflictError(idempotency_key)
            return deepcopy(existing.receipt)

        receipt = TicketReceipt(
            ticket_id=str(uuid4()),
            external_id=f"SIM-{len(self._operations) + 1:06d}",
            status=TicketStatus.SUBMITTED,
            idempotency_key=idempotency_key,
            payload_hash=request.payload_hash,
            created_at=datetime.now(timezone.utc),
        )
        self._operations[idempotency_key] = OperationReceipt(
            status="COMPLETED", receipt=receipt
        )
        return deepcopy(receipt)

    def lookup_operation(self, idempotency_key: str) -> OperationReceipt | None:
        receipt = self._operations.get(idempotency_key)
        return deepcopy(receipt) if receipt is not None else None

    @property
    def ticket_count(self) -> int:
        return len(self._operations)
