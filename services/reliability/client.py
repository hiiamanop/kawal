from __future__ import annotations

from typing import Any, Callable

from contracts.models import (
    OperationReceipt,
    TicketCreateRequest,
    TicketReceipt,
)
from services.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from services.reliability.dlq import DeadLetterQueue
from services.simulator.store import (
    IdempotencyConflictError,
    TicketCloseRequest,
    TicketSimulator,
    TicketTransferRequest,
    TicketUpdateRequest,
)


class MaxRetriesExceededError(Exception):
    pass


class ReliableTicketClient:
    def __init__(
        self,
        simulator: TicketSimulator,
        circuit_breaker: CircuitBreaker | None = None,
        dlq: DeadLetterQueue | None = None,
        max_retries: int = 3,
        sleeper: Callable[[float], None] = lambda _: None,
    ) -> None:
        self.simulator = simulator
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.dlq = dlq or DeadLetterQueue()
        self.max_retries = max_retries
        self._sleeper = sleeper

    def reconcile_create(
        self, request: TicketCreateRequest, idempotency_key: str
    ) -> TicketReceipt | None:
        op = self.simulator.lookup_operation(idempotency_key)
        if op is not None and op.status == "COMPLETED":
            if op.receipt.payload_hash != request.payload_hash:
                raise IdempotencyConflictError(idempotency_key)
            return op.receipt
        return None

    def create_ticket(
        self, request: TicketCreateRequest, idempotency_key: str
    ) -> TicketReceipt:
        reconciled = self.reconcile_create(request, idempotency_key)
        if reconciled is not None:
            return reconciled

        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            if not self.circuit_breaker.allow_request():
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

            if attempt > 1:
                reconciled = self.reconcile_create(request, idempotency_key)
                if reconciled is not None:
                    self.circuit_breaker.record_success()
                    return reconciled

            try:
                receipt = self.simulator.create_ticket(request, idempotency_key)
                self.circuit_breaker.record_success()
                return receipt
            except IdempotencyConflictError:
                self.circuit_breaker.record_failure()
                raise
            except Exception as exc:
                self.circuit_breaker.record_failure()
                last_exception = exc

                reconciled = self.reconcile_create(request, idempotency_key)
                if reconciled is not None:
                    self.circuit_breaker.record_success()
                    return reconciled

                if attempt < self.max_retries:
                    self._sleeper(0.01 * (2 ** (attempt - 1)))
                    continue

        self.dlq.enqueue(
            source_ref=idempotency_key,
            case_id=request.case_id,
            tenant_id=request.tenant_id,
            reason="MAX_RETRIES_EXCEEDED",
            error_type=type(last_exception).__name__ if last_exception else "UnknownError",
            error_message=str(last_exception),
            payload={
                "title": request.title,
                "category": request.category,
                "priority": request.priority,
            },
            attempt_count=self.max_retries,
        )
        raise MaxRetriesExceededError(
            f"Failed to create ticket after {self.max_retries} attempts: {last_exception}"
        ) from last_exception

    def update_ticket(
        self,
        ticket_id: str,
        request: TicketUpdateRequest,
        idempotency_key: str,
        if_match_revision: int | None = None,
    ) -> TicketReceipt:
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        try:
            receipt = self.simulator.update_ticket(
                ticket_id=ticket_id,
                request=request,
                idempotency_key=idempotency_key,
                if_match_revision=if_match_revision,
            )
            self.circuit_breaker.record_success()
            return receipt
        except Exception:
            self.circuit_breaker.record_failure()
            raise

    def transfer_ticket(
        self,
        ticket_id: str,
        request: TicketTransferRequest,
        idempotency_key: str,
        if_match_revision: int | None = None,
    ) -> TicketReceipt:
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        try:
            receipt = self.simulator.transfer_ticket(
                ticket_id=ticket_id,
                request=request,
                idempotency_key=idempotency_key,
                if_match_revision=if_match_revision,
            )
            self.circuit_breaker.record_success()
            return receipt
        except Exception:
            self.circuit_breaker.record_failure()
            raise

    def close_ticket(
        self,
        ticket_id: str,
        request: TicketCloseRequest,
        idempotency_key: str,
        if_match_revision: int | None = None,
    ) -> TicketReceipt:
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        try:
            receipt = self.simulator.close_ticket(
                ticket_id=ticket_id,
                request=request,
                idempotency_key=idempotency_key,
                if_match_revision=if_match_revision,
            )
            self.circuit_breaker.record_success()
            return receipt
        except Exception:
            self.circuit_breaker.record_failure()
            raise

    def lookup_operation(self, idempotency_key: str) -> OperationReceipt | None:
        return self.simulator.lookup_operation(idempotency_key)
