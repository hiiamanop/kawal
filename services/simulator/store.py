from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import (
    Category,
    OperationReceipt,
    TicketCreateRequest,
    TicketPriority,
    TicketReceipt,
    TicketStatus,
    TicketVisibility,
)


class IdempotencyConflictError(Exception):
    pass


class TicketNotFoundError(Exception):
    pass


class StaleRevisionError(Exception):
    pass


class InvalidStateTransitionError(Exception):
    pass


class FaultInjectionError(Exception):
    pass


class FaultProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fail_before_commit: bool = False
    commit_then_timeout: bool = False
    http_status: int | None = None
    delay_seconds: float = 0.0
    max_fault_count: int | None = None


class TicketUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, min_length=1, max_length=64_000)
    priority: TicketPriority | None = None
    visibility: TicketVisibility | None = None
    category: Category | None = None
    payload_hash: str = Field(min_length=1)


class TicketTransferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_jurisdiction_id: str = Field(min_length=1)
    target_authority_unit_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=1000)
    payload_hash: str = Field(min_length=1)


class TicketCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str = Field(min_length=1, max_length=1000)
    payload_hash: str = Field(min_length=1)


class TicketRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket_id: str
    external_id: str
    tenant_id: str
    case_id: str
    case_revision: int
    revision: int = 1
    category: Category
    priority: TicketPriority
    visibility: TicketVisibility
    title: str
    description: str
    jurisdiction_id: str
    authority_unit_id: str
    status: TicketStatus
    created_at: datetime
    updated_at: datetime
    custody_trail: tuple[dict[str, Any], ...] = ()
    close_reason: str | None = None


class TicketSimulator:
    def __init__(self) -> None:
        self._lock = Lock()
        self._operations: dict[str, OperationReceipt] = {}
        self._tickets: dict[str, TicketRecord] = {}
        self._fault_profile: FaultProfile | None = None
        self._fault_count: int = 0

    def set_fault_profile(self, profile: FaultProfile) -> None:
        with self._lock:
            self._fault_profile = profile
            self._fault_count = 0

    def clear_fault_profile(self) -> None:
        with self._lock:
            self._fault_profile = None
            self._fault_count = 0

    def _apply_fault(self, phase: str) -> None:
        if self._fault_profile is None:
            return
        if (
            self._fault_profile.max_fault_count is not None
            and self._fault_count >= self._fault_profile.max_fault_count
        ):
            return

        if phase == "before_commit":
            if self._fault_profile.http_status is not None:
                self._fault_count += 1
                raise FaultInjectionError(f"HTTP_{self._fault_profile.http_status}")
            if self._fault_profile.fail_before_commit:
                self._fault_count += 1
                raise FaultInjectionError("FAIL_BEFORE_COMMIT")

        elif phase == "after_commit":
            if self._fault_profile.commit_then_timeout:
                self._fault_count += 1
                raise TimeoutError("COMMIT_THEN_TIMEOUT")

    def create_ticket(
        self, request: TicketCreateRequest, idempotency_key: str
    ) -> TicketReceipt:
        with self._lock:
            existing = self._operations.get(idempotency_key)
            if existing is not None:
                if existing.receipt.payload_hash != request.payload_hash:
                    raise IdempotencyConflictError(idempotency_key)
                return deepcopy(existing.receipt)

            self._apply_fault("before_commit")

            now = datetime.now(timezone.utc)
            ticket_id = str(uuid4())
            external_id = f"SIM-{len(self._tickets) + 1:06d}"
            receipt = TicketReceipt(
                ticket_id=ticket_id,
                external_id=external_id,
                status=TicketStatus.SUBMITTED,
                idempotency_key=idempotency_key,
                payload_hash=request.payload_hash,
                created_at=now,
            )
            ticket = TicketRecord(
                ticket_id=ticket_id,
                external_id=external_id,
                tenant_id=request.tenant_id,
                case_id=request.case_id,
                case_revision=request.case_revision,
                revision=1,
                category=request.category,
                priority=request.priority,
                visibility=request.visibility,
                title=request.title,
                description=request.description,
                jurisdiction_id=request.jurisdiction_id,
                authority_unit_id=request.authority_unit_id,
                status=TicketStatus.SUBMITTED,
                created_at=now,
                updated_at=now,
            )
            self._tickets[ticket_id] = ticket
            self._operations[idempotency_key] = OperationReceipt(
                status="COMPLETED", receipt=receipt
            )

            self._apply_fault("after_commit")

            return deepcopy(receipt)

    def get_ticket(self, ticket_id: str) -> TicketRecord:
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                raise TicketNotFoundError(ticket_id)
            return deepcopy(ticket)

    def update_ticket(
        self,
        ticket_id: str,
        request: TicketUpdateRequest,
        idempotency_key: str,
        if_match_revision: int | None = None,
    ) -> TicketReceipt:
        with self._lock:
            existing = self._operations.get(idempotency_key)
            if existing is not None:
                if existing.receipt.payload_hash != request.payload_hash:
                    raise IdempotencyConflictError(idempotency_key)
                return deepcopy(existing.receipt)

            self._apply_fault("before_commit")

            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                raise TicketNotFoundError(ticket_id)

            if if_match_revision is not None and ticket.revision != if_match_revision:
                raise StaleRevisionError(
                    f"Current revision is {ticket.revision}, but If-Match was {if_match_revision}"
                )

            now = datetime.now(timezone.utc)
            new_revision = ticket.revision + 1
            updated = ticket.model_copy(
                update={
                    "revision": new_revision,
                    "title": request.title if request.title is not None else ticket.title,
                    "description": request.description if request.description is not None else ticket.description,
                    "priority": request.priority if request.priority is not None else ticket.priority,
                    "visibility": request.visibility if request.visibility is not None else ticket.visibility,
                    "category": request.category if request.category is not None else ticket.category,
                    "updated_at": now,
                }
            )
            self._tickets[ticket_id] = updated

            receipt = TicketReceipt(
                ticket_id=ticket.ticket_id,
                external_id=ticket.external_id,
                status=ticket.status,
                idempotency_key=idempotency_key,
                payload_hash=request.payload_hash,
                created_at=now,
            )
            self._operations[idempotency_key] = OperationReceipt(
                status="COMPLETED", receipt=receipt
            )

            self._apply_fault("after_commit")

            return deepcopy(receipt)

    def transfer_ticket(
        self,
        ticket_id: str,
        request: TicketTransferRequest,
        idempotency_key: str,
        if_match_revision: int | None = None,
    ) -> TicketReceipt:
        with self._lock:
            existing = self._operations.get(idempotency_key)
            if existing is not None:
                if existing.receipt.payload_hash != request.payload_hash:
                    raise IdempotencyConflictError(idempotency_key)
                return deepcopy(existing.receipt)

            self._apply_fault("before_commit")

            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                raise TicketNotFoundError(ticket_id)

            if if_match_revision is not None and ticket.revision != if_match_revision:
                raise StaleRevisionError(
                    f"Current revision is {ticket.revision}, but If-Match was {if_match_revision}"
                )

            if ticket.status == TicketStatus.CLOSED:
                raise InvalidStateTransitionError("Cannot transfer a closed ticket")

            now = datetime.now(timezone.utc)
            custody_event = {
                "from_jurisdiction": ticket.jurisdiction_id,
                "from_unit": ticket.authority_unit_id,
                "to_jurisdiction": request.target_jurisdiction_id,
                "to_unit": request.target_authority_unit_id,
                "reason": request.reason,
                "timestamp": now.isoformat(),
            }
            new_trail = ticket.custody_trail + (custody_event,)
            updated = ticket.model_copy(
                update={
                    "revision": ticket.revision + 1,
                    "jurisdiction_id": request.target_jurisdiction_id,
                    "authority_unit_id": request.target_authority_unit_id,
                    "custody_trail": new_trail,
                    "updated_at": now,
                }
            )
            self._tickets[ticket_id] = updated

            receipt = TicketReceipt(
                ticket_id=ticket.ticket_id,
                external_id=ticket.external_id,
                status=ticket.status,
                idempotency_key=idempotency_key,
                payload_hash=request.payload_hash,
                created_at=now,
            )
            self._operations[idempotency_key] = OperationReceipt(
                status="COMPLETED", receipt=receipt
            )

            self._apply_fault("after_commit")

            return deepcopy(receipt)

    def close_ticket(
        self,
        ticket_id: str,
        request: TicketCloseRequest,
        idempotency_key: str,
        if_match_revision: int | None = None,
    ) -> TicketReceipt:
        with self._lock:
            existing = self._operations.get(idempotency_key)
            if existing is not None:
                if existing.receipt.payload_hash != request.payload_hash:
                    raise IdempotencyConflictError(idempotency_key)
                return deepcopy(existing.receipt)

            self._apply_fault("before_commit")

            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                raise TicketNotFoundError(ticket_id)

            if if_match_revision is not None and ticket.revision != if_match_revision:
                raise StaleRevisionError(
                    f"Current revision is {ticket.revision}, but If-Match was {if_match_revision}"
                )

            if ticket.status == TicketStatus.CLOSED:
                raise InvalidStateTransitionError("Ticket is already closed")

            now = datetime.now(timezone.utc)
            updated = ticket.model_copy(
                update={
                    "revision": ticket.revision + 1,
                    "status": TicketStatus.CLOSED,
                    "close_reason": request.reason,
                    "updated_at": now,
                }
            )
            self._tickets[ticket_id] = updated

            receipt = TicketReceipt(
                ticket_id=ticket.ticket_id,
                external_id=ticket.external_id,
                status=TicketStatus.CLOSED,
                idempotency_key=idempotency_key,
                payload_hash=request.payload_hash,
                created_at=now,
            )
            self._operations[idempotency_key] = OperationReceipt(
                status="COMPLETED", receipt=receipt
            )

            self._apply_fault("after_commit")

            return deepcopy(receipt)

    def lookup_operation(self, idempotency_key: str) -> OperationReceipt | None:
        with self._lock:
            receipt = self._operations.get(idempotency_key)
            return deepcopy(receipt) if receipt is not None else None

    @property
    def ticket_count(self) -> int:
        with self._lock:
            return len(self._tickets)

    @property
    def operation_count(self) -> int:
        with self._lock:
            return len(self._operations)
