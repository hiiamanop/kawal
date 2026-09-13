from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Response, status

from contracts.models import OperationReceipt, TicketCreateRequest, TicketReceipt
from services.simulator.store import (
    FaultInjectionError,
    FaultProfile,
    IdempotencyConflictError,
    InvalidStateTransitionError,
    StaleRevisionError,
    TicketCloseRequest,
    TicketNotFoundError,
    TicketRecord,
    TicketSimulator,
    TicketTransferRequest,
    TicketUpdateRequest,
)


def _parse_if_match(if_match: str | None) -> int | None:
    if if_match is None:
        return None
    raw = if_match.strip().strip('"')
    try:
        return int(raw)
    except ValueError:
        return None


def _handle_fault(exc: Exception) -> None:
    if isinstance(exc, FaultInjectionError):
        msg = str(exc)
        if msg.startswith("HTTP_"):
            try:
                code = int(msg.split("_")[1])
                raise HTTPException(status_code=code, detail=msg)
            except ValueError:
                pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    if isinstance(exc, TimeoutError):
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="SIMULATED_TIMEOUT")


def create_app(simulator: TicketSimulator | None = None) -> FastAPI:
    store = simulator or TicketSimulator()
    app = FastAPI(title="KAWAL Ticket Simulator", version="0.2.0")

    @app.get("/health", status_code=status.HTTP_200_OK)
    def health_check() -> dict[str, str]:
        return {"status": "healthy"}

    @app.post("/v1/tickets", response_model=TicketReceipt, status_code=status.HTTP_201_CREATED)
    def create_ticket(
        request: TicketCreateRequest,
        response: Response,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
    ) -> TicketReceipt:
        try:
            existing = store.lookup_operation(idempotency_key)
            receipt = store.create_ticket(request, idempotency_key)
        except IdempotencyConflictError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="IDEMPOTENCY_PAYLOAD_MISMATCH",
            )
        except (FaultInjectionError, TimeoutError) as exc:
            _handle_fault(exc)
            raise

        if existing is not None:
            response.status_code = status.HTTP_200_OK
        return receipt

    @app.get("/v1/operations/{idempotency_key}", response_model=OperationReceipt)
    def lookup_operation(idempotency_key: str) -> OperationReceipt:
        operation = store.lookup_operation(idempotency_key)
        if operation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="OPERATION_NOT_FOUND",
            )
        return operation

    @app.get("/v1/tickets/{ticket_id}", response_model=TicketRecord)
    def get_ticket(ticket_id: str) -> TicketRecord:
        try:
            return store.get_ticket(ticket_id)
        except TicketNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TICKET_NOT_FOUND",
            )

    @app.patch("/v1/tickets/{ticket_id}", response_model=TicketReceipt)
    def update_ticket(
        ticket_id: str,
        request: TicketUpdateRequest,
        response: Response,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> TicketReceipt:
        revision = _parse_if_match(if_match)
        try:
            existing = store.lookup_operation(idempotency_key)
            receipt = store.update_ticket(
                ticket_id=ticket_id,
                request=request,
                idempotency_key=idempotency_key,
                if_match_revision=revision,
            )
        except IdempotencyConflictError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="IDEMPOTENCY_PAYLOAD_MISMATCH",
            )
        except TicketNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TICKET_NOT_FOUND",
            )
        except StaleRevisionError as exc:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail=str(exc),
            )
        except (FaultInjectionError, TimeoutError) as exc:
            _handle_fault(exc)
            raise

        if existing is not None:
            response.status_code = status.HTTP_200_OK
        return receipt

    @app.post("/v1/tickets/{ticket_id}/transfer", response_model=TicketReceipt)
    def transfer_ticket(
        ticket_id: str,
        request: TicketTransferRequest,
        response: Response,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> TicketReceipt:
        revision = _parse_if_match(if_match)
        try:
            existing = store.lookup_operation(idempotency_key)
            receipt = store.transfer_ticket(
                ticket_id=ticket_id,
                request=request,
                idempotency_key=idempotency_key,
                if_match_revision=revision,
            )
        except IdempotencyConflictError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="IDEMPOTENCY_PAYLOAD_MISMATCH",
            )
        except TicketNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TICKET_NOT_FOUND",
            )
        except StaleRevisionError as exc:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail=str(exc),
            )
        except InvalidStateTransitionError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        except (FaultInjectionError, TimeoutError) as exc:
            _handle_fault(exc)
            raise

        if existing is not None:
            response.status_code = status.HTTP_200_OK
        return receipt

    @app.post("/v1/tickets/{ticket_id}/close", response_model=TicketReceipt)
    def close_ticket(
        ticket_id: str,
        request: TicketCloseRequest,
        response: Response,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> TicketReceipt:
        revision = _parse_if_match(if_match)
        try:
            existing = store.lookup_operation(idempotency_key)
            receipt = store.close_ticket(
                ticket_id=ticket_id,
                request=request,
                idempotency_key=idempotency_key,
                if_match_revision=revision,
            )
        except IdempotencyConflictError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="IDEMPOTENCY_PAYLOAD_MISMATCH",
            )
        except TicketNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TICKET_NOT_FOUND",
            )
        except StaleRevisionError as exc:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail=str(exc),
            )
        except InvalidStateTransitionError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        except (FaultInjectionError, TimeoutError) as exc:
            _handle_fault(exc)
            raise

        if existing is not None:
            response.status_code = status.HTTP_200_OK
        return receipt

    @app.post("/v1/test/fault-profiles", status_code=status.HTTP_200_OK)
    def set_fault_profile(profile: FaultProfile) -> dict[str, str]:
        store.set_fault_profile(profile)
        return {"status": "FAULT_PROFILE_SET"}

    @app.delete("/v1/test/fault-profiles", status_code=status.HTTP_200_OK)
    def clear_fault_profile() -> dict[str, str]:
        store.clear_fault_profile()
        return {"status": "FAULT_PROFILE_CLEARED"}

    return app


app = create_app()
