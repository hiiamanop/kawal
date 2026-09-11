from fastapi import FastAPI, Header, HTTPException, Response, status

from contracts.models import OperationReceipt, TicketCreateRequest, TicketReceipt
from services.simulator.store import IdempotencyConflictError, TicketSimulator


def create_app(simulator: TicketSimulator | None = None) -> FastAPI:
    store = simulator or TicketSimulator()
    app = FastAPI(title="KAWAL Ticket Simulator", version="0.1.0")

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
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="IDEMPOTENCY_PAYLOAD_MISMATCH")
        if existing is not None:
            response.status_code = status.HTTP_200_OK
        return receipt

    @app.get("/v1/operations/{idempotency_key}", response_model=OperationReceipt)
    def lookup_operation(idempotency_key: str) -> OperationReceipt:
        operation = store.lookup_operation(idempotency_key)
        if operation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OPERATION_NOT_FOUND")
        return operation

    return app


app = create_app()
