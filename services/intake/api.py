from __future__ import annotations

import hmac
import os
from typing import Any, Mapping

from fastapi import FastAPI, Header, HTTPException, Request, Response, status

from services.intake.openwa import OpenWAConnector
from services.observability import metrics


def _message_payload_from_event(payload: Mapping[str, Any], tenant_id: str) -> dict[str, Any] | None:
    event_name = str(payload.get("event", "message.received")).strip().lower()
    if event_name != "message.received":
        return None
    raw = payload.get("data")
    source = raw if isinstance(raw, Mapping) else payload
    normalized = dict(source)
    normalized["tenant_id"] = tenant_id
    normalized.setdefault("source_message_id", source.get("id"))
    normalized.setdefault("conversation_id", source.get("chatId") or source.get("chat_id") or source.get("from"))
    normalized.setdefault("text", source.get("body") or source.get("text"))
    normalized.setdefault("received_at", source.get("timestamp") or source.get("createdAt"))
    return normalized


def create_intake_app(
    connector: OpenWAConnector,
    webhook_secret: str,
    tenant_id: str,
) -> FastAPI:
    if not webhook_secret:
        raise ValueError("webhook_secret is required")
    if not tenant_id:
        raise ValueError("tenant_id is required")

    app = FastAPI(title="KAWAL OpenWA Intake", version="0.1.0")

    @app.get("/health")
    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(metrics.render_prometheus(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/webhooks/openwa", status_code=status.HTTP_202_ACCEPTED)
    async def receive_openwa_event(
        request: Request,
        x_kawal_webhook_secret: str | None = Header(default=None),
    ) -> dict[str, str]:
        if x_kawal_webhook_secret is None or not hmac.compare_digest(
            x_kawal_webhook_secret, webhook_secret
        ):
            metrics.increment("kawal_webhook_requests_total", labels={"outcome": "unauthorized"})
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="WEBHOOK_UNAUTHORIZED")
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="INVALID_JSON") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="INVALID_EVENT_PAYLOAD")

        message_payload = _message_payload_from_event(payload, tenant_id)
        if message_payload is None:
            metrics.increment("kawal_webhook_requests_total", labels={"outcome": "ignored"})
            return {"status": "IGNORED"}
        try:
            accepted = connector.callback(message_payload)
        except ValueError as exc:
            metrics.increment("kawal_webhook_requests_total", labels={"outcome": "invalid"})
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        outcome = "accepted" if accepted else "ignored"
        metrics.increment("kawal_webhook_requests_total", labels={"outcome": outcome})
        return {"status": "ACCEPTED" if accepted else "IGNORED"}

    return app


def create_live_intake_app(connector: OpenWAConnector) -> FastAPI:
    return create_intake_app(
        connector=connector,
        webhook_secret=os.environ["KAWAL_OPENWA_WEBHOOK_SECRET"],
        tenant_id=os.environ["KAWAL_OPENWA_TENANT_ID"],
    )
