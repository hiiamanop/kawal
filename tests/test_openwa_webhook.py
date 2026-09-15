from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from services.intake.api import create_intake_app
from services.intake.openwa import OpenWAConnector


class IntakeSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object, object, object]] = []

    def accept(self, message: object, quoted: object, case_key: object, connector_id: object, account_id: object) -> bool:
        self.calls.append((message, quoted, case_key, connector_id, account_id))
        return True


def _client() -> tuple[TestClient, IntakeSpy]:
    spy = IntakeSpy()
    connector = OpenWAConnector(intake_service=spy, connector_id="openwa", account_id="628000@c.us")
    return TestClient(create_intake_app(connector, webhook_secret="test-secret", tenant_id="tenant-live")), spy


def test_health_and_metrics_endpoints_are_available() -> None:
    client, _ = _client()
    assert client.get("/healthz").json() == {"status": "healthy"}
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert "text/plain" in metrics_response.headers["content-type"]


def test_openwa_webhook_rejects_missing_or_invalid_secret() -> None:
    client, _ = _client()
    payload = {"event": "message.received", "data": {}}
    assert client.post("/v1/webhooks/openwa", json=payload).status_code == 401
    assert client.post("/v1/webhooks/openwa", json=payload, headers={"X-KAWAL-Webhook-Secret": "wrong"}).status_code == 401
    assert 'kawal_webhook_requests_total{outcome="unauthorized"}' in client.get("/metrics").text


def test_openwa_webhook_accepts_incoming_direct_message() -> None:
    client, spy = _client()
    response = client.post(
        "/v1/webhooks/openwa",
        headers={"X-KAWAL-Webhook-Secret": "test-secret"},
        json={
            "event": "message.received",
            "data": {
                "id": "false_628123@c.us_ABC",
                "chatId": "628123@c.us",
                "body": "Lapor jalan berlubang di Jl. Merdeka.",
                "timestamp": "2026-09-14T14:00:00Z",
                "fromMe": False,
            },
        },
    )
    assert response.status_code == 202
    assert response.json() == {"status": "ACCEPTED"}
    assert len(spy.calls) == 1
    message, quoted, case_key, connector_id, account_id = spy.calls[0]
    assert message.tenant_id == "tenant-live"
    assert message.conversation_id == "628123@c.us"
    assert message.source_message_id == "false_628123@c.us_ABC"
    assert connector_id == "openwa"
    assert account_id == "628000@c.us"


def test_openwa_webhook_ignores_self_sent_receipt_and_non_message_event() -> None:
    client, spy = _client()
    headers = {"X-KAWAL-Webhook-Secret": "test-secret"}
    receipt = client.post("/v1/webhooks/openwa", headers=headers, json={"event": "message.ack", "data": {"id": "x"}})
    assert receipt.status_code == 202
    assert receipt.json() == {"status": "IGNORED"}
    assert not spy.calls

    self_sent = client.post(
        "/v1/webhooks/openwa",
        headers=headers,
        json={
            "event": "message.received",
            "data": {"id": "true_628000@c.us_X", "chatId": "628000@c.us", "body": "outbound", "timestamp": "2026-09-14T14:00:00Z", "fromMe": True},
        },
    )
    assert self_sent.status_code == 202
    assert self_sent.json() == {"status": "IGNORED"}
    assert not spy.calls
