from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from services.intake.api import create_intake_app
from services.intake.openwa import OpenWAConnector
from services.intelligence.geocoding import GeocodingResult, ReverseGeocoder


class SpyIntake:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object, object, object]] = []

    def accept(self, message: object, quoted: object, case_key: object, connector_id: object, account_id: object) -> bool:
        self.calls.append((message, quoted, case_key, connector_id, account_id))
        return True


def test_reverse_geocoder_centroid_fallback() -> None:
    geocoder = ReverseGeocoder(timeout_seconds=0.1)  # Force quick offline fallback
    # Coordinates in Bojongloa Kaler / Kopo (-6.9389, 107.5889)
    res = geocoder.reverse_geocode(-6.9389, 107.5889)

    assert isinstance(res, GeocodingResult)
    assert res.district == "Bojongloa Kaler"
    assert "Kota Bandung" in res.formatted_address
    assert res.latitude == -6.9389


def test_reverse_geocoder_uses_provided_address_hint() -> None:
    geocoder = ReverseGeocoder()
    hint = "Jl. Asia Afrika No. 100, Kelurahan Braga, Sumur Bandung"
    res = geocoder.reverse_geocode(-6.9175, 107.6111, address_hint=hint)

    assert res.formatted_address == hint
    assert res.is_fallback is False


def test_openwa_webhook_pin_location_enrichment() -> None:
    spy = SpyIntake()
    connector = OpenWAConnector(intake_service=spy, connector_id="openwa", account_id="628000@c.us")
    app = create_intake_app(connector=connector, webhook_secret="test-secret", tenant_id="tenant-bdg")
    client = TestClient(app)

    # Citizen shares location pin without text
    response = client.post(
        "/v1/webhooks/openwa",
        headers={"X-KAWAL-Webhook-Secret": "test-secret"},
        json={
            "event": "message.received",
            "data": {
                "id": "false_628123@c.us_LOC_1",
                "chatId": "628123@c.us",
                "body": None,
                "type": "location",
                "latitude": -6.9556,
                "longitude": 107.6333,
                "timestamp": "2026-09-14T14:00:00Z",
                "fromMe": False,
            },
        },
    )

    assert response.status_code == 202
    assert response.json() == {"status": "ACCEPTED"}
    assert len(spy.calls) == 1
    msg = spy.calls[0][0]
    # Message text automatically populated with pinned coordinates and address
    assert "[Pin Lokasi]:" in msg.text
    assert "Kota Bandung" in msg.text
