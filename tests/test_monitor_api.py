from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from services.monitor.app import create_monitor_app

from tests.test_monitor_helpers import _write_fixture


def test_index_renders_console_shell(tmp_path) -> None:
    client = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db"))
    response = client.get("/")
    assert response.status_code == 200
    assert "KAWAL Monitor" in response.text
    assert "Keluhan viral" in response.text
    assert "Scan sekarang" in response.text


def test_legacy_tracking_route_is_gone(tmp_path) -> None:
    client = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db"))
    assert client.get("/track/TKT-1").status_code == 404
    assert client.get("/api/tickets").status_code == 404


def test_scan_then_update_then_forward(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    client = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db"))
    scan = client.post(
        "/api/monitor/scan",
        json={
            "query": "jalan rusak",
            "limit": 10,
            "source_mode": "replay",
            "analysis_mode": "rule",
            "fixture_dir": str(fixture_dir),
        },
    )
    assert scan.status_code == 200
    case = client.get("/api/monitor/cases").json()["cases"][0]
    update = client.patch(
        f"/api/monitor/cases/{case['case_id']}",
        json={"status": "MONITORING", "note": "Pantau"},
    )
    assert update.status_code == 200
    forward = client.post(
        f"/api/monitor/cases/{case['case_id']}/forward",
        json={"note": "Teruskan"},
    )
    assert forward.status_code == 200
    packet = forward.json()
    markdown_path = Path(packet["markdown_path"])
    assert "DRAFT" in markdown_path.read_text()
    download = client.get(f"/api/monitor/forwards/{markdown_path.name}")
    assert download.status_code == 200
    assert "DRAFT" in download.text


def test_console_has_feed_detail_and_composer(tmp_path) -> None:
    html = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db")).get("/").text
    for text in ["Keluhan viral", "Tanya atau scan keluhan", "Detail", "Siap Diteruskan", "Monitoring"]:
        assert text in html
    assert "cdn.tailwindcss.com" not in html
