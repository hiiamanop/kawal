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


def test_case_payload_exposes_analysis_fields(tmp_path) -> None:
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
    assert case["case"]["triage"]["category"] == "ROAD"
    assert case["case"]["triage"]["risk"] in {"LOW", "MEDIUM", "HIGH", "URGENT"}
    assert case["case"]["status"] in {"ACCEPTED", "NEEDS_REVIEW", "REJECTED"}


def test_operator_status_workflow_reaches_forwarded(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    client = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db"))
    assert client.post(
        "/api/monitor/scan",
        json={
            "query": "jalan rusak",
            "limit": 10,
            "source_mode": "replay",
            "analysis_mode": "rule",
            "fixture_dir": str(fixture_dir),
        },
    ).status_code == 200
    case_id = client.get("/api/monitor/cases").json()["cases"][0]["case_id"]
    assert client.patch(
        f"/api/monitor/cases/{case_id}", json={"status": "MONITORING"}
    ).json()["case"]["operator_status"] == "MONITORING"
    assert client.patch(
        f"/api/monitor/cases/{case_id}", json={"status": "READY_TO_FORWARD"}
    ).json()["case"]["operator_status"] == "READY_TO_FORWARD"
    assert client.post(
        f"/api/monitor/cases/{case_id}/forward", json={"note": "Siap"}
    ).status_code == 200
    forwarded = client.patch(
        f"/api/monitor/cases/{case_id}", json={"status": "FORWARDED"}
    )
    assert forwarded.status_code == 200
    assert forwarded.json()["case"]["operator_status"] == "FORWARDED"


def test_console_exposes_full_status_actions() -> None:
    js = Path("services/monitor/static/app.js").read_text()
    for text in ["Mark forwarded", "Abaikan", "Kembali monitor"]:
        assert text in js


def test_forward_in_terminal_state_writes_no_packet(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    client = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db"))
    assert client.post(
        "/api/monitor/scan",
        json={
            "query": "jalan rusak",
            "limit": 10,
            "source_mode": "replay",
            "analysis_mode": "rule",
            "fixture_dir": str(fixture_dir),
        },
    ).status_code == 200
    case_id = client.get("/api/monitor/cases").json()["cases"][0]["case_id"]
    for status in ("MONITORING", "READY_TO_FORWARD", "FORWARDED"):
        assert client.patch(
            f"/api/monitor/cases/{case_id}", json={"status": status}
        ).status_code == 200
    response = client.post(
        f"/api/monitor/cases/{case_id}/forward", json={"note": "Terlambat"}
    )
    assert response.status_code == 409
    forwards = tmp_path / "forwards"
    assert not forwards.exists() or not list(forwards.iterdir())


def test_forward_write_failure_does_not_advance_status(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    client = TestClient(
        create_monitor_app(db_path=tmp_path / "monitor.db"),
        raise_server_exceptions=False,
    )
    assert client.post(
        "/api/monitor/scan",
        json={
            "query": "jalan rusak",
            "limit": 10,
            "source_mode": "replay",
            "analysis_mode": "rule",
            "fixture_dir": str(fixture_dir),
        },
    ).status_code == 200
    case = client.get("/api/monitor/cases").json()["cases"][0]
    (tmp_path / "forwards").write_text("not a directory")
    response = client.post(
        f"/api/monitor/cases/{case['case_id']}/forward", json={"note": "Gagal"}
    )
    assert response.status_code == 500
    saved = client.get(f"/api/monitor/cases/{case['case_id']}").json()["case"]
    assert saved["operator_status"] == "NEW"
    assert client.get(f"/api/monitor/cases/{case['case_id']}").json()["actions"] == []


def test_console_exposes_live_replay_and_analysis_controls(tmp_path) -> None:
    html = TestClient(create_monitor_app(db_path=tmp_path / "monitor.db")).get("/").text
    for text in ["source-mode", "analysis-mode", "fixture-dir", "Replay", "LLM"]:
        assert text in html


def test_detail_panel_exposes_audit_fields() -> None:
    js = Path("services/monitor/static/app.js").read_text()
    for text in ["Evidence spans", "Reason codes", "Trust decision", "Cache", "Queries"]:
        assert text in js
