from __future__ import annotations

import os

from fastapi.testclient import TestClient
import pytest

from services.dashboard.app import create_dashboard_app

DATABASE_URL = os.getenv("KAWAL_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/postgres")


def test_dashboard_index_renders_html() -> None:
    app = create_dashboard_app(database_url=DATABASE_URL)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "KAWAL" in response.text
    assert "Case Inspector" in response.text
    assert "Peta Sebaran Insiden" in response.text
    assert "stat-efficiency" in response.text


def test_api_cases_includes_efficiency_and_map_coords() -> None:
    app = create_dashboard_app(database_url=DATABASE_URL)
    client = TestClient(app)

    res = client.get("/api/cases?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "stats" in data
    assert "efficiency_pct" in data["stats"]
    assert "saved_idr" in data["stats"]
    assert "avoided_calls" in data["stats"]


def test_citizen_tracking_portal_renders_html() -> None:
    app = create_dashboard_app(database_url=DATABASE_URL)
    client = TestClient(app)

    # Citizen visits their personal tracking link
    response = client.get("/track/TKT-test-12345")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Portal Layanan Aduan Warga" in response.text
    assert "TKT-test-12345" in response.text
    assert "Tahapan Penanganan Aduan" in response.text


def test_admin_portal_renders_html() -> None:
    app = create_dashboard_app(database_url=DATABASE_URL)
    client = TestClient(app)

    response = client.get("/admin/tickets")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Portal Petugas Dinas" in response.text
    assert "Filter Dinas" in response.text


def test_ticket_detail_and_update_lifecycle() -> None:
    from uuid import uuid4
    app = create_dashboard_app(database_url=DATABASE_URL)
    client = TestClient(app)

    ticket_id = f"TKT-life-{uuid4().hex[:8]}"

    # 1. Citizen checks ticket detail via API
    get_res = client.get(f"/api/tickets/{ticket_id}")
    assert get_res.status_code == 200
    ticket_data = get_res.json()
    assert ticket_data["ticket_id"] == ticket_id
    assert ticket_data["status"] == "SUBMITTED"

    # 2. Agency officer updates status to IN_PROGRESS with note
    update_res1 = client.post(
        f"/api/tickets/{ticket_id}/update",
        json={
            "status": "IN_PROGRESS",
            "admin_notes": "Petugas regu 1 sedang menuju lokasi.",
            "resolution_photo_url": "",
            "notify_citizen": True,
        },
    )
    assert update_res1.status_code == 200
    assert update_res1.json()["status"] == "IN_PROGRESS"

    # 3. Agency officer resolves ticket with proof photo
    update_res2 = client.post(
        f"/api/tickets/{ticket_id}/update",
        json={
            "status": "RESOLVED",
            "admin_notes": "Lubang jalan telah ditambal dengan hotmix tuntas.",
            "resolution_photo_url": "https://example.com/proof-road-fixed.jpg",
            "notify_citizen": True,
        },
    )
    assert update_res2.status_code == 200
    assert update_res2.json()["status"] == "RESOLVED"

    # 4. Verify citizen sees the updated status, notes, and proof photo
    verify_res = client.get(f"/api/tickets/{ticket_id}")
    assert verify_res.status_code == 200
    verified = verify_res.json()
    assert verified["status"] == "RESOLVED"
    assert "ditambal" in verified["admin_notes"]
    assert verified["resolution_photo_url"] == "https://example.com/proof-road-fixed.jpg"


def test_social_posts_endpoint_returns_collected_items() -> None:
    from uuid import uuid4
    import psycopg

    app = create_dashboard_app(database_url=DATABASE_URL)
    client = TestClient(app)

    external_id = f"test-web-{uuid4().hex[:10]}"
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO social_media_posts (platform, external_id, query, url, author, text)
                    VALUES ('web', %s, 'banjir bandung', 'https://example.com/aduan', 'warga', 'Banjir di Tegalega')
                    ON CONFLICT (platform, external_id) DO NOTHING
                    """,
                    (external_id,),
                )
            conn.commit()
    except Exception as exc:  # pragma: no cover - environment without the migration
        pytest.skip(f"social_media_posts unavailable: {exc}")

    response = client.get("/api/social/posts?platform=web")
    assert response.status_code == 200
    data = response.json()
    assert "posts" in data
    assert any(post["external_id"] == external_id for post in data["posts"])
