from __future__ import annotations

import os

from fastapi.testclient import TestClient
import pytest

from services.dashboard.app import create_dashboard_app

DATABASE_URL = os.getenv("KAWAL_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres")


@pytest.mark.integration
def test_dashboard_index_renders_html() -> None:
    app = create_dashboard_app(database_url=DATABASE_URL)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "KAWAL" in response.text
    assert "Live Case Inspector" in response.text


@pytest.mark.integration
def test_dashboard_api_cases_list_and_stats() -> None:
    app = create_dashboard_app(database_url=DATABASE_URL)
    client = TestClient(app)

    response = client.get("/api/cases?limit=10")
    assert response.status_code == 200
    data = response.json()

    assert "cases" in data
    assert "stats" in data
    assert isinstance(data["cases"], list)
    assert isinstance(data["stats"], dict)
    assert "total" in data["stats"]
    assert "ticketed" in data["stats"]

    if data["cases"]:
        first_case_id = data["cases"][0]["case_id"]
        detail_res = client.get(f"/api/cases/{first_case_id}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["case_id"] == first_case_id
        assert "snapshot" in detail_data
        assert "audit_traces" in detail_data
