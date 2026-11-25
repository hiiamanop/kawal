from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from services.harness.sources import (
    AgentReachSourceError,
    AgentReachXSource,
    ReplayXSource,
    SourceError,
)


FIXTURE = Path(__file__).parent / "fixtures" / "x_posts" / "synthetic_road_bandung_001.json"


def test_replay_source_reads_fixture() -> None:
    posts = ReplayXSource(FIXTURE).fetch()
    assert len(posts) == 1
    assert posts[0].platform == "x"
    assert posts[0].post_id == "synthetic-road-bandung-001"
    assert "Jl. Kopo" in posts[0].text
    assert posts[0].collected_at is not None


def test_replay_source_reports_missing_fixture(tmp_path: Path) -> None:
    with pytest.raises(SourceError) as exc:
        ReplayXSource(tmp_path / "missing.json").fetch()
    assert exc.value.code == "FIXTURE_NOT_FOUND"


def test_agent_reach_source_parses_items() -> None:
    payload = {
        "ok": True,
        "items": [
            {
                "id": "123",
                "text": "Jalan rusak di Jl. Kopo Bandung",
                "url": "https://x.com/warga/status/123",
                "author": "warga",
                "created_at": "2026-11-20T08:00:00Z",
            }
        ],
    }

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    source = AgentReachXSource(binary="agent-reach", runner=runner)
    posts = source.fetch("jalan rusak bandung", limit=5)
    assert len(posts) == 1
    assert posts[0].post_id == "123"
    assert posts[0].author_handle == "warga"
    assert posts[0].posted_at is not None


def test_agent_reach_source_reports_need_setup() -> None:
    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("agent-reach")

    source = AgentReachXSource(binary="agent-reach", runner=runner)
    with pytest.raises(AgentReachSourceError) as exc:
        source.fetch("jalan rusak bandung")
    assert exc.value.code == "NEED_SETUP"


def test_agent_reach_source_reports_upstream_error() -> None:
    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout=json.dumps(
                {"ok": False, "error": {"code": "NEED_LOGIN", "message": "login X"}}
            ),
            stderr="",
        )

    source = AgentReachXSource(binary="agent-reach", runner=runner)
    with pytest.raises(AgentReachSourceError) as exc:
        source.fetch("jalan rusak bandung")
    assert exc.value.code == "NEED_LOGIN"
