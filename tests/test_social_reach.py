from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from services.intake.social_reach import (
    AgentReachClient,
    AgentReachError,
    InMemorySocialPostStore,
    normalize_item,
    parse_channels,
    parse_queries,
)


def _completed(payload: dict[str, Any], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["agent-reach"],
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr="",
    )


def test_agent_reach_search_parses_envelope_and_normalizes_items() -> None:
    captured: list[list[str]] = []

    def runner(command, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(list(command))
        return _completed(
            {
                "schema_version": 1,
                "ok": True,
                "channel": "twitter",
                "action": "search",
                "backend": "opencli",
                "items": [
                    {
                        "id": "12345",
                        "username": "warga_bdg",
                        "text": "Banjir lagi di Tegalega, tolong tangani!",
                        "url": "https://x.com/warga_bdg/status/12345",
                        "published_at": "2026-11-10T08:15:00.000Z",
                    }
                ],
                "error": None,
            }
        )

    client = AgentReachClient(binary="/usr/bin/agent-reach", runner=runner)
    posts = client.search("twitter", "banjir bandung", limit=3)

    assert captured[0] == [
        "/usr/bin/agent-reach",
        "twitter",
        "search",
        "banjir bandung",
        "--limit",
        "3",
    ]
    assert len(posts) == 1
    post = posts[0]
    assert post.platform == "twitter"
    assert post.external_id == "12345"
    assert post.author == "warga_bdg"
    assert post.published_at == "2026-11-10T08:15:00+00:00"
    assert "Tegalega" in post.text


def test_agent_reach_error_envelope_is_structured() -> None:
    def runner(command, **kwargs):  # type: ignore[no-untyped-def]
        return _completed(
            {
                "schema_version": 1,
                "ok": False,
                "channel": "facebook",
                "action": "search",
                "error": {
                    "code": "NEED_LOGIN",
                    "message": "Login required",
                    "hint": "agent-reach login facebook",
                },
            }
        )

    client = AgentReachClient(binary="agent-reach", runner=runner)
    with pytest.raises(AgentReachError) as exc_info:
        client.search("facebook", "banjir")

    assert exc_info.value.code == "NEED_LOGIN"
    assert "agent-reach login facebook" in str(exc_info.value)


def test_agent_reach_missing_binary_is_reported_as_setup_problem() -> None:
    def runner(command, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError(command[0])

    client = AgentReachClient(binary="agent-reach-missing", runner=runner)
    with pytest.raises(AgentReachError) as exc_info:
        client.search("twitter", "banjir")

    assert exc_info.value.code == "NEED_SETUP"
    assert "uv tool install" in (exc_info.value.hint or "")


def test_agent_reach_tolerates_stray_log_lines() -> None:
    def runner(command, **kwargs):  # type: ignore[no-untyped-def]
        payload = {
            "schema_version": 1,
            "ok": True,
            "items": [{"id": "1", "text": "tes"}],
            "error": None,
        }
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="INFO starting\n" + json.dumps(payload) + "\n",
            stderr="",
        )

    client = AgentReachClient(binary="agent-reach", runner=runner)
    posts = client.search("instagram", "sampah")
    assert posts[0].external_id == "1"


def test_agent_reach_collect_isolates_failures_and_deduplicates() -> None:
    def runner(command, **kwargs):  # type: ignore[no-untyped-def]
        channel = command[1]
        if channel == "facebook":
            return _completed(
                {
                    "ok": False,
                    "error": {"code": "NEED_LOGIN", "message": "Login required"},
                }
            )
        return _completed(
            {
                "ok": True,
                "items": [
                    {"id": "same-id", "text": "Banjir di Kordon"},
                    {"id": "same-id", "text": "Banjir di Kordon"},
                ],
                "error": None,
            }
        )

    client = AgentReachClient(binary="agent-reach", runner=runner)
    result = client.collect(("banjir bandung",), channels=("twitter", "facebook"))

    assert len(result.posts) == 1
    assert result.posts[0].platform == "twitter"
    assert len(result.errors) == 1
    assert "facebook:banjir bandung" in result.errors[0]
    assert "NEED_LOGIN" in result.errors[0]


def test_social_post_store_deduplicates_by_platform_and_external_id() -> None:
    post = normalize_item(
        "twitter",
        "banjir bandung",
        {"id": "42", "text": "Genangan di Jalan Peta", "username": "warga"},
    )
    store = InMemorySocialPostStore()

    assert store.upsert_many([post]) == 1
    assert store.upsert_many([post]) == 0
    assert len(store.posts) == 1


def test_normalize_item_generates_stable_external_id_without_source_id() -> None:
    item = {"text": "Sampah menumpuk di Gasibu", "author": "warga"}
    first = normalize_item("facebook", "sampah bandung", item)
    second = normalize_item("facebook", "sampah bandung", dict(item))

    assert first.external_id.startswith("sha256:")
    assert first.external_id == second.external_id


def test_parse_channels_and_queries() -> None:
    assert parse_channels("twitter, facebook,,") == ("twitter", "facebook")
    assert parse_queries(["banjir bandung"]) == ("banjir bandung",)
