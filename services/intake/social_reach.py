"""Social-media collection through Agent Reach (https://github.com/Panniantong/Agent-Reach).

Agent Reach is an external CLI that returns exactly one JSON envelope per
invocation. This module wraps it as a small, injectable client so KAWAL can
search public conversations on X/Twitter, Facebook, and Instagram without
embedding scraping logic of its own.

The integration is intentionally read-only: KAWAL never posts, comments, likes,
or messages through Agent Reach.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import json
import os
import shutil
import subprocess
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

AGENT_REACH_TIMEOUT_SECONDS = 60.0
DEFAULT_SOCIAL_CHANNELS: tuple[str, ...] = ("twitter", "facebook", "instagram")
DEFAULT_SOCIAL_QUERIES: tuple[str, ...] = (
    "banjir bandung",
    "jalan rusak bandung",
    "sampah menumpuk bandung",
)


class AgentReachError(RuntimeError):
    """A structured failure returned by the Agent Reach CLI."""

    def __init__(self, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.code}: {self.message} (hint: {self.hint})"
        return f"{self.code}: {self.message}"


class SocialPost(BaseModel):
    """A normalized public social-media item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: str
    external_id: str
    query: str
    url: str = ""
    author: str = ""
    text: str
    published_at: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SocialCollectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    posts: tuple[SocialPost, ...] = ()
    errors: tuple[str, ...] = ()


class SocialPostStore(Protocol):
    def upsert_many(self, posts: Sequence[SocialPost]) -> int: ...


class InMemorySocialPostStore:
    """Deterministic store for tests and dry runs."""

    def __init__(self) -> None:
        self._posts: dict[tuple[str, str], SocialPost] = {}

    def upsert_many(self, posts: Sequence[SocialPost]) -> int:
        inserted = 0
        for post in posts:
            key = (post.platform, post.external_id)
            if key not in self._posts:
                inserted += 1
            self._posts[key] = post
        return inserted

    @property
    def posts(self) -> tuple[SocialPost, ...]:
        return tuple(self._posts.values())


class PostgresSocialPostStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def upsert_many(self, posts: Sequence[SocialPost]) -> int:
        if not posts:
            return 0
        import psycopg

        rows = 0
        with psycopg.connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                for post in posts:
                    cursor.execute(
                        """
                        INSERT INTO social_media_posts (
                            platform, external_id, query, url, author, text,
                            published_at, raw
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (platform, external_id) DO UPDATE SET
                            query = EXCLUDED.query,
                            url = EXCLUDED.url,
                            author = EXCLUDED.author,
                            text = EXCLUDED.text,
                            published_at = EXCLUDED.published_at,
                            raw = EXCLUDED.raw,
                            fetched_at = clock_timestamp()
                        """,
                        (
                            post.platform,
                            post.external_id,
                            post.query,
                            post.url,
                            post.author,
                            post.text,
                            post.published_at,
                            json.dumps(post.raw, ensure_ascii=False, default=str),
                        ),
                    )
                    rows += 1
            connection.commit()
        return rows


def resolve_agent_reach_binary(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env_value = os.getenv("KAWAL_AGENT_REACH_BIN")
    if env_value:
        return env_value
    found = shutil.which("agent-reach")
    if found:
        return found
    local_bin = Path.home() / ".local" / "bin" / "agent-reach"
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)
    return "agent-reach"


def _first_text(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = value.get("name") or value.get("username") or value.get("screen_name")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _parse_timestamp(value: str) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        if normalized.isdigit():
            return datetime.fromtimestamp(int(normalized), tz=timezone.utc).isoformat()
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def normalize_item(platform: str, query: str, item: Mapping[str, Any]) -> SocialPost:
    text = _first_text(item, ("text", "content", "full_text", "description", "title"))
    author = _first_text(item, ("author", "username", "user", "screen_name", "account"))
    url = _first_text(item, ("url", "link", "permalink", "post_url"))
    external_id = _first_text(item, ("id", "tweet_id", "post_id", "note_id", "video_id"))

    if not external_id:
        if url:
            external_id = url
        else:
            digest = hashlib.sha256(f"{platform}|{author}|{text}".encode("utf-8")).hexdigest()
            external_id = f"sha256:{digest}"

    published_at = _parse_timestamp(
        _first_text(item, ("published_at", "created_at", "timestamp", "date", "time"))
    )

    raw = dict(item)
    return SocialPost(
        platform=platform,
        external_id=external_id[:256],
        query=query,
        url=url,
        author=author[:256],
        text=text or "(tanpa teks)",
        published_at=published_at or None,
        raw=raw,
    )


class AgentReachClient:
    """Thin JSON-envelope client for the ``agent-reach`` CLI."""

    def __init__(
        self,
        binary: str | None = None,
        timeout_seconds: float = AGENT_REACH_TIMEOUT_SECONDS,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.binary = resolve_agent_reach_binary(binary)
        self.timeout_seconds = timeout_seconds
        self._runner = runner or subprocess.run
        self._env = dict(env) if env is not None else None

    def _run(self, args: Sequence[str]) -> dict[str, Any]:
        command = [self.binary, *args]
        try:
            completed = self._runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env=self._env,
            )
        except FileNotFoundError as exc:
            raise AgentReachError(
                "NEED_SETUP",
                "Agent Reach CLI tidak ditemukan",
                "Install dengan: uv tool install https://github.com/Panniantong/Agent-Reach/releases/latest/download/agent-reach.zip",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AgentReachError("TIMEOUT", f"agent-reach melebihi {self.timeout_seconds:.0f}s") from exc

        stdout = (completed.stdout or "").strip()
        if not stdout:
            stderr = (completed.stderr or "").strip()
            raise AgentReachError(
                "BAD_OUTPUT",
                f"agent-reach tidak mengembalikan JSON (exit {completed.returncode})",
                stderr or None,
            )

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            # Agent Reach prints exactly one JSON object; tolerate stray log lines.
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                raise AgentReachError("BAD_OUTPUT", "agent-reach mengembalikan output non-JSON")

        if not isinstance(payload, dict):
            raise AgentReachError("BAD_OUTPUT", "agent-reach mengembalikan JSON non-objek")
        if payload.get("ok") is False:
            error = payload.get("error") or {}
            raise AgentReachError(
                str(error.get("code") or "UPSTREAM_BROKEN"),
                str(error.get("message") or "agent-reach gagal"),
                str(error.get("hint") or "") or None,
            )
        return payload

    def search(self, channel: str, query: str, limit: int = 10) -> tuple[SocialPost, ...]:
        payload = self._run([channel, "search", query, "--limit", str(limit)])
        items = payload.get("items") or []
        if not isinstance(items, list):
            raise AgentReachError("BAD_OUTPUT", "field items bukan list")
        return tuple(
            normalize_item(channel, query, item)
            for item in items
            if isinstance(item, Mapping)
        )

    def collect(
        self,
        queries: Sequence[str],
        channels: Sequence[str] = DEFAULT_SOCIAL_CHANNELS,
        limit: int = 10,
    ) -> SocialCollectionResult:
        """Collect posts for every query/channel pair, isolating per-pair failures."""

        posts: list[SocialPost] = []
        errors: list[str] = []
        seen: set[tuple[str, str]] = set()
        for channel in channels:
            for query in queries:
                try:
                    for post in self.search(channel, query, limit=limit):
                        key = (post.platform, post.external_id)
                        if key in seen:
                            continue
                        seen.add(key)
                        posts.append(post)
                except AgentReachError as exc:
                    errors.append(f"{channel}:{query} -> {exc}")
        return SocialCollectionResult(posts=tuple(posts), errors=tuple(errors))


def parse_channels(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_SOCIAL_CHANNELS
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)


def parse_queries(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_SOCIAL_QUERIES
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)
