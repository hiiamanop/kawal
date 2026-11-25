from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

from contracts.harness import SocialPost


class SourceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AgentReachSourceError(SourceError):
    pass


def _first_text(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            for nested_key in ("name", "username", "screen_name", "id"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return ""


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1e11:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                timestamp = float(text)
            except ValueError:
                return None
            if timestamp > 1e11:
                timestamp /= 1000.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return None


def _media_urls(item: Mapping[str, Any]) -> tuple[str, ...]:
    urls: list[str] = []
    media = item.get("media_urls") or item.get("media") or item.get("images")
    if isinstance(media, str):
        urls.append(media)
    elif isinstance(media, Sequence) and not isinstance(media, (str, bytes)):
        for entry in media:
            if isinstance(entry, str):
                urls.append(entry)
            elif isinstance(entry, Mapping):
                url = entry.get("url") or entry.get("media_url") or entry.get("src")
                if isinstance(url, str) and url:
                    urls.append(url)
    return tuple(urls)


def _post_from_item(item: Mapping[str, Any], collected_at: datetime) -> SocialPost | None:
    text = _first_text(item, ("text", "content", "full_text", "body"))
    if not text:
        return None

    url = _first_text(item, ("url", "link", "permalink", "post_url"))
    post_id = _first_text(item, ("id", "tweet_id", "post_id", "note_id"))
    if not post_id:
        post_id = url or hashlib.sha256(text.encode("utf-8")).hexdigest()

    author = _first_text(
        item, ("author", "username", "user", "screen_name", "account", "author_handle")
    )
    posted_at = _parse_datetime(
        item.get("posted_at")
        or item.get("created_at")
        or item.get("timestamp")
        or item.get("date")
    )
    return SocialPost(
        post_id=str(post_id),
        platform="x",
        url=url,
        author_handle=author,
        text=text,
        media_urls=_media_urls(item),
        posted_at=posted_at,
        collected_at=collected_at,
        raw=dict(item),
    )


class ReplayXSource:
    """Read a recorded X post or Agent-Reach response from a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def fetch(self, query: str = "", limit: int = 1) -> list[SocialPost]:
        del query
        if not self._path.is_file():
            raise SourceError("FIXTURE_NOT_FOUND", f"Fixture not found: {self._path}")
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceError("FIXTURE_INVALID", f"Invalid fixture: {self._path}") from exc

        collected_at = datetime.now(timezone.utc)
        if isinstance(payload, Mapping) and isinstance(payload.get("post"), Mapping):
            post = SocialPost.model_validate(payload["post"])
            return [post]

        items = payload.get("items") if isinstance(payload, Mapping) else payload
        if isinstance(items, Mapping):
            items = [items]
        if not isinstance(items, list):
            raise SourceError("FIXTURE_INVALID", "Fixture must contain a post or an items list")

        posts = [
            post
            for item in items
            if isinstance(item, Mapping)
            for post in [_post_from_item(item, collected_at)]
            if post is not None
        ]
        return posts[: max(0, limit)]


class AgentReachXSource:
    """Read X search results through the Agent-Reach CLI.

    The CLI is read-only. This adapter never posts, replies, or sends DMs.
    """

    def __init__(
        self,
        binary: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._binary = (
            binary
            or os.getenv("KAWAL_AGENT_REACH_BIN")
            or shutil.which("agent-reach")
            or "agent-reach"
        )
        self._runner = runner or subprocess.run
        self._timeout_seconds = timeout_seconds

    def fetch(self, query: str, limit: int = 5) -> list[SocialPost]:
        if not query.strip():
            raise AgentReachSourceError("INVALID_QUERY", "Search query must not be empty")
        if limit < 1:
            return []

        command = [
            self._binary,
            "twitter",
            "search",
            query,
            "--limit",
            str(limit),
        ]
        try:
            completed = self._runner(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AgentReachSourceError(
                "NEED_SETUP",
                "Agent-Reach CLI is not installed or not executable",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AgentReachSourceError(
                "TIMEOUT", f"Agent-Reach exceeded {self._timeout_seconds:.0f}s"
            ) from exc

        stdout = (completed.stdout or "").strip()
        payload = self._parse_payload(stdout)
        if isinstance(payload, Mapping) and payload.get("ok") is False:
            error = payload.get("error")
            code = "UPSTREAM_ERROR"
            message = "Agent-Reach returned an error"
            if isinstance(error, Mapping):
                code = str(error.get("code") or code)
                message = str(error.get("message") or message)
            raise AgentReachSourceError(code, message)

        items = payload.get("items") if isinstance(payload, Mapping) else payload
        if isinstance(items, Mapping):
            items = [items]
        if not isinstance(items, list):
            raise AgentReachSourceError("BAD_OUTPUT", "Agent-Reach returned no items list")

        collected_at = datetime.now(timezone.utc)
        posts = [
            post
            for item in items
            if isinstance(item, Mapping)
            for post in [_post_from_item(item, collected_at)]
            if post is not None
        ]
        return posts[:limit]

    @staticmethod
    def _parse_payload(stdout: str) -> Any:
        if not stdout:
            raise AgentReachSourceError("BAD_OUTPUT", "Agent-Reach returned empty output")
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            raise AgentReachSourceError("BAD_OUTPUT", "Agent-Reach returned non-JSON output")
