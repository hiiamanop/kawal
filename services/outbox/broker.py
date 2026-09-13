from __future__ import annotations

import collections
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BrokerMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    topic: str
    partition_key: str
    payload: dict[str, Any]
    headers: dict[str, str] = Field(default_factory=dict)
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    offset: int = 0


class EventBroker(Protocol):
    def publish(
        self,
        topic: str,
        key: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        event_id: str | None = None,
    ) -> BrokerMessage: ...

    def poll(
        self,
        topic: str,
        group_id: str,
        max_records: int = 10,
    ) -> list[BrokerMessage]: ...

    def commit_offset(
        self,
        topic: str,
        group_id: str,
        offset: int,
    ) -> None: ...

    def message_count(self, topic: str) -> int: ...


class InMemoryEventBroker:
    """Thread-safe in-memory event broker implementing Kafka/Redpanda semantics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._topics: dict[str, list[BrokerMessage]] = collections.defaultdict(list)
        self._offsets: dict[tuple[str, str], int] = collections.defaultdict(int)

    def publish(
        self,
        topic: str,
        key: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        event_id: str | None = None,
    ) -> BrokerMessage:
        with self._lock:
            eid = event_id or str(uuid4())
            current_offset = len(self._topics[topic])
            msg = BrokerMessage(
                event_id=eid,
                topic=topic,
                partition_key=key,
                payload=payload,
                headers=headers or {},
                offset=current_offset,
            )
            self._topics[topic].append(msg)
            return msg

    def poll(
        self,
        topic: str,
        group_id: str,
        max_records: int = 10,
    ) -> list[BrokerMessage]:
        with self._lock:
            current_offset = self._offsets[(topic, group_id)]
            available = self._topics[topic][current_offset : current_offset + max_records]
            return list(available)

    def commit_offset(
        self,
        topic: str,
        group_id: str,
        offset: int,
    ) -> None:
        with self._lock:
            # Advance committed offset to offset + 1 (next message to read)
            self._offsets[(topic, group_id)] = max(
                self._offsets[(topic, group_id)], offset + 1
            )

    def message_count(self, topic: str) -> int:
        with self._lock:
            return len(self._topics[topic])

    def clear(self) -> None:
        with self._lock:
            self._topics.clear()
            self._offsets.clear()
