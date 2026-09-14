from __future__ import annotations

import collections
from datetime import datetime, timezone
import json
from threading import Lock
from typing import Any, Protocol, Sequence
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
    partition: int = 0
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


class RedpandaBrokerUnavailableError(RuntimeError):
    pass


class RedpandaEventBroker:
    """Kafka-compatible client for a live Redpanda broker.

    This implementation requires the optional ``kafka-python-ng`` dependency and
    explicitly raises RedpandaBrokerUnavailableError if the broker is unavailable.
    It never falls back silently to an in-memory broker in a live deployment.
    """

    def __init__(
        self,
        bootstrap_servers: str = "127.0.0.1:19092",
        client_id: str = "kawal-outbox",
        request_timeout_ms: int = 10_000,
    ) -> None:
        try:
            from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
            from kafka.admin import NewTopic
            from kafka.errors import KafkaError, NoBrokersAvailable
        except ImportError as exc:
            raise RedpandaBrokerUnavailableError(
                "kafka-python-ng is required for RedpandaEventBroker; install requirements.txt"
            ) from exc

        self._bootstrap_servers = bootstrap_servers
        self._client_id = client_id
        self._request_timeout_ms = request_timeout_ms
        self._KafkaConsumer = KafkaConsumer
        self._KafkaProducer = KafkaProducer
        self._KafkaAdminClient = KafkaAdminClient
        self._NewTopic = NewTopic
        self._KafkaError = KafkaError
        self._NoBrokersAvailable = NoBrokersAvailable
        self._producer: Any | None = None
        self._consumers: dict[tuple[str, str], Any] = {}

    def _get_producer(self) -> Any:
        if self._producer is not None:
            return self._producer
        try:
            self._producer = self._KafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                client_id=self._client_id,
                acks="all",
                retries=3,
                value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode("utf-8"),
                key_serializer=lambda key: key.encode("utf-8"),
                request_timeout_ms=self._request_timeout_ms,
            )
            return self._producer
        except self._NoBrokersAvailable as exc:
            raise RedpandaBrokerUnavailableError(
                f"Redpanda broker unavailable at {self._bootstrap_servers}"
            ) from exc

    def ensure_topics(self, topics: Sequence[str], partitions: int = 1, replication_factor: int = 1) -> None:
        try:
            admin = self._KafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                client_id=self._client_id,
                request_timeout_ms=self._request_timeout_ms,
            )
            existing = set(admin.list_topics())
            new_topics = [
                self._NewTopic(name=topic, num_partitions=partitions, replication_factor=replication_factor)
                for topic in topics if topic not in existing
            ]
            if new_topics:
                admin.create_topics(new_topics=new_topics, validate_only=False)
            admin.close()
        except self._NoBrokersAvailable as exc:
            raise RedpandaBrokerUnavailableError(
                f"Redpanda broker unavailable at {self._bootstrap_servers}"
            ) from exc

    def publish(
        self,
        topic: str,
        key: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        event_id: str | None = None,
    ) -> BrokerMessage:
        producer = self._get_producer()
        eid = event_id or str(uuid4())
        encoded_headers = [(name, value.encode("utf-8")) for name, value in (headers or {}).items()]
        if not any(name == "event_id" for name, _ in encoded_headers):
            encoded_headers.append(("event_id", eid.encode("utf-8")))
        try:
            metadata = producer.send(topic, key=key, value=payload, headers=encoded_headers).get(
                timeout=self._request_timeout_ms / 1000
            )
        except self._KafkaError as exc:
            raise RedpandaBrokerUnavailableError(f"Could not publish {eid} to {topic}: {exc}") from exc
        return BrokerMessage(
            event_id=eid,
            topic=topic,
            partition_key=key,
            payload=payload,
            headers=headers or {},
            partition=metadata.partition,
            offset=metadata.offset,
        )

    def _get_consumer(self, topic: str, group_id: str) -> Any:
        cache_key = (topic, group_id)
        if cache_key not in self._consumers:
            try:
                self._consumers[cache_key] = self._KafkaConsumer(
                    topic,
                    bootstrap_servers=self._bootstrap_servers,
                    group_id=group_id,
                    client_id=f"{self._client_id}-{group_id}",
                    enable_auto_commit=False,
                    auto_offset_reset="earliest",
                    value_deserializer=lambda value: json.loads(value.decode("utf-8")),
                    key_deserializer=lambda key: key.decode("utf-8") if key else "",
                    consumer_timeout_ms=1000,
                )
            except self._NoBrokersAvailable as exc:
                raise RedpandaBrokerUnavailableError(
                    f"Redpanda broker unavailable at {self._bootstrap_servers}"
                ) from exc
        return self._consumers[cache_key]

    def poll(self, topic: str, group_id: str, max_records: int = 10) -> list[BrokerMessage]:
        consumer = self._get_consumer(topic, group_id)
        raw_records = consumer.poll(timeout_ms=1000, max_records=max_records)
        messages: list[BrokerMessage] = []
        for records in raw_records.values():
            for record in records:
                decoded_headers = {
                    key: value.decode("utf-8")
                    for key, value in record.headers
                    if value is not None
                }
                messages.append(BrokerMessage(
                    event_id=decoded_headers.get("event_id", str(uuid4())),
                    topic=record.topic,
                    partition_key=record.key or "",
                    payload=record.value,
                    headers=decoded_headers,
                    partition=record.partition,
                    offset=record.offset,
                ))
        return messages

    def commit_offset(self, topic: str, group_id: str, offset: int) -> None:
        consumer = self._get_consumer(topic, group_id)
        try:
            # This consumer is intentionally polled and committed synchronously only after
            # the database transaction succeeds in AtomicInboxConsumer.
            consumer.commit()
        except self._KafkaError as exc:
            raise RedpandaBrokerUnavailableError(
                f"Could not commit consumer offset for {topic}/{group_id}: {exc}"
            ) from exc

    def message_count(self, topic: str) -> int:
        # Kafka does not expose a cheap portable topic count; monitoring should use consumer lag.
        return -1

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush(timeout=self._request_timeout_ms / 1000)
            self._producer.close()
            self._producer = None
        for consumer in self._consumers.values():
            consumer.close()
        self._consumers.clear()


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
