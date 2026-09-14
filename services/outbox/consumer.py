from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

from psycopg import Connection

from infra.db import TransactionRunner
from services.outbox.broker import BrokerMessage, EventBroker


class AtomicInboxConsumer:
    """PRD §32 idempotent consumer using transactional inbox deduplication."""

    def __init__(
        self,
        consumer_name: str,
        transaction_runner: TransactionRunner,
        broker: EventBroker,
    ) -> None:
        self._consumer_name = consumer_name
        self._transaction_runner = transaction_runner
        self._broker = broker

    @property
    def consumer_name(self) -> str:
        return self._consumer_name

    def process_message(
        self,
        message: BrokerMessage,
        handler: Callable[[Connection, BrokerMessage], Any],
    ) -> bool:
        """Process an incoming broker message with atomic inbox deduplication.

        Returns True if newly processed, False if duplicate skipped.
        """
        try:
            event_uuid = UUID(message.event_id)
        except ValueError:
            import hashlib
            event_uuid = UUID(hashlib.md5(message.event_id.encode("utf-8")).hexdigest())

        def _atomic_step(connection: Connection) -> bool:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO inbox (consumer_name, event_id, aggregate_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (consumer_name, event_id) DO NOTHING
                    RETURNING event_id
                    """,
                    (self._consumer_name, event_uuid, message.partition_key),
                )
                if cursor.fetchone() is None:
                    # Duplicate event detected, skip state mutation safely
                    return False

                # Execute state mutation inside the exact same database transaction
                handler(connection, message)
                return True

        processed = self._transaction_runner.run(_atomic_step)

        # Commit offset after successful database transaction
        self._broker.commit_offset(
            topic=message.topic,
            group_id=self._consumer_name,
            offset=message.offset,
        )

        return processed

    def consume_batch(
        self,
        topic: str,
        handler: Callable[[Connection, BrokerMessage], Any],
        max_records: int = 10,
        on_error: Callable[[BrokerMessage, Exception], bool] | None = None,
    ) -> int:
        """Poll and process a batch; errors are re-raised unless explicitly handled."""
        messages = self._broker.poll(
            topic=topic,
            group_id=self._consumer_name,
            max_records=max_records,
        )
        processed_count = 0
        for msg in messages:
            try:
                if self.process_message(msg, handler):
                    processed_count += 1
            except Exception as exc:
                if on_error is None or not on_error(msg, exc):
                    raise
        return processed_count
