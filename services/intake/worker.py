from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from psycopg import Connection

from infra.db import TransactionRunner
from services.intake.assembly import ConversationAssembler
from services.intake.service import AssemblyWorker
from services.outbox.broker import BrokerMessage
from services.outbox.consumer import AtomicInboxConsumer

logger = logging.getLogger("intake-assembly-worker")


class IntakeAssemblyWorker:
    """Consumes intake.messages.v1 broker events and processes due assembly timers."""

    def __init__(
        self,
        transaction_runner: TransactionRunner,
        consumer: AtomicInboxConsumer | None = None,
        assembler: ConversationAssembler | None = None,
    ) -> None:
        self._runner = transaction_runner
        self._consumer = consumer
        self._assembler = assembler or ConversationAssembler()
        self._worker = AssemblyWorker(self._runner, self._assembler)

    def process_due_timers(self, now: datetime | None = None) -> list[str]:
        """Assemble all conversations whose timers are due."""
        current_time = now or datetime.now(timezone.utc)
        assembled_conversations: list[str] = []
        while True:
            conv_id = self._worker.process_due(current_time)
            if conv_id is None:
                break
            assembled_conversations.append(conv_id)
        return assembled_conversations

    def consume_messages_once(self, max_records: int = 10) -> int:
        """Poll and acknowledge intake.messages.v1 events."""
        if self._consumer is None:
            return 0
        return self._consumer.consume_batch(
            topic="intake.messages.v1",
            handler=self._handle_intake_message,
            max_records=max_records,
        )

    def _handle_intake_message(self, connection: Connection, event: BrokerMessage) -> None:
        """Acknowledges intake event into inbox table for deduplication audit."""
        logger.debug("Received intake.messages.v1 event: %s", event.event_id)

    def step(self, now: datetime | None = None, max_records: int = 10) -> tuple[int, list[str]]:
        """Run one step of broker polling followed by due timer assembly."""
        msg_count = self.consume_messages_once(max_records=max_records)
        assembled = self.process_due_timers(now=now)
        return msg_count, assembled
