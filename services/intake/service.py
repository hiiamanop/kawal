from datetime import datetime
from time import perf_counter

from contracts.models import RawMessage
from infra.db import TransactionRunner
from services.intake.assembly import AssemblyClaim, ConversationAssembler


TARGET_PERSIST_LATENCY_MS = 500


class IntakeService:
    def __init__(self, transaction_runner: TransactionRunner, assembler: ConversationAssembler) -> None:
        self._transaction_runner = transaction_runner
        self._assembler = assembler

    def accept(
        self,
        message: RawMessage,
        quoted_source_message_id: str | None = None,
        case_key: str | None = None,
        connector_id: str = "replay",
        account_id: str = "research",
    ) -> bool:
        return self._transaction_runner.run(
            lambda connection: self._assembler.ingest(
                connection, message, quoted_source_message_id, case_key, connector_id, account_id
            )
        )

    def accept_with_latency(
        self,
        message: RawMessage,
        quoted_source_message_id: str | None = None,
        case_key: str | None = None,
        connector_id: str = "replay",
        account_id: str = "research",
    ) -> tuple[bool, float]:
        started = perf_counter()
        accepted = self.accept(
            message, quoted_source_message_id, case_key, connector_id, account_id
        )
        return accepted, (perf_counter() - started) * 1000


class AssemblyWorker:
    def __init__(self, transaction_runner: TransactionRunner, assembler: ConversationAssembler) -> None:
        self._transaction_runner = transaction_runner
        self._assembler = assembler

    def claim(self, now: datetime) -> AssemblyClaim | None:
        return self._transaction_runner.run(
            lambda connection: self._assembler.claim(connection, now)
        )

    def assemble(self, claim: AssemblyClaim, now: datetime) -> tuple[datetime | None, str | None]:
        return self._transaction_runner.run(
            lambda connection: self._assembler.assemble(connection, claim, now)
        )

    def finalize(
        self,
        claim: AssemblyClaim,
        now: datetime,
        last_received_at: datetime | None = None,
        last_message_id: str | None = None,
    ) -> bool:
        return self._transaction_runner.run(
            lambda connection: self._assembler.finalize(
                connection, claim, now, last_received_at, last_message_id
            )
        )

    def process_due(self, now: datetime) -> str | None:
        claim = self.claim(now)
        if claim is None:
            return None
        last_received_at, last_message_id = self.assemble(claim, now)
        self.finalize(claim, now, last_received_at, last_message_id)
        return claim.conversation_id
