from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid5

from contracts.models import (
    AnalysisResult,
    CaseSnapshot,
    Category,
    ProcessingState,
    RawMessage,
    ReplayFixture,
    RiskLevel,
    Sensitivity,
    TicketReceipt,
    PolicyInput,
)
from infra.db import TransactionRunner
from services.core.orchestrator import build_ticket_command
from services.core.persistence import M1Store
from services.core.policy import evaluate_ticket_creation
from services.outbox.relay import OutboxRelay
from services.simulator.store import TicketSimulator


EVENT_NAMESPACE = UUID("00000000-0000-0000-0000-000000000001")


def load_fixture(path: Path) -> ReplayFixture:
    return ReplayFixture.model_validate_json(path.read_text(encoding="utf-8"))


def build_replay_case(
    fixture: ReplayFixture,
) -> tuple[tuple[RawMessage, ...], CaseSnapshot, AnalysisResult]:
    started_at = datetime(2026, 9, 11, tzinfo=timezone.utc)
    messages = tuple(
        RawMessage(
            message_id=f"{fixture.tenant_id}:{bubble.source_message_id}",
            tenant_id=fixture.tenant_id,
            conversation_id=fixture.conversation_id,
            source_message_id=bubble.source_message_id,
            text=bubble.text,
            received_at=started_at + timedelta(seconds=bubble.offset_seconds),
        )
        for bubble in fixture.bubbles
    )
    evidence_hash = sha256("\n".join(message.text for message in messages).encode("utf-8")).hexdigest()
    snapshot = CaseSnapshot(
        case_id=f"case-{fixture.scenario_id}",
        tenant_id=fixture.tenant_id,
        conversation_id=fixture.conversation_id,
        revision=1,
        state=ProcessingState.READY,
        messages=messages,
        evidence_hash=evidence_hash,
        created_at=started_at + timedelta(seconds=max(bubble.offset_seconds for bubble in fixture.bubbles)),
    )
    analysis = AnalysisResult(
        category=Category.ROAD,
        risk=RiskLevel.MEDIUM,
        sensitivity=Sensitivity.NORMAL,
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-BINA-MARGA-01",
        missing_fields=(),
        has_mandatory_evidence=True,
    )
    return messages, snapshot, analysis


def replay_to_ticket(fixture: ReplayFixture, simulator: TicketSimulator) -> TicketReceipt:
    _, snapshot, analysis = build_replay_case(fixture)
    command = build_ticket_command(snapshot, analysis)
    if command is None:
        raise RuntimeError("M1 fixture must produce an allowed ticket command")
    return simulator.create_ticket(command.request, command.idempotency_key)


def replay_to_durable_ticket(
    fixture: ReplayFixture,
    transaction_runner: TransactionRunner,
    store: M1Store,
    relay: OutboxRelay,
) -> TicketReceipt:
    messages, snapshot, analysis = build_replay_case(fixture)
    command = build_ticket_command(snapshot, analysis)
    if command is None:
        raise RuntimeError("M1 fixture must produce an allowed ticket command")
    policy = evaluate_ticket_creation(
        PolicyInput(
            tenant_id=snapshot.tenant_id,
            case_id=snapshot.case_id,
            revision=snapshot.revision,
            category=analysis.category,
            jurisdiction_id=analysis.jurisdiction_id,
            authority_unit_id=analysis.authority_unit_id,
            missing_fields=analysis.missing_fields,
            has_mandatory_evidence=analysis.has_mandatory_evidence,
            idempotency_key=command.idempotency_key,
        )
    )
    event_id = uuid5(EVENT_NAMESPACE, f"{fixture.tenant_id}:{fixture.scenario_id}")
    transaction_runner.run(
        lambda connection: store.persist_case(
            connection,
            event_id=event_id,
            snapshot=snapshot,
            messages=messages,
            command=command,
            policy=policy,
        )
    )
    receipt = relay.dispatch_next()
    if receipt is not None:
        return receipt
    operation = relay.lookup_operation(command.idempotency_key)
    if operation is None:
        raise RuntimeError("M1 replay did not enqueue a ticket command")
    return operation
