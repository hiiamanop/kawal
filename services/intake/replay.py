from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import json

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
)
from services.core.orchestrator import build_ticket_command
from services.simulator.store import TicketSimulator


def load_fixture(path: Path) -> ReplayFixture:
    return ReplayFixture.model_validate_json(path.read_text(encoding="utf-8"))


def replay_to_ticket(fixture: ReplayFixture, simulator: TicketSimulator) -> TicketReceipt:
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
    evidence_hash = sha256(
        "\n".join(message.text for message in messages).encode("utf-8")
    ).hexdigest()
    snapshot = CaseSnapshot(
        case_id="case-001",
        tenant_id=fixture.tenant_id,
        conversation_id=fixture.conversation_id,
        revision=1,
        state=ProcessingState.READY,
        messages=messages,
        evidence_hash=evidence_hash,
        created_at=started_at + timedelta(seconds=max(b.offset_seconds for b in fixture.bubbles)),
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
    command = build_ticket_command(snapshot, analysis)
    if command is None:
        raise RuntimeError("M1 fixture must produce an allowed ticket command")
    return simulator.create_ticket(command.request, command.idempotency_key)


def main() -> None:
    fixture = load_fixture(Path("tests/fixtures/three_bubble_road_complaint.json"))
    receipt = replay_to_ticket(fixture, TicketSimulator())
    print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True))


if __name__ == "__main__":
    main()
