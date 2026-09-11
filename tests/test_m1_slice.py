from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from fastapi.testclient import TestClient

from contracts.models import (
    AnalysisResult,
    CaseSnapshot,
    Category,
    PolicyInput,
    ProcessingState,
    RawMessage,
    RiskLevel,
    Sensitivity,
)
from services.core.orchestrator import build_ticket_command
from services.core.policy import evaluate_ticket_creation
from services.intake.replay import load_fixture, replay_to_ticket
from services.simulator.app import create_app
from services.simulator.store import TicketSimulator


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "three_bubble_road_complaint.json"


def test_replay_creates_one_submitted_ticket() -> None:
    simulator = TicketSimulator()
    receipt = replay_to_ticket(load_fixture(FIXTURE_PATH), simulator)

    assert receipt.status == "SUBMITTED"
    assert receipt.idempotency_key == "research:case-001:ticket:create:v1"
    assert simulator.ticket_count == 1


def test_replaying_the_same_fixture_is_idempotent() -> None:
    simulator = TicketSimulator()
    fixture = load_fixture(FIXTURE_PATH)

    first = replay_to_ticket(fixture, simulator)
    second = replay_to_ticket(fixture, simulator)

    assert first.ticket_id == second.ticket_id
    assert simulator.ticket_count == 1


def test_policy_denies_missing_mandatory_evidence() -> None:
    result = evaluate_ticket_creation(
        PolicyInput(
            tenant_id="research",
            case_id="case-001",
            revision=1,
            category=Category.ROAD,
            jurisdiction_id="JUR-FICT-01",
            authority_unit_id="UNIT-BINA-MARGA-01",
            missing_fields=("location",),
            has_mandatory_evidence=False,
            idempotency_key="research:case-001:ticket:create:v1",
        )
    )

    assert result.decision == "DENY"
    assert result.reason_codes == ("MANDATORY_EVIDENCE_MISSING",)


def test_simulator_returns_conflict_for_mutated_payload() -> None:
    simulator = TicketSimulator()
    command = _build_command()
    simulator.create_ticket(command.request, command.idempotency_key)

    app = create_app(simulator)
    client = TestClient(app)
    mutated = command.request.model_copy(update={"payload_hash": "0" * 64})
    response = client.post(
        "/v1/tickets",
        headers={"Idempotency-Key": command.idempotency_key},
        json=mutated.model_dump(mode="json"),
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "IDEMPOTENCY_PAYLOAD_MISMATCH"


def test_ticket_api_returns_existing_receipt_and_operation_lookup() -> None:
    app = create_app()
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "healthy"}
    command = _build_command()
    request_json = command.request.model_dump(mode="json")
    headers = {"Idempotency-Key": command.idempotency_key}

    created = client.post("/v1/tickets", headers=headers, json=request_json)
    repeated = client.post("/v1/tickets", headers=headers, json=request_json)
    operation = client.get(f"/v1/operations/{command.idempotency_key}")

    assert created.status_code == 201
    assert repeated.status_code == 200
    assert created.json()["ticket_id"] == repeated.json()["ticket_id"]
    assert operation.status_code == 200
    assert operation.json()["receipt"]["ticket_id"] == created.json()["ticket_id"]


def _build_command():
    fixture = load_fixture(FIXTURE_PATH)
    messages = tuple(
        RawMessage(
            message_id=f"research:{bubble.source_message_id}",
            tenant_id="research",
            conversation_id=fixture.conversation_id,
            source_message_id=bubble.source_message_id,
            text=bubble.text,
            received_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        )
        for bubble in fixture.bubbles
    )
    snapshot = CaseSnapshot(
        case_id="case-001",
        tenant_id="research",
        conversation_id=fixture.conversation_id,
        revision=1,
        state=ProcessingState.READY,
        messages=messages,
        evidence_hash=sha256(b"fixture").hexdigest(),
        created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
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
    assert command is not None
    return command
