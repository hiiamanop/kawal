from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from contracts.models import (
    Category,
    TicketCreateRequest,
    TicketPriority,
    TicketStatus,
    TicketVisibility,
)
from services.simulator.app import create_app
from services.simulator.store import (
    FaultInjectionError,
    FaultProfile,
    IdempotencyConflictError,
    InvalidStateTransitionError,
    StaleRevisionError,
    TicketCloseRequest,
    TicketNotFoundError,
    TicketSimulator,
    TicketTransferRequest,
    TicketUpdateRequest,
)


def _sample_create_request(case_id: str = "case-1", title: str = "Aduan Jalan Rusak") -> TicketCreateRequest:
    return TicketCreateRequest(
        tenant_id="tenant-bdg",
        case_id=case_id,
        case_revision=1,
        category=Category.ROAD,
        priority=TicketPriority.NORMAL,
        visibility=TicketVisibility.NORMAL,
        title=title,
        description="Jalan berlubang cukup dalam di Jl. Dago No. 10.",
        jurisdiction_id="JUR-FICT-01",
        authority_unit_id="UNIT-BINA-MARGA-01",
        source_decision_id="dec-1",
        payload_hash="a" * 64,
    )


def test_create_ticket_idempotent() -> None:
    simulator = TicketSimulator()
    req = _sample_create_request()
    key = "tenant-bdg:case-1:ticket:create:v1"

    receipt1 = simulator.create_ticket(req, key)
    assert receipt1.status == TicketStatus.SUBMITTED
    assert receipt1.idempotency_key == key
    assert simulator.ticket_count == 1

    receipt2 = simulator.create_ticket(req, key)
    assert receipt2.ticket_id == receipt1.ticket_id
    assert receipt2.external_id == receipt1.external_id
    assert simulator.ticket_count == 1


def test_create_ticket_conflict_on_mutated_payload() -> None:
    simulator = TicketSimulator()
    req = _sample_create_request()
    key = "tenant-bdg:case-1:ticket:create:v1"
    simulator.create_ticket(req, key)

    mutated = req.model_copy(update={"payload_hash": "b" * 64})
    with pytest.raises(IdempotencyConflictError):
        simulator.create_ticket(mutated, key)
    assert simulator.ticket_count == 1


def test_get_ticket_and_not_found() -> None:
    simulator = TicketSimulator()
    with pytest.raises(TicketNotFoundError):
        simulator.get_ticket("non-existent")

    receipt = simulator.create_ticket(_sample_create_request(), "key-1")
    ticket = simulator.get_ticket(receipt.ticket_id)
    assert ticket.ticket_id == receipt.ticket_id
    assert ticket.revision == 1
    assert ticket.status == TicketStatus.SUBMITTED


def test_update_ticket_success_and_stale_revision() -> None:
    simulator = TicketSimulator()
    receipt = simulator.create_ticket(_sample_create_request(), "create-key")
    ticket_id = receipt.ticket_id

    update_req = TicketUpdateRequest(
        title="Aduan Jalan Rusak (Pembaruan)",
        priority=TicketPriority.HIGH,
        payload_hash="c" * 64,
    )

    with pytest.raises(StaleRevisionError):
        simulator.update_ticket(ticket_id, update_req, "update-key-1", if_match_revision=99)

    updated_receipt = simulator.update_ticket(
        ticket_id, update_req, "update-key-1", if_match_revision=1
    )
    assert updated_receipt.ticket_id == ticket_id

    ticket = simulator.get_ticket(ticket_id)
    assert ticket.revision == 2
    assert ticket.title == "Aduan Jalan Rusak (Pembaruan)"
    assert ticket.priority == TicketPriority.HIGH


def test_transfer_ticket_records_custody_trail() -> None:
    simulator = TicketSimulator()
    receipt = simulator.create_ticket(_sample_create_request(), "create-key")
    ticket_id = receipt.ticket_id

    transfer_req = TicketTransferRequest(
        target_jurisdiction_id="JUR-FICT-02",
        target_authority_unit_id="UNIT-BINA-MARGA-02",
        reason="Kewenangan lintas wilayah kecamatan tetangga",
        payload_hash="d" * 64,
    )
    simulator.transfer_ticket(ticket_id, transfer_req, "transfer-key-1", if_match_revision=1)

    ticket = simulator.get_ticket(ticket_id)
    assert ticket.revision == 2
    assert ticket.jurisdiction_id == "JUR-FICT-02"
    assert ticket.authority_unit_id == "UNIT-BINA-MARGA-02"
    assert len(ticket.custody_trail) == 1
    assert ticket.custody_trail[0]["from_jurisdiction"] == "JUR-FICT-01"
    assert ticket.custody_trail[0]["to_jurisdiction"] == "JUR-FICT-02"


def test_close_ticket_and_cannot_reclose_or_transfer() -> None:
    simulator = TicketSimulator()
    receipt = simulator.create_ticket(_sample_create_request(), "create-key")
    ticket_id = receipt.ticket_id

    close_req = TicketCloseRequest(
        reason="Perbaikan telah selesai dilaksanakan di lapangan",
        payload_hash="e" * 64,
    )
    simulator.close_ticket(ticket_id, close_req, "close-key-1", if_match_revision=1)

    closed_ticket = simulator.get_ticket(ticket_id)
    assert closed_ticket.status == TicketStatus.CLOSED
    assert closed_ticket.close_reason == "Perbaikan telah selesai dilaksanakan di lapangan"
    assert closed_ticket.revision == 2

    with pytest.raises(InvalidStateTransitionError):
        simulator.close_ticket(ticket_id, close_req, "close-key-2")

    transfer_req = TicketTransferRequest(
        target_jurisdiction_id="JUR-FICT-02",
        target_authority_unit_id="UNIT-BINA-MARGA-02",
        reason="Pindah unit",
        payload_hash="f" * 64,
    )
    with pytest.raises(InvalidStateTransitionError):
        simulator.transfer_ticket(ticket_id, transfer_req, "transfer-key-2")


def test_fault_profile_fail_before_commit() -> None:
    simulator = TicketSimulator()
    simulator.set_fault_profile(FaultProfile(fail_before_commit=True))

    with pytest.raises(FaultInjectionError):
        simulator.create_ticket(_sample_create_request(), "fail-key")

    assert simulator.ticket_count == 0
    assert simulator.lookup_operation("fail-key") is None


def test_fault_profile_commit_then_timeout() -> None:
    simulator = TicketSimulator()
    simulator.set_fault_profile(FaultProfile(commit_then_timeout=True))

    key = "tenant-bdg:case-timeout:ticket:create:v1"
    with pytest.raises(TimeoutError):
        simulator.create_ticket(_sample_create_request(case_id="case-timeout"), key)

    assert simulator.ticket_count == 1
    op = simulator.lookup_operation(key)
    assert op is not None
    assert op.status == "COMPLETED"


def test_simulator_fastapi_endpoints() -> None:
    simulator = TicketSimulator()
    app = create_app(simulator)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    req = _sample_create_request()
    resp = client.post(
        "/v1/tickets",
        json=req.model_dump(mode="json"),
        headers={"Idempotency-Key": "key-api-1"},
    )
    assert resp.status_code == 201
    ticket_id = resp.json()["ticket_id"]

    resp2 = client.post(
        "/v1/tickets",
        json=req.model_dump(mode="json"),
        headers={"Idempotency-Key": "key-api-1"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["ticket_id"] == ticket_id

    get_resp = client.get(f"/v1/tickets/{ticket_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "SUBMITTED"

    op_resp = client.get("/v1/operations/key-api-1")
    assert op_resp.status_code == 200
    assert op_resp.json()["status"] == "COMPLETED"

    update_resp = client.patch(
        f"/v1/tickets/{ticket_id}",
        json={"title": "Judul Baru", "payload_hash": "hash-up"},
        headers={"Idempotency-Key": "up-key-1", "If-Match": "1"},
    )
    assert update_resp.status_code == 200

    transfer_resp = client.post(
        f"/v1/tickets/{ticket_id}/transfer",
        json={
            "target_jurisdiction_id": "JUR-FICT-03",
            "target_authority_unit_id": "UNIT-BINA-MARGA-03",
            "reason": "Alih unit",
            "payload_hash": "hash-tr",
        },
        headers={"Idempotency-Key": "tr-key-1", "If-Match": "2"},
    )
    assert transfer_resp.status_code == 200

    close_resp = client.post(
        f"/v1/tickets/{ticket_id}/close",
        json={"reason": "Selesai", "payload_hash": "hash-cl"},
        headers={"Idempotency-Key": "cl-key-1", "If-Match": "3"},
    )
    assert close_resp.status_code == 200
    assert close_resp.json()["status"] == "CLOSED"
