from __future__ import annotations

from contracts.models import (
    Category,
    DecisionMode,
    DeliveryStatus,
    ProcessingState,
    TicketStatus,
)
from services.clarification.dispatcher import ClarificationDispatcher
from services.core.pipeline import CaseProcessingPipeline
from services.intake.openwa import OpenWAConnector
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


def test_pipeline_complete_complaint_executes_to_ticketed() -> None:
    simulator = TicketSimulator()
    connector = OpenWAConnector()
    pipeline = CaseProcessingPipeline(
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=connector),
        default_jurisdiction_id="JUR-FICT-01",
    )

    messages = [
        "Selamat pagi min, lapor aduan jalan berlubang parah.",
        "Lokasinya Jl. Raya Kopo No. 50, RT 41 RW 31, Kelurahan Babakan Asih, Kecamatan Bojongloa Kaler, Kota Bandung. Mohon segera diperbaiki.",
    ]

    result = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="62811223344@c.us",
        messages=messages,
        case_id="case-pipe-road-1",
    )

    assert result.decision_mode == DecisionMode.EXECUTE
    assert result.processing_state == ProcessingState.TICKETED
    assert result.category == Category.ROAD
    assert result.completeness == "SUFFICIENT"
    assert result.ticket_receipt is not None
    assert result.ticket_receipt.status == TicketStatus.SUBMITTED
    assert simulator.ticket_count == 1

    ticket = simulator.get_ticket(result.ticket_receipt.ticket_id)
    assert ticket.authority_unit_id == "UNIT-BINA-MARGA-01"
    assert ticket.title == "Aduan jalan rusak"


def test_pipeline_incomplete_complaint_requests_clarification() -> None:
    simulator = TicketSimulator()
    connector = OpenWAConnector()
    pipeline = CaseProcessingPipeline(
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=connector),
        default_jurisdiction_id="JUR-FICT-01",
    )

    messages = [
        "Halo petugas, aduan sampah menumpuk di pinggir jalan dekat pasar.",
    ]

    result = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="62811223344@c.us",
        messages=messages,
        case_id="case-pipe-incomplete-1",
    )

    assert result.decision_mode == DecisionMode.REQUEST_CLARIFICATION
    assert result.processing_state == ProcessingState.WAITING_CLARIFICATION
    assert result.ticket_receipt is None
    assert simulator.ticket_count == 0

    assert result.clarification_session is not None
    assert result.clarification_session.current_round == 1
    assert result.outbound_receipt is not None
    assert result.outbound_receipt.status == DeliveryStatus.SENT

    assert len(connector.sent_messages) == 1
    sent_msg = connector.sent_messages[0]
    assert "Halo, terima kasih telah menghubungi layanan aduan." in sent_msg["text"]


def test_pipeline_resolves_after_citizen_replies_with_address() -> None:
    simulator = TicketSimulator()
    connector = OpenWAConnector()
    pipeline = CaseProcessingPipeline(
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=connector),
        default_jurisdiction_id="JUR-FICT-01",
    )

    # Initial incomplete report
    res1 = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="628555666777@c.us",
        messages=["Lapor tumpukan sampah bau menyengat."],
        case_id="case-pipe-resolve-1",
        revision=1,
    )
    assert res1.processing_state == ProcessingState.WAITING_CLARIFICATION
    assert simulator.ticket_count == 0

    # Citizen replies with complete address
    updated_messages = [
        "Lapor tumpukan sampah bau menyengat.",
        "Lokasinya Jl. Cihampelas No. 25, RT 02 RW 04, Kelurahan Cipaganti, Kecamatan Coblong, Kota Bandung.",
    ]
    res2 = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="628555666777@c.us",
        messages=updated_messages,
        case_id="case-pipe-resolve-1",
        revision=2,
    )
    assert res2.decision_mode == DecisionMode.EXECUTE
    assert res2.processing_state == ProcessingState.TICKETED
    assert res2.ticket_receipt is not None
    assert simulator.ticket_count == 1


def test_pipeline_reject_spam_or_opinion() -> None:
    simulator = TicketSimulator()
    connector = OpenWAConnector()
    pipeline = CaseProcessingPipeline(
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=connector),
    )

    result = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="628111222333@c.us",
        messages=["Sekadar opini dan salam sapa untuk rekan dinas di hari libur ini."],
        case_id="case-pipe-spam-1",
    )

    assert result.decision_mode == DecisionMode.REJECT_IGNORE
    assert result.processing_state == ProcessingState.REJECTED
    assert result.ticket_receipt is None
    assert simulator.ticket_count == 0
    assert len(connector.sent_messages) == 0


def test_pipeline_idempotent_replay_zero_duplicate_tickets() -> None:
    simulator = TicketSimulator()
    connector = OpenWAConnector()
    pipeline = CaseProcessingPipeline(
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=connector),
        default_jurisdiction_id="JUR-FICT-01",
    )

    messages = [
        "Lapor aspal jalan ambles parah.",
        "Lokasinya Jl. Sukajadi No. 10, RT 01 RW 02, Kelurahan Sukawarna, Kecamatan Sukajadi, Kota Bandung.",
    ]

    # First run
    res1 = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="628111999888@c.us",
        messages=messages,
        case_id="case-idemp-pipe-1",
    )
    assert simulator.ticket_count == 1

    # Second run with exact same case ID and messages (replay)
    res2 = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="628111999888@c.us",
        messages=messages,
        case_id="case-idemp-pipe-1",
    )

    # Invariant: Exactly one ticket remains in simulator, receipt matches!
    assert simulator.ticket_count == 1
    assert res1.ticket_receipt is not None and res2.ticket_receipt is not None
    assert res1.ticket_receipt.ticket_id == res2.ticket_receipt.ticket_id
    assert res1.ticket_receipt.idempotency_key == res2.ticket_receipt.idempotency_key
