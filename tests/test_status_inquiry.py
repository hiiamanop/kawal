from __future__ import annotations

import pytest

from contracts.models import DecisionMode, OutboundPurpose
from services.clarification.dispatcher import ClarificationDispatcher
from services.core.pipeline import CaseProcessingPipeline
from services.intake.openwa import OpenWAConnector
from services.intelligence.semantic_cache import InMemoryCacheStore, SemanticCacheEngine
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


def test_status_inquiry_returns_active_ticket_info_and_tracking_link() -> None:
    simulator = TicketSimulator()
    connector = OpenWAConnector()
    cache_store = InMemoryCacheStore()
    semantic_cache = SemanticCacheEngine(store=cache_store)

    pipeline = CaseProcessingPipeline(
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=connector),
        semantic_cache=semantic_cache,
    )

    # 1. Citizen submits a complete complaint and receives a ticket
    msg_complaint = [
        "Selamat pagi, lapor jalan berlubang di Jl. Kopo No. 10, RT 01 RW 02, Kelurahan Kopo, Kecamatan Bojongloa Kaler, Kota Bandung.",
    ]
    res_ticket = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="citizen-status@c.us",
        messages=msg_complaint,
        case_id="case-stat-1",
    )
    assert res_ticket.ticket_receipt is not None
    issued_ticket_id = res_ticket.ticket_receipt.ticket_id

    # 2. Citizen asks for status via WhatsApp
    msg_inquiry = ["min mau cek status laporan saya kemarin"]
    res_inquiry = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="citizen-status@c.us",
        messages=msg_inquiry,
        case_id="case-stat-2",
    )

    assert "STATUS_INQUIRY_ANSWERED" in res_inquiry.reason_codes
    assert res_inquiry.outbound_command is not None
    assert res_inquiry.outbound_command.purpose == OutboundPurpose.STATUS_UPDATE
    assert issued_ticket_id in res_inquiry.outbound_command.text
    assert f"/track/{issued_ticket_id}" in res_inquiry.outbound_command.text


def test_status_inquiry_without_prior_ticket() -> None:
    cache_store = InMemoryCacheStore()
    pipeline = CaseProcessingPipeline(
        semantic_cache=SemanticCacheEngine(store=cache_store),
    )

    res = pipeline.process_messages(
        tenant_id="tenant-unknown",
        conversation_id="unknown-citizen@c.us",
        messages=["cek status tiket"],
        case_id="case-stat-3",
    )

    assert "STATUS_INQUIRY_ANSWERED" in res.reason_codes
    assert res.outbound_command is not None
    assert "belum menemukan tiket aktif" in res.outbound_command.text
