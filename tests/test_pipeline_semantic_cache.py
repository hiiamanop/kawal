from __future__ import annotations

import pytest

from contracts.models import Category, DecisionMode, ProcessingState
from services.clarification.dispatcher import ClarificationDispatcher
from services.core.pipeline import CaseProcessingPipeline
from services.intake.openwa import OpenWAConnector
from services.intelligence.semantic_cache import InMemoryCacheStore, SemanticCacheEngine
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


def test_pipeline_semantic_deduplication_avoids_duplicate_tickets() -> None:
    simulator = TicketSimulator()
    connector = OpenWAConnector()
    cache_store = InMemoryCacheStore()
    semantic_cache = SemanticCacheEngine(store=cache_store)

    pipeline = CaseProcessingPipeline(
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=connector),
        semantic_cache=semantic_cache,
        default_jurisdiction_id="JUR-FICT-01",
    )

    # 1. Citizen A reports road damage on Buah Batu
    msg_a = [
        "Selamat pagi min, lapor aduan jalan berlubang parah sering bikin motor jatuh.",
        "Lokasinya Jl. Terusan Buah Batu No. 120, RT 02 RW 05, Kelurahan Kujangsari, Kecamatan Bandung Kidul, Kota Bandung.",
    ]
    res_a = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="citizen-a@c.us",
        messages=msg_a,
        case_id="case-citizen-a",
    )

    assert res_a.decision_mode == DecisionMode.EXECUTE
    assert res_a.processing_state == ProcessingState.TICKETED
    assert simulator.ticket_count == 1
    ticket_a_id = res_a.ticket_receipt.ticket_id

    # 2. Citizen B reports same damaged road on Buah Batu shortly after
    msg_b = [
        "Min tolong ini jalanan buah batu rusak parah bolong gede banget bahaya.",
        "Lokasi di jalan terusan buah batu bandung.",
    ]
    res_b = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="citizen-b@c.us",
        messages=msg_b,
        case_id="case-citizen-b",
    )

    # Must be linked to the same active incident ticket
    assert res_b.decision_mode == DecisionMode.EXECUTE
    assert res_b.processing_state == ProcessingState.TICKETED
    assert res_b.ticket_receipt.ticket_id == ticket_a_id
    assert "DUPLICATE_INCIDENT_LINKED" in res_b.reason_codes
    # CRITICAL: No new ticket was created in simulator!
    assert simulator.ticket_count == 1
    # Citizen B receives friendly message referencing existing ticket
    assert ticket_a_id in res_b.outbound_command.text

    # 3. Citizen C reports road damage in a DIFFERENT location (Kopo)
    msg_c = [
        "Lapor jalan rusak parah banyak lubang di kopo.",
        "Lokasinya Jl. Raya Kopo No. 50, RT 01 RW 02, Kelurahan Kopo, Kecamatan Bojongloa Kaler, Kota Bandung.",
    ]
    res_c = pipeline.process_messages(
        tenant_id="tenant-bdg",
        conversation_id="citizen-c@c.us",
        messages=msg_c,
        case_id="case-citizen-c",
    )

    # Different location must create a new distinct ticket!
    assert res_c.decision_mode == DecisionMode.EXECUTE
    assert res_c.processing_state == ProcessingState.TICKETED
    assert res_c.ticket_receipt.ticket_id != ticket_a_id
    assert simulator.ticket_count == 2
