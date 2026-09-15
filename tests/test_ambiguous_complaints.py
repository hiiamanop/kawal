from __future__ import annotations

import pytest

from contracts.models import Category, DecisionMode, ProcessingState
from services.clarification.dispatcher import ClarificationDispatcher
from services.core.pipeline import CaseProcessingPipeline
from services.intelligence.causality import analyze_causality
from services.intake.openwa import OpenWAConnector
from services.ml import LocalMLRuntime
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


def test_hazard_distress_cues_prevent_false_rejection() -> None:
    """Verifies that an emergency hazard (tree branch on electric wire with sparks)

    is never falsely rejected as an inquiry or non-complaint.
    """
    runtime = LocalMLRuntime.live_from_artifacts()
    if not runtime.is_available():
        pytest.skip("ONNX runtime artifacts not available")

    pipeline = CaseProcessingPipeline(
        ml_runtime=runtime,
        ticket_client=ReliableTicketClient(simulator=TicketSimulator()),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=OpenWAConnector()),
        default_jurisdiction_id="JUR-FICT-01",
        defer_execution=False,
    )

    text = "min ada dahan pohon roboh nimpa kabel tiang listrik sampe keluar percikan api di pinggir jalan bahaya bgt"
    res = pipeline.process_messages(
        tenant_id="tenant-ambiguous",
        conversation_id="conv-hazard@c.us",
        messages=[text],
        case_id="case-hazard-01",
    )

    assert res.decision_mode != DecisionMode.REJECT_IGNORE, "Emergency hazard must not be rejected"
    assert res.category == Category.FIRE_RESCUE
    assert res.risk.value in ("HIGH", "URGENT")
    assert res.clarification_session is not None


def test_causal_root_cause_water_pipe_over_asphalt_damage() -> None:
    """Verifies that a complaint with causal connector 'bikin' prioritizes

    the root cause (leaking water pipe -> CLEAN_WATER) over surface road damage.
    """
    runtime = LocalMLRuntime.live_from_artifacts()
    if not runtime.is_available():
        pytest.skip("ONNX runtime artifacts not available")

    text = "pipa pdam bocor semburan air deres banget bikin aspal jalan ambles berlubang parah"
    causal = analyze_causality(text, ml_runtime=runtime)
    assert causal.has_causal_relation is True
    assert causal.root_cause_category == Category.CLEAN_WATER
    assert causal.secondary_category == Category.ROAD

    pipeline = CaseProcessingPipeline(
        ml_runtime=runtime,
        ticket_client=ReliableTicketClient(simulator=TicketSimulator()),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=OpenWAConnector()),
        default_jurisdiction_id="JUR-FICT-01",
        defer_execution=False,
    )
    res = pipeline.process_messages(
        tenant_id="tenant-ambiguous",
        conversation_id="conv-causal-water@c.us",
        messages=[text],
        case_id="case-causal-water-01",
    )
    assert res.category == Category.CLEAN_WATER


def test_causal_root_cause_waste_over_drainage_flood() -> None:
    """Verifies that a complaint with causal connector 'gara2' prioritizes

    the root cause (garbage accumulation -> WASTE) over drain overflow.
    """
    runtime = LocalMLRuntime.live_from_artifacts()
    if not runtime.is_available():
        pytest.skip("ONNX runtime artifacts not available")

    text = "min tolong got mampet air comberan meluap ke jalan gara2 tumpukan sampah plastik dibuang warga sembarangan"
    causal = analyze_causality(text, ml_runtime=runtime)
    assert causal.has_causal_relation is True
    assert causal.root_cause_category == Category.WASTE
    assert causal.secondary_category == Category.DRAINAGE_FLOOD

    pipeline = CaseProcessingPipeline(
        ml_runtime=runtime,
        ticket_client=ReliableTicketClient(simulator=TicketSimulator()),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=OpenWAConnector()),
        default_jurisdiction_id="JUR-FICT-01",
        defer_execution=False,
    )
    res = pipeline.process_messages(
        tenant_id="tenant-ambiguous",
        conversation_id="conv-causal-waste@c.us",
        messages=[text],
        case_id="case-causal-waste-01",
    )
    assert res.category == Category.WASTE


def test_public_order_sidewalk_encroachment() -> None:
    """Verifies that sidewalk vendor encroachment is routed to PUBLIC_ORDER (Satpol PP)

    rather than ROAD (Bina Marga).
    """
    runtime = LocalMLRuntime.live_from_artifacts()
    if not runtime.is_available():
        pytest.skip("ONNX runtime artifacts not available")

    pipeline = CaseProcessingPipeline(
        ml_runtime=runtime,
        ticket_client=ReliableTicketClient(simulator=TicketSimulator()),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=OpenWAConnector()),
        default_jurisdiction_id="JUR-FICT-01",
        defer_execution=False,
    )
    text = "trotoar dipake jualan pkl terus motor pada parkir sembarangan bikin macet parah susah jalan kaki"
    res = pipeline.process_messages(
        tenant_id="tenant-ambiguous",
        conversation_id="conv-pkl@c.us",
        messages=[text],
        case_id="case-pkl-01",
    )
    assert res.category == Category.PUBLIC_ORDER


def test_multi_domain_end_to_end_clarification_to_ticket() -> None:
    """Verifies multi-turn clarification on a multi-domain complaint (PDAM pipe causing road collapse):

    Turn 1 asks for location -> Turn 2 citizen replies -> Ticket issued to PDAM (UNIT-PDAM-01).
    """
    runtime = LocalMLRuntime.live_from_artifacts()
    if not runtime.is_available():
        pytest.skip("ONNX runtime artifacts not available")

    simulator = TicketSimulator()
    pipeline = CaseProcessingPipeline(
        ml_runtime=runtime,
        ticket_client=ReliableTicketClient(simulator=simulator),
        clarification_dispatcher=ClarificationDispatcher(openwa_connector=OpenWAConnector()),
        default_jurisdiction_id="JUR-FICT-01",
        defer_execution=False,
    )

    msg1 = "min pipa pdam bocor semburan air deres banget bikin aspal jalan ambles berlubang parah"
    res1 = pipeline.process_messages(
        tenant_id="tenant-ambiguous",
        conversation_id="conv-pdam-e2e@c.us",
        messages=[msg1],
        case_id="case-pdam-e2e-01",
        revision=1,
    )
    assert res1.decision_mode == DecisionMode.REQUEST_CLARIFICATION
    assert res1.category == Category.CLEAN_WATER

    msg2 = "itu min lokasinya di Jl. Cisitu Lama No. 20 RT 03 RW 10 Kelurahan Dago Kecamatan Coblong Kota Bandung"
    res2 = pipeline.process_messages(
        tenant_id="tenant-ambiguous",
        conversation_id="conv-pdam-e2e@c.us",
        messages=[msg1, msg2],
        case_id="case-pdam-e2e-01",
        revision=2,
        previous_category=res1.category,
    )
    assert res2.decision_mode == DecisionMode.EXECUTE
    assert res2.category == Category.CLEAN_WATER
    assert res2.processing_state == ProcessingState.TICKETED
    assert res2.ticket_receipt is not None

    ticket = simulator.get_ticket(res2.ticket_receipt.ticket_id)
    assert ticket.authority_unit_id == "UNIT-PDAM-01"
    assert ticket.title == "Aduan air bersih"
