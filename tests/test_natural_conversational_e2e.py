from __future__ import annotations

import pytest

from contracts.models import Category, DecisionMode, ProcessingState
from services.clarification.dispatcher import ClarificationDispatcher
from services.core.pipeline import CaseProcessingPipeline
from services.intake.openwa import OpenWAConnector
from services.ml import LocalMLRuntime
from services.reliability.client import ReliableTicketClient
from services.simulator.store import TicketSimulator


def test_landmark_disentanglement_road_near_hospital_multi_turn() -> None:
    """Verifies that mentioning a hospital as a landmark does not distort a road damage complaint into HEALTH_SERVICE."""
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

    tenant_id = "tenant-landmark-test"
    conv_id = "628999111222@c.us"
    case_id = "case-landmark-road-01"

    # Turn 1: Informal citizen complaint with landmark "rs immanuel"
    msg1 = (
        "halo min punten pisan, ini jalan kopo deket rs immanuel aspalnya ancur parah "
        "bolong2 gede bgt, semalem ada bapak2 jatoh dari motor pas ujan. tolong bgt segera ditambal bahaya"
    )
    res1 = pipeline.process_messages(
        tenant_id=tenant_id,
        conversation_id=conv_id,
        messages=[msg1],
        case_id=case_id,
        revision=1,
    )

    assert res1.category == Category.ROAD
    assert res1.decision_mode == DecisionMode.REQUEST_CLARIFICATION
    assert res1.completeness in ("INCOMPLETE", "AMBIGUOUS")
    assert res1.clarification_session is not None

    # Turn 2: Citizen replies repeating landmark "rs immanuel" and giving full address
    msg2 = (
        "itu min patokannya pas depan rs immanuel, tepatnya di Jl. Raya Kopo No. 120 "
        "RT 04 RW 02 Kelurahan Babakan Asih Kecamatan Bojongloa Kaler Kota Bandung"
    )
    res2 = pipeline.process_messages(
        tenant_id=tenant_id,
        conversation_id=conv_id,
        messages=[msg1, msg2],
        case_id=case_id,
        revision=2,
        previous_category=res1.category,
    )

    assert res2.category == Category.ROAD
    assert res2.decision_mode == DecisionMode.EXECUTE
    assert res2.processing_state == ProcessingState.TICKETED
    assert res2.completeness == "SUFFICIENT"
    assert res2.ticket_receipt is not None

    ticket = simulator.get_ticket(res2.ticket_receipt.ticket_id)
    assert ticket.authority_unit_id == "UNIT-BINA-MARGA-01"
    assert ticket.title == "Aduan jalan rusak"


def test_genuine_health_complaint_is_not_masked_away() -> None:
    """Verifies that genuine healthcare issues are correctly classified as HEALTH_SERVICE even with landmarks."""
    runtime = LocalMLRuntime.live_from_artifacts()
    if not runtime.is_available():
        pytest.skip("ONNX runtime artifacts not available")

    text = "halo min tolong, dokter jaga di rs immanuel ga ada yg masuk pasien igd terlantar butuh penanganan segera"
    res = runtime.predict(text)
    assert res.prediction is not None
    assert res.prediction.category == "HEALTH_SERVICE"
    assert res.prediction.confidence > 0.90


def test_road_outside_school_vs_classroom_damage() -> None:
    """Verifies that road issues outside a school remain ROAD, while classroom damage remains EDUCATION."""
    runtime = LocalMLRuntime.live_from_artifacts()
    if not runtime.is_available():
        pytest.skip("ONNX runtime artifacts not available")

    road_near_school = "min jalan rusak parah berlubang pas depan sdn bojongloa banyak anak sekolah keserempet"
    res_road = runtime.predict(road_near_school)
    assert res_road.prediction is not None
    assert res_road.prediction.category == "ROAD"

    classroom_damage = "min atap ruang kelas di sdn bojongloa ambruk plafon jebol membahayakan proses belajar mengajar"
    res_school = runtime.predict(classroom_damage)
    assert res_school.prediction is not None
    assert res_school.prediction.category == "EDUCATION"
