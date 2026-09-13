from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts.models import (
    Category,
    DeliveryStatus,
    RawMessage,
)
from services.clarification.dispatcher import (
    ClarificationDispatcher,
    format_clarification_message,
)
from services.clarification.engine import (
    ClarificationStatus,
    start_clarification_session,
)
from services.intake.openwa import OpenWAConnector


def _make_raw_reply(
    text: str,
    quoted_id: str = "msg-kawal-q1",
    conv_id: str = "628999888777@c.us",
) -> RawMessage:
    return RawMessage(
        message_id=f"tenant-bdg:{conv_id}:src-reply-1",
        tenant_id="tenant-bdg",
        conversation_id=conv_id,
        source_message_id="src-reply-1",
        text=text,
        received_at=datetime.now(timezone.utc),
    )


def test_format_clarification_message() -> None:
    session = start_clarification_session(
        case_id="case-disp-1",
        tenant_id="tenant-bdg",
        conversation_id="628111222333@c.us",
        category=Category.ROAD,
        missing_fields=("location",),
    )
    formatted = format_clarification_message(session)
    assert "Halo, terima kasih telah menghubungi layanan aduan." in formatted
    assert "1." in formatted
    assert "Jalan rusak atau berlubang" in formatted
    assert "Balas pesan ini langsung" in formatted


def test_dispatch_clarification_request_idempotent() -> None:
    connector = OpenWAConnector()
    dispatcher = ClarificationDispatcher(openwa_connector=connector)

    session = start_clarification_session(
        case_id="case-disp-2",
        tenant_id="tenant-bdg",
        conversation_id="628111222333@c.us",
        category=Category.DRAINAGE_FLOOD,
        missing_fields=("location", "time"),
    )

    # First dispatch
    session_out, receipt1 = dispatcher.dispatch_clarification_request(session, quoted_source_message_id="msg-orig-1")
    assert receipt1.status == DeliveryStatus.SENT
    assert receipt1.idempotency_key == "tenant-bdg:628111222333@c.us:clarification:round_1"
    assert len(connector.sent_messages) == 1
    assert connector.sent_messages[0]["quoted_msg_id"] == "msg-orig-1"

    # Second dispatch with identical session (replay)
    _, receipt2 = dispatcher.dispatch_clarification_request(session, quoted_source_message_id="msg-orig-1")
    assert receipt2.status == DeliveryStatus.SENT
    assert receipt2.send_id == receipt1.send_id
    assert receipt2.source_message_id == receipt1.source_message_id

    # Crucial: transport was called ONLY ONCE! Zero duplicate sends over WhatsApp
    assert len(connector.sent_messages) == 1


def test_handle_citizen_reply_resolves_session() -> None:
    connector = OpenWAConnector()
    dispatcher = ClarificationDispatcher(openwa_connector=connector)

    session = start_clarification_session(
        case_id="case-disp-3",
        tenant_id="tenant-bdg",
        conversation_id="628999888777@c.us",
        category=Category.WASTE,
        missing_fields=("location",),
    )
    _, q_receipt = dispatcher.dispatch_clarification_request(session)
    assert len(connector.sent_messages) == 1

    # Citizen replies with full location
    citizen_reply = _make_raw_reply(
        text="Lokasi sampahnya di Jl. Cihampelas No. 20, RT 03 RW 05.",
        quoted_id=q_receipt.source_message_id or "q-msg",
        conv_id="628999888777@c.us",
    )

    updated_session, next_receipt = dispatcher.handle_citizen_reply(
        session=session,
        reply_message=citizen_reply,
        resolved_fields=("location",),
    )

    assert updated_session.status == ClarificationStatus.RESOLVED
    assert "location" in updated_session.resolved_fields
    assert next_receipt is None
    # No additional messages dispatched since problem is resolved
    assert len(connector.sent_messages) == 1


def test_handle_citizen_reply_advances_round_and_auto_dispatches_round_2() -> None:
    connector = OpenWAConnector()
    dispatcher = ClarificationDispatcher(openwa_connector=connector)

    session = start_clarification_session(
        case_id="case-disp-4",
        tenant_id="tenant-bdg",
        conversation_id="628999888777@c.us",
        category=Category.ROAD,
        missing_fields=("location", "time"),
    )
    _, receipt_r1 = dispatcher.dispatch_clarification_request(session)
    assert len(connector.sent_messages) == 1

    # Citizen reply only answers time, location remains missing
    citizen_reply_r1 = _make_raw_reply(
        text="Kejadiannya baru tadi pagi pas jam berangkat kerja.",
        quoted_id=receipt_r1.source_message_id or "q1",
    )

    updated_session, receipt_r2 = dispatcher.handle_citizen_reply(
        session=session,
        reply_message=citizen_reply_r1,
        resolved_fields=("time",),
    )

    assert updated_session.status == ClarificationStatus.WAITING_REPLY
    assert updated_session.current_round == 2
    assert "time" in updated_session.resolved_fields
    assert "location" in updated_session.missing_fields

    # Auto-dispatch triggered for round 2!
    assert receipt_r2 is not None
    assert receipt_r2.status == DeliveryStatus.SENT
    assert receipt_r2.idempotency_key == "tenant-bdg:628999888777@c.us:clarification:round_2"
    assert len(connector.sent_messages) == 2


def test_clarification_stops_at_round_3_limit_without_infinite_loop() -> None:
    connector = OpenWAConnector()
    dispatcher = ClarificationDispatcher(openwa_connector=connector)

    session = start_clarification_session(
        case_id="case-disp-5",
        tenant_id="tenant-bdg",
        conversation_id="628999888777@c.us",
        category=Category.ROAD,
        missing_fields=("location",),
    )
    dispatcher.dispatch_clarification_request(session)
    assert len(connector.sent_messages) == 1

    # Round 1 reply (no resolution) -> triggers Round 2 dispatch
    s2, r2 = dispatcher.handle_citizen_reply(session, _make_raw_reply("jawaban 1"))
    assert s2.current_round == 2
    assert len(connector.sent_messages) == 2

    # Round 2 reply (no resolution) -> triggers Round 3 dispatch
    s3, r3 = dispatcher.handle_citizen_reply(s2, _make_raw_reply("jawaban 2"))
    assert s3.current_round == 3
    assert len(connector.sent_messages) == 3

    # Round 3 reply (still no resolution) -> status UNRESOLVED_LIMIT, no round 4 dispatch!
    s4, r4 = dispatcher.handle_citizen_reply(s3, _make_raw_reply("jawaban 3"))
    assert s4.status == ClarificationStatus.UNRESOLVED_LIMIT
    assert s4.current_round == 3
    assert r4 is None
    # Still only 3 messages sent
    assert len(connector.sent_messages) == 3
