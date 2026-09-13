from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts.models import Category
from services.clarification.engine import (
    ClarificationStatus,
    check_session_expiry,
    generate_round_questions,
    start_clarification_session,
    submit_clarification_reply,
)


def test_generate_round_questions_max_two() -> None:
    questions = generate_round_questions(
        category=Category.ROAD,
        missing_fields=("location", "time", "issue", "extra_field"),
    )
    assert len(questions) <= 2
    assert len(questions) == 2
    fields = [q.field for q in questions]
    assert "location" in fields
    assert "time" in fields


def test_image_blocked_fallback_question() -> None:
    questions = generate_round_questions(
        category=Category.ROAD,
        missing_fields=("location",),
        image_blocked=True,
    )
    assert len(questions) == 2
    fields = [q.field for q in questions]
    assert "image_fallback" in fields
    assert "location" in fields


def test_start_session_creates_round_1() -> None:
    now = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
    session = start_clarification_session(
        case_id="case-100",
        tenant_id="tenant-bdg",
        conversation_id="conv-100",
        category=Category.WASTE,
        missing_fields=("location", "time"),
        now=now,
    )
    assert session.status == ClarificationStatus.WAITING_REPLY
    assert session.current_round == 1
    assert len(session.rounds) == 1
    assert session.expires_at == now + timedelta(hours=72)
    assert len(session.rounds[0].questions) <= 2


def test_clarification_resolves_in_first_round() -> None:
    now = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
    session = start_clarification_session(
        case_id="case-101",
        tenant_id="tenant-bdg",
        conversation_id="conv-101",
        category=Category.ROAD,
        missing_fields=("location",),
        now=now,
    )
    assert session.status == ClarificationStatus.WAITING_REPLY

    reply_time = now + timedelta(minutes=15)
    updated = submit_clarification_reply(
        session=session,
        reply_text="Lokasinya di Jl. Dago No. 10 RT 01 RW 02 Kelurahan Dago.",
        newly_resolved_fields=("location",),
        now=reply_time,
    )
    assert updated.status == ClarificationStatus.RESOLVED
    assert "location" in updated.resolved_fields


def test_clarification_advances_to_round_2_and_3_then_exceeds_limit() -> None:
    now = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
    session = start_clarification_session(
        case_id="case-102",
        tenant_id="tenant-bdg",
        conversation_id="conv-102",
        category=Category.ROAD,
        missing_fields=("location", "time"),
        now=now,
    )
    assert session.current_round == 1

    # Round 1 reply: resolves nothing
    t1 = now + timedelta(hours=1)
    s2 = submit_clarification_reply(
        session=session,
        reply_text="Halo min belum tahu pastinya.",
        newly_resolved_fields=(),
        now=t1,
    )
    assert s2.current_round == 2
    assert s2.status == ClarificationStatus.WAITING_REPLY
    assert len(s2.rounds) == 2

    # Round 2 reply: resolves "time", but location still missing
    t2 = t1 + timedelta(hours=1)
    s3 = submit_clarification_reply(
        session=s2,
        reply_text="Kejadiannya tadi pagi.",
        newly_resolved_fields=("time",),
        now=t2,
    )
    assert s3.current_round == 3
    assert s3.status == ClarificationStatus.WAITING_REPLY
    assert len(s3.rounds) == 3

    # Round 3 reply: fails to resolve location again
    t3 = t2 + timedelta(hours=1)
    s4 = submit_clarification_reply(
        session=s3,
        reply_text="Saya cuma lewat saja min.",
        newly_resolved_fields=(),
        now=t3,
    )
    assert s4.status == ClarificationStatus.UNRESOLVED_LIMIT
    assert s4.current_round == 3
    assert len(s4.rounds) == 3


def test_clarification_72h_expiry() -> None:
    now = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
    session = start_clarification_session(
        case_id="case-103",
        tenant_id="tenant-bdg",
        conversation_id="conv-103",
        category=Category.DRAINAGE_FLOOD,
        missing_fields=("location",),
        now=now,
    )

    t_check = now + timedelta(hours=72, minutes=5)
    expired_session = check_session_expiry(session, now=t_check)
    assert expired_session.status == ClarificationStatus.UNRESOLVED_EXPIRED

    late_reply = submit_clarification_reply(
        session=session,
        reply_text="Jl. Cihampelas No. 5",
        newly_resolved_fields=("location",),
        now=t_check,
    )
    assert late_reply.status == ClarificationStatus.UNRESOLVED_EXPIRED


def test_resolved_fields_never_reasked() -> None:
    now = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
    session = start_clarification_session(
        case_id="case-104",
        tenant_id="tenant-bdg",
        conversation_id="conv-104",
        category=Category.WASTE,
        missing_fields=("location", "time"),
        now=now,
    )

    t1 = now + timedelta(minutes=10)
    s2 = submit_clarification_reply(
        session=session,
        reply_text="Lokasi di TPS RW 04",
        newly_resolved_fields=("location",),
        now=t1,
    )
    assert s2.current_round == 2
    round_2_fields = [q.field for q in s2.rounds[1].questions]
    assert "location" not in round_2_fields
    assert "time" in round_2_fields
