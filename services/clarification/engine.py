from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import Category

MAX_QUESTIONS_PER_ROUND = 2
MAX_ROUNDS = 3
EXPIRY_HOURS = 72.0


class ClarificationStatus(StrEnum):
    WAITING_REPLY = "WAITING_REPLY"
    RESOLVED = "RESOLVED"
    UNRESOLVED_LIMIT = "UNRESOLVED_LIMIT"
    UNRESOLVED_EXPIRED = "UNRESOLVED_EXPIRED"


class ClarificationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(min_length=1)
    question_text: str = Field(min_length=1)


class ClarificationRound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_number: int = Field(ge=1, le=MAX_ROUNDS)
    questions: tuple[ClarificationQuestion, ...] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_ROUND)
    asked_at: datetime
    reply_text: str | None = None
    replied_at: datetime | None = None


class ClarificationSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    case_id: str
    tenant_id: str
    conversation_id: str
    category: Category
    status: ClarificationStatus
    rounds: tuple[ClarificationRound, ...]
    current_round: int = Field(ge=1, le=MAX_ROUNDS)
    missing_fields: tuple[str, ...]
    resolved_fields: tuple[str, ...] = ()
    image_fallback_active: bool = False
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


CATEGORY_LOCATION_QUESTIONS: dict[Category, str] = {
    Category.ROAD: "Jalan rusak atau berlubang berada di ruas jalan mana dan dekat patokan apa?",
    Category.DRAINAGE_FLOOD: "Titik genangan atau saluran tersumbat berada di jalan mana dan dekat patokan apa?",
    Category.WASTE: "Tumpukan sampah berada di jalan atau lokasi mana dan dekat patokan apa?",
    Category.CLEAN_WATER: "Pipa bocor atau gangguan air bersih terjadi di alamat atau ruas jalan mana?",
    Category.CIVIL_ADMIN: "Layanan administrasi kependudukan di kantor atau kelurahan mana yang dimaksud?",
    Category.HEALTH_SERVICE: "Fasilitas kesehatan atau puskesmas mana yang Anda maksud?",
    Category.PUBLIC_ORDER: "Gangguan ketertiban umum terjadi di ruas jalan atau lokasi mana?",
    Category.TRANSPORTATION: "Fasilitas transportasi atau rambu di jalan/persimpangan mana yang bermasalah?",
    Category.FIRE_RESCUE: "Lokasi kebakaran atau kebutuhan evakuasi tepatnya di alamat atau patokan mana?",
    Category.SOCIAL_AFFAIRS: "Warga yang membutuhkan bantuan sosial terpantau di jalan atau lokasi mana?",
    Category.EDUCATION: "Sekolah atau fasilitas pendidikan mana yang dimaksud?",
    Category.PARKS_HOUSING: "Taman umum atau kawasan perumahan mana yang mengalami kendala fasilitas?",
}

GENERIC_FIELD_QUESTIONS: dict[str, str] = {
    "location": "Mohon sebutkan nama jalan, nomor, RT/RW, kelurahan, atau patokan lokasi spesifik aduan Anda.",
    "location_incomplete": "Alamat belum lengkap. Mohon sebutkan nama jalan, RT/RW, atau kelurahan lokasi kejadian.",
    "location_ambiguous": "Patokan lokasi yang Anda sebutkan masih ambigu. Mohon jelaskan nama kelurahan dan kecamatan atau patokan terdekat.",
    "time": "Kapan permasalahan ini mulai terjadi atau terakhir kali terpantau?",
    "issue": "Bisa jelaskan lebih detail fasilitas atau sarana apa yang mengalami kerusakan?",
    "image_fallback": "Lampiran gambar Anda tidak dapat diproses secara otomatis demi privasi. Mohon jelaskan kondisi kerusakan dan patokan lokasi secara tertulis.",
}


def get_question_for_field(field: str, category: Category) -> str:
    norm_field = field.lower().strip()
    if norm_field in ("location", "location_incomplete") and category in CATEGORY_LOCATION_QUESTIONS:
        return CATEGORY_LOCATION_QUESTIONS[category]
    if norm_field in GENERIC_FIELD_QUESTIONS:
        return GENERIC_FIELD_QUESTIONS[norm_field]
    if "loc" in norm_field or "alamat" in norm_field:
        return GENERIC_FIELD_QUESTIONS["location"]
    if "time" in norm_field or "waktu" in norm_field:
        return GENERIC_FIELD_QUESTIONS["time"]
    return f"Mohon berikan informasi lebih lanjut mengenai {field} agar aduan dapat kami proses."


def generate_round_questions(
    category: Category,
    missing_fields: Sequence[str],
    resolved_fields: Sequence[str] = (),
    already_asked_fields: Sequence[str] = (),
    image_blocked: bool = False,
) -> tuple[ClarificationQuestion, ...]:
    resolved_set = {f.lower().strip() for f in resolved_fields}
    asked_set = {f.lower().strip() for f in already_asked_fields}
    candidates: list[ClarificationQuestion] = []

    if image_blocked and "image_fallback" not in resolved_set and "image_fallback" not in asked_set:
        candidates.append(
            ClarificationQuestion(
                field="image_fallback",
                question_text=GENERIC_FIELD_QUESTIONS["image_fallback"],
            )
        )

    for field in missing_fields:
        norm = field.lower().strip()
        if norm in resolved_set or norm in asked_set:
            continue
        if any(c.field == norm for c in candidates):
            continue
        q_text = get_question_for_field(field, category)
        candidates.append(ClarificationQuestion(field=norm, question_text=q_text))
        if len(candidates) >= MAX_QUESTIONS_PER_ROUND:
            break

    if not candidates and missing_fields:
        for field in missing_fields:
            norm = field.lower().strip()
            if norm not in resolved_set:
                q_text = get_question_for_field(field, category)
                candidates.append(ClarificationQuestion(field=norm, question_text=q_text))
                if len(candidates) >= MAX_QUESTIONS_PER_ROUND:
                    break

    return tuple(candidates[:MAX_QUESTIONS_PER_ROUND])


def start_clarification_session(
    case_id: str,
    tenant_id: str,
    conversation_id: str,
    category: Category,
    missing_fields: Sequence[str],
    image_blocked: bool = False,
    now: datetime | None = None,
) -> ClarificationSession:
    current_time = now or datetime.now(timezone.utc)
    expires_at = current_time + timedelta(hours=EXPIRY_HOURS)

    questions = generate_round_questions(
        category=category,
        missing_fields=missing_fields,
        resolved_fields=(),
        already_asked_fields=(),
        image_blocked=image_blocked,
    )

    if not questions:
        questions = (
            ClarificationQuestion(
                field="location",
                question_text=get_question_for_field("location", category),
            ),
        )

    round_1 = ClarificationRound(
        round_number=1,
        questions=questions,
        asked_at=current_time,
    )

    return ClarificationSession(
        session_id=str(uuid4()),
        case_id=case_id,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        category=category,
        status=ClarificationStatus.WAITING_REPLY,
        rounds=(round_1,),
        current_round=1,
        missing_fields=tuple(f.lower().strip() for f in missing_fields),
        resolved_fields=(),
        image_fallback_active=image_blocked,
        created_at=current_time,
        updated_at=current_time,
        expires_at=expires_at,
    )


def submit_clarification_reply(
    session: ClarificationSession,
    reply_text: str,
    newly_resolved_fields: Sequence[str] = (),
    now: datetime | None = None,
) -> ClarificationSession:
    current_time = now or datetime.now(timezone.utc)

    if current_time > session.expires_at:
        return session.model_copy(
            update={
                "status": ClarificationStatus.UNRESOLVED_EXPIRED,
                "updated_at": current_time,
            }
        )

    all_resolved = set(session.resolved_fields) | {
        f.lower().strip() for f in newly_resolved_fields
    }
    if session.image_fallback_active and len(reply_text.strip()) > 10:
        all_resolved.add("image_fallback")

    current_round_obj = session.rounds[-1].model_copy(
        update={
            "reply_text": reply_text,
            "replied_at": current_time,
        }
    )
    updated_rounds = session.rounds[:-1] + (current_round_obj,)

    remaining_missing = tuple(
        f for f in session.missing_fields if f not in all_resolved
    )

    if not remaining_missing:
        return session.model_copy(
            update={
                "status": ClarificationStatus.RESOLVED,
                "rounds": updated_rounds,
                "resolved_fields": tuple(sorted(all_resolved)),
                "updated_at": current_time,
            }
        )

    if session.current_round >= MAX_ROUNDS:
        return session.model_copy(
            update={
                "status": ClarificationStatus.UNRESOLVED_LIMIT,
                "rounds": updated_rounds,
                "resolved_fields": tuple(sorted(all_resolved)),
                "updated_at": current_time,
            }
        )

    next_round_number = session.current_round + 1
    already_asked = [
        q.field for r in updated_rounds for q in r.questions
    ]
    next_questions = generate_round_questions(
        category=session.category,
        missing_fields=remaining_missing,
        resolved_fields=tuple(all_resolved),
        already_asked_fields=already_asked,
        image_blocked=session.image_fallback_active and "image_fallback" not in all_resolved,
    )

    if not next_questions:
        return session.model_copy(
            update={
                "status": ClarificationStatus.RESOLVED,
                "rounds": updated_rounds,
                "resolved_fields": tuple(sorted(all_resolved)),
                "updated_at": current_time,
            }
        )

    next_round_obj = ClarificationRound(
        round_number=next_round_number,
        questions=next_questions,
        asked_at=current_time,
    )

    return session.model_copy(
        update={
            "status": ClarificationStatus.WAITING_REPLY,
            "current_round": next_round_number,
            "rounds": updated_rounds + (next_round_obj,),
            "resolved_fields": tuple(sorted(all_resolved)),
            "updated_at": current_time,
        }
    )


def check_session_expiry(
    session: ClarificationSession,
    now: datetime | None = None,
) -> ClarificationSession:
    current_time = now or datetime.now(timezone.utc)
    if current_time > session.expires_at and session.status == ClarificationStatus.WAITING_REPLY:
        return session.model_copy(
            update={
                "status": ClarificationStatus.UNRESOLVED_EXPIRED,
                "updated_at": current_time,
            }
        )
    return session
