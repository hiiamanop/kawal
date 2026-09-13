from datetime import datetime, timezone
from hashlib import sha256
import json
from uuid import uuid4

from contracts.models import (
    AnalysisResult,
    CaseSnapshot,
    Category,
    DecisionMode,
    DecisionRecord,
    PolicyInput,
    TicketCommand,
    TicketCreateRequest,
    TicketPriority,
    TicketVisibility,
)
from services.core.policy import evaluate_ticket_creation


CATEGORY_TITLES: dict[Category, str] = {
    Category.ROAD: "Aduan jalan rusak",
    Category.DRAINAGE_FLOOD: "Aduan genangan dan drainase",
    Category.WASTE: "Aduan sampah",
    Category.CLEAN_WATER: "Aduan air bersih",
    Category.CIVIL_ADMIN: "Aduan administrasi kependudukan",
    Category.HEALTH_SERVICE: "Aduan layanan kesehatan",
    Category.PUBLIC_ORDER: "Aduan ketertiban umum",
    Category.TRANSPORTATION: "Aduan sarana transportasi",
    Category.FIRE_RESCUE: "Aduan pemadam kebakaran",
    Category.SOCIAL_AFFAIRS: "Aduan masalah sosial",
    Category.EDUCATION: "Aduan sarana pendidikan",
    Category.PARKS_HOUSING: "Aduan taman dan perumahan",
}


def build_ticket_command(
    snapshot: CaseSnapshot,
    analysis: AnalysisResult,
    title: str | None = None,
) -> TicketCommand | None:
    idempotency_key = f"{snapshot.tenant_id}:{snapshot.case_id}:ticket:create:v1"
    policy = evaluate_ticket_creation(
        PolicyInput(
            tenant_id=snapshot.tenant_id,
            case_id=snapshot.case_id,
            revision=snapshot.revision,
            category=analysis.category,
            jurisdiction_id=analysis.jurisdiction_id,
            authority_unit_id=analysis.authority_unit_id,
            missing_fields=analysis.missing_fields,
            has_mandatory_evidence=analysis.has_mandatory_evidence,
            idempotency_key=idempotency_key,
        )
    )
    if policy.decision != "ALLOW":
        return None

    resolved_title = title or CATEGORY_TITLES.get(analysis.category, f"Aduan {analysis.category.value}")
    description = "\n".join(message.text for message in snapshot.messages)
    payload = {
        "tenant_id": snapshot.tenant_id,
        "case_id": snapshot.case_id,
        "case_revision": snapshot.revision,
        "category": analysis.category,
        "priority": TicketPriority.NORMAL,
        "visibility": TicketVisibility.NORMAL,
        "title": resolved_title,
        "description": description,
        "jurisdiction_id": analysis.jurisdiction_id,
        "authority_unit_id": analysis.authority_unit_id,
    }
    payload_hash = sha256(
        json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    decision = DecisionRecord(
        decision_id=str(uuid4()),
        case_id=snapshot.case_id,
        revision=snapshot.revision,
        mode=DecisionMode.EXECUTE,
        reason_codes=policy.reason_codes,
        created_at=datetime.now(timezone.utc),
    )
    request = TicketCreateRequest(
        **payload,
        source_decision_id=decision.decision_id,
        payload_hash=payload_hash,
    )
    return TicketCommand(
        decision=decision,
        request=request,
        idempotency_key=idempotency_key,
    )
