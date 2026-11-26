from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from services.monitor.models import ForwardPacket, MonitorCase


def _engagement(case: MonitorCase) -> dict[str, object]:
    return {
        "likes": case.post.raw.get("likes"),
        "views": case.post.raw.get("views"),
        "viral_score": case.viral_score,
    }


def write_forward_packet(
    case: MonitorCase,
    note: str,
    output_dir: Path,
) -> ForwardPacket:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{case.case_id}-{stamp}"
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"

    complaint = case.run.case
    category = complaint.triage.category.value if complaint is not None else None
    risk = complaint.triage.risk.value if complaint is not None else None
    location = complaint.extraction.location if complaint is not None else None
    source_intent = complaint.source_intent if complaint is not None else None
    trust = complaint.status if complaint is not None else None
    operator_note = note.strip() or case.notes.strip()

    lines = [
        f"# Forwarding Draft — {case.case_id}",
        "",
        "**DRAFT — NOT SENT**",
        "",
        f"- Case ID: `{case.case_id}`",
        f"- Department: {case.department or '-'}",
        f"- Trust status: {trust or '-'}",
        f"- Category: {category or '-'}",
        f"- Risk: {risk or '-'}",
        f"- Location: {location or '-'}",
        f"- Source intent: {source_intent or '-'}",
        f"- Author: @{case.post.author_handle or '-'}",
        f"- Posted at: {case.post.posted_at.isoformat() if case.post.posted_at else '-'}",
        f"- Source URL: {case.post.url or '-'}",
        f"- Engagement: {_engagement(case)}",
        "",
        "## Original post",
        "",
        case.post.text,
        "",
        "## Operator note",
        "",
        operator_note or "-",
        "",
        "## Reminder",
        "",
        "This is a local draft. Verify the location and jurisdiction before sending it "
        "through an official channel.",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "draft": True,
                "department": case.department,
                "trust_status": trust,
                "category": category,
                "risk": risk,
                "location": location,
                "source_intent": source_intent,
                "operator_note": operator_note,
                "post": case.post.model_dump(mode="json"),
                "engagement": _engagement(case),
                "created_at": now.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ForwardPacket(
        case_id=case.case_id,
        markdown_path=str(markdown_path),
        json_path=str(json_path),
        created_at=now,
    )
