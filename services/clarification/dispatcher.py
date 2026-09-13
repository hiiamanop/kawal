from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Sequence

from contracts.models import (
    Category,
    DeliveryStatus,
    OutboundMessageCommand,
    OutboundPurpose,
    OutboundSendReceipt,
    RawMessage,
)
from services.clarification.engine import (
    ClarificationSession,
    ClarificationStatus,
    start_clarification_session,
    submit_clarification_reply,
)
from services.intake.openwa import OpenWAConnector


def format_clarification_message(session: ClarificationSession) -> str:
    """Format the current round's questions politely for WhatsApp delivery."""
    current_round = session.rounds[-1]
    lines = [
        "Halo, terima kasih telah menghubungi layanan aduan.",
        "Agar laporan Anda dapat segera ditindaklanjuti oleh petugas, mohon bantu kami melengkapi informasi berikut:",
    ]
    for idx, q in enumerate(current_round.questions, 1):
        lines.append(f"{idx}. {q.question_text}")
    lines.append("Balas pesan ini langsung dengan informasi tersebut. Terima kasih.")
    return "\n\n".join(lines)


class ClarificationDispatcher:
    def __init__(self, openwa_connector: OpenWAConnector | None = None) -> None:
        self._openwa_connector = openwa_connector or OpenWAConnector()

    @property
    def connector(self) -> OpenWAConnector:
        return self._openwa_connector

    def dispatch_clarification_request(
        self,
        session: ClarificationSession,
        quoted_source_message_id: str | None = None,
        connection: object = None,
    ) -> tuple[ClarificationSession, OutboundSendReceipt]:
        """Format and send clarification questions for the current round via WhatsApp with idempotency."""
        text = format_clarification_message(session)
        idempotency_key = (
            f"{session.tenant_id}:{session.conversation_id}:clarification:round_{session.current_round}"
        )
        payload_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        command = OutboundMessageCommand(
            tenant_id=session.tenant_id,
            conversation_id=session.conversation_id,
            case_id=session.case_id,
            recipient_phone=session.conversation_id,
            text=text,
            quoted_source_message_id=quoted_source_message_id,
            idempotency_key=idempotency_key,
            purpose=OutboundPurpose.CLARIFICATION,
            payload_hash=payload_hash,
        )

        receipt = self._openwa_connector.send_text(command, connection=connection)
        return session, receipt

    def handle_citizen_reply(
        self,
        session: ClarificationSession,
        reply_message: RawMessage,
        resolved_fields: Sequence[str] = (),
        now: datetime | None = None,
        connection: object = None,
    ) -> tuple[ClarificationSession, OutboundSendReceipt | None]:
        """Process incoming citizen reply, update session, and auto-dispatch next round if needed."""
        updated_session = submit_clarification_reply(
            session=session,
            reply_text=reply_message.text,
            newly_resolved_fields=resolved_fields,
            now=now,
        )

        # If still missing fields and within 3 rounds, automatically dispatch the next round questions
        if (
            updated_session.status == ClarificationStatus.WAITING_REPLY
            and updated_session.current_round > session.current_round
        ):
            _, next_receipt = self.dispatch_clarification_request(
                session=updated_session,
                quoted_source_message_id=reply_message.source_message_id,
                connection=connection,
            )
            return updated_session, next_receipt

        return updated_session, None
