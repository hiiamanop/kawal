from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from contracts.models import (
    DeliveryStatus,
    OutboundMessageCommand,
    OutboundPurpose,
    RawMessage,
)
from services.intake.openwa import OpenWAConnector
from services.intake.send_ledger import IdempotencyConflictError, InMemorySendLedger


def _make_cmd(
    key: str = "tenant-bdg:conv-wa-1:clarification:r1",
    payload_hash: str = "1" * 64,
) -> OutboundMessageCommand:
    return OutboundMessageCommand(
        tenant_id="tenant-bdg",
        conversation_id="62811223344@c.us",
        case_id="case-wa-1",
        recipient_phone="62811223344@c.us",
        text="Mohon sebutkan nama jalan dan nomor rumah.",
        quoted_source_message_id="msg-citizen-1",
        idempotency_key=key,
        purpose=OutboundPurpose.CLARIFICATION,
        payload_hash=payload_hash,
    )


def test_openwa_send_text_success() -> None:
    connector = OpenWAConnector()
    cmd = _make_cmd()

    receipt = connector.send_text(cmd)
    assert receipt.status == DeliveryStatus.SENT
    assert receipt.idempotency_key == cmd.idempotency_key
    assert receipt.source_message_id is not None
    assert receipt.source_message_id.startswith("false_62811223344@c.us_")

    assert len(connector.sent_messages) == 1
    sent = connector.sent_messages[0]
    assert sent["to"] == cmd.recipient_phone
    assert sent["text"] == cmd.text
    assert sent["quoted_msg_id"] == "msg-citizen-1"


def test_openwa_send_text_idempotency_prevents_duplicate_whatsapp_send() -> None:
    connector = OpenWAConnector()
    cmd = _make_cmd()

    # First send
    receipt1 = connector.send_text(cmd)
    assert receipt1.status == DeliveryStatus.SENT
    assert len(connector.sent_messages) == 1

    # Second send with same idempotency key
    receipt2 = connector.send_text(cmd)
    assert receipt2.status == DeliveryStatus.SENT
    assert receipt2.send_id == receipt1.send_id
    assert receipt2.source_message_id == receipt1.source_message_id

    # Crucial: transport was called ONLY ONCE! Zero duplicate messages over WhatsApp.
    assert len(connector.sent_messages) == 1


def test_openwa_send_text_conflict_on_mutated_payload() -> None:
    connector = OpenWAConnector()
    cmd = _make_cmd(payload_hash="a" * 64)
    connector.send_text(cmd)

    mutated = cmd.model_copy(update={"payload_hash": "b" * 64})
    with pytest.raises(IdempotencyConflictError):
        connector.send_text(mutated)

    assert len(connector.sent_messages) == 1


def test_openwa_send_text_timeout_records_unknown_and_prevents_blind_resend() -> None:
    """PRD §42: Unknown outcome reconciliation.

    When network times out, mark DELIVERY_UNKNOWN and do not blind resend.
    """
    call_count = 0

    def timeout_transport(**kwargs: object) -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        raise TimeoutError("Network timed out reaching WhatsApp gateway")

    connector = OpenWAConnector(transport=timeout_transport)
    cmd = _make_cmd(key="tenant-bdg:conv-wa-timeout:clarification:r1")

    # Send attempt times out
    receipt = connector.send_text(cmd)
    assert receipt.status == DeliveryStatus.DELIVERY_UNKNOWN
    assert receipt.source_message_id is None
    assert call_count == 1

    # Immediate retry attempt checks ledger and detects DELIVERY_UNKNOWN:
    # Does NOT blindly call transport again!
    receipt_retry = connector.send_text(cmd)
    assert receipt_retry.status == DeliveryStatus.DELIVERY_UNKNOWN
    assert call_count == 1  # Not incremented!
