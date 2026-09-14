from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import TYPE_CHECKING, Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from contracts.models import (
    DeliveryStatus,
    OutboundMessageCommand,
    OutboundSendReceipt,
    RawMessage,
)
from services.intake.send_ledger import (
    IdempotencyConflictError,
    InMemorySendLedger,
)

if TYPE_CHECKING:
    from services.intake.service import IntakeService


class OpenWAApiError(RuntimeError):
    pass


class OpenWAHttpTransport:
    """HTTP adapter for a running OpenWA API Gateway session."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        session_id: str,
        timeout_seconds: float = 15.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("OpenWA base_url must use http or https")
        if not api_key:
            raise ValueError("OpenWA api_key is required")
        if not session_id:
            raise ValueError("OpenWA session_id is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> OpenWAHttpTransport:
        return cls(
            base_url=os.environ["KAWAL_OPENWA_BASE_URL"],
            api_key=os.environ["KAWAL_OPENWA_API_KEY"],
            session_id=os.environ["KAWAL_OPENWA_SESSION_ID"],
            timeout_seconds=float(os.getenv("KAWAL_OPENWA_TIMEOUT_SECONDS", "15")),
        )

    def send_text(
        self,
        to: str,
        text: str,
        quoted_msg_id: str | None = None,
    ) -> Mapping[str, Any]:
        body: dict[str, Any] = {"chatId": to, "text": text}
        if quoted_msg_id:
            body["quotedMessageId"] = quoted_msg_id
        req = Request(
            url=f"{self.base_url}/api/sessions/{self.session_id}/messages/send-text",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": self.api_key,
            },
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenWAApiError(f"OpenWA HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise TimeoutError(f"OpenWA transport unavailable: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenWAApiError("OpenWA returned invalid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise OpenWAApiError("OpenWA returned non-object response")
        return parsed


def _compact_prefix(
    tenant_id: str, connector_id: str, account_id: str, max_len: int = 63
) -> str:
    candidate = f"{tenant_id}:{connector_id}:{account_id}"
    if len(candidate) <= max_len:
        return candidate

    avail = max_len - 2
    if avail <= 0:
        return candidate[:max_len]

    t_len = min(len(tenant_id), max(1, min(avail // 4, 15)))
    remaining = avail - t_len
    half = remaining // 2
    if len(connector_id) <= half:
        c_len = len(connector_id)
        a_len = min(len(account_id), remaining - c_len)
    elif len(account_id) <= (remaining - half):
        a_len = len(account_id)
        c_len = min(len(connector_id), remaining - a_len)
    else:
        c_len = half
        a_len = remaining - c_len

    return f"{tenant_id[:t_len]}:{connector_id[:c_len]}:{account_id[:a_len]}"[:max_len]


def build_normalized_message_id(
    tenant_id: str, connector_id: str, account_id: str, source_message_id: str
) -> str:
    raw_id = f"{tenant_id}:{connector_id}:{account_id}:{source_message_id}"
    if len(raw_id) <= 128:
        return raw_id
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    prefix = _compact_prefix(tenant_id, connector_id, account_id, max_len=63)
    return f"{prefix}:{digest}"[:128]


class OpenWAConnector:
    def __init__(
        self,
        intake_service: Any = None,
        connector_id: str = "openwa",
        account_id: str = "research",
        send_ledger: Any | None = None,
        transport: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self._intake_service = intake_service
        self._connector_id = connector_id
        self._account_id = account_id
        self._send_ledger = send_ledger or InMemorySendLedger()
        self._transport = transport or self._default_mock_transport
        self._sent_messages: list[dict[str, Any]] = []

    @classmethod
    def live_from_environment(
        cls,
        intake_service: Any,
        account_id: str,
        send_ledger: Any,
    ) -> OpenWAConnector:
        transport = OpenWAHttpTransport.from_environment()
        return cls(
            intake_service=intake_service,
            connector_id="openwa",
            account_id=account_id,
            send_ledger=send_ledger,
            transport=transport.send_text,
        )

    def _default_mock_transport(
        self, to: str, text: str, quoted_msg_id: str | None = None
    ) -> Mapping[str, Any]:
        msg_id = f"false_{to}_{uuid4().hex[:12]}"
        record = {
            "id": msg_id,
            "to": to,
            "text": text,
            "quoted_msg_id": quoted_msg_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._sent_messages.append(record)
        return {"id": msg_id, "status": "SENT"}

    @property
    def sent_messages(self) -> list[dict[str, Any]]:
        return list(self._sent_messages)

    @property
    def connector_id(self) -> str:
        return self._connector_id

    @property
    def account_id(self) -> str:
        return self._account_id

    def send_text(
        self, command: OutboundMessageCommand, connection: Any = None
    ) -> OutboundSendReceipt:
        send_id, is_new = self._send_ledger.record_pending_send(connection, command)
        if not is_new:
            existing = self._send_ledger.lookup_by_idempotency_key(
                connection, command.idempotency_key
            )
            if existing:
                sent_at = existing.get("updated_at") or datetime.now(timezone.utc)
                if isinstance(sent_at, str):
                    sent_at = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
                return OutboundSendReceipt(
                    send_id=send_id,
                    idempotency_key=command.idempotency_key,
                    source_message_id=existing.get("source_message_id"),
                    status=DeliveryStatus(existing.get("status", DeliveryStatus.SENT.value)),
                    sent_at=sent_at,
                    payload_hash=command.payload_hash,
                )

        now = datetime.now(timezone.utc)
        try:
            response = self._transport(
                to=command.recipient_phone,
                text=command.text,
                quoted_msg_id=command.quoted_source_message_id,
            )
            source_message_id = str(
                response.get("id")
                or response.get("messageId")
                or f"false_{command.recipient_phone}_{uuid4().hex[:12]}"
            )
            self._send_ledger.mark_sent(connection, send_id, source_message_id)
            return OutboundSendReceipt(
                send_id=send_id,
                idempotency_key=command.idempotency_key,
                source_message_id=source_message_id,
                status=DeliveryStatus.SENT,
                sent_at=now,
                payload_hash=command.payload_hash,
            )
        except TimeoutError as exc:
            self._send_ledger.mark_unknown_outcome(connection, send_id, str(exc))
            return OutboundSendReceipt(
                send_id=send_id,
                idempotency_key=command.idempotency_key,
                source_message_id=None,
                status=DeliveryStatus.DELIVERY_UNKNOWN,
                sent_at=now,
                payload_hash=command.payload_hash,
            )
        except Exception as exc:
            self._send_ledger.mark_failed(connection, send_id, str(exc))
            raise

    def is_self_sent(self, payload: Mapping[str, Any]) -> bool:
        for key in ("fromMe", "from_me", "isMe", "is_me", "self_sent", "selfSent", "is_self"):
            value = payload.get(key)
            if value is True or (isinstance(value, str) and value.strip().lower() in ("true", "1")):
                return True
        sender = payload.get("sender")
        if isinstance(sender, Mapping):
            for key in ("isMe", "is_me"):
                value = sender.get(key)
                if value is True or (isinstance(value, str) and value.strip().lower() in ("true", "1")):
                    return True
            sender_id = sender.get("id") or sender.get("jid")
            if isinstance(sender_id, str) and self._account_id and sender_id == self._account_id:
                return True
        for key in ("from", "sender_id", "author"):
            value = payload.get(key)
            if isinstance(value, str) and self._account_id and value == self._account_id:
                return True
        msg_id = payload.get("id") or payload.get("source_message_id")
        return isinstance(msg_id, str) and msg_id.startswith("true_")

    def is_receipt_only(self, payload: Mapping[str, Any]) -> bool:
        for key in ("receipt_only", "receiptOnly", "isReceipt", "is_receipt"):
            value = payload.get(key)
            if value is True or (isinstance(value, str) and value.strip().lower() in ("true", "1")):
                return True
        event = str(payload.get("event", "")).strip().lower()
        if event in ("ack", "onack", "message_ack", "receipt", "delivery", "delivery_receipt", "read", "read_receipt"):
            return True
        msg_type = str(payload.get("type", "")).strip().lower()
        if msg_type in ("ack", "receipt", "delivery", "delivery_receipt", "read", "read_receipt", "status_receipt"):
            return True
        if not (payload.get("text") or payload.get("body")):
            if payload.get("ack") is not None:
                return True
            if str(payload.get("status", "")).strip().lower() in ("delivered", "read", "viewed", "played", "sent", "ack"):
                return True
        return False

    def is_group_chat(self, payload: Mapping[str, Any]) -> bool:
        for key in ("isGroupMsg", "isGroup", "is_group_msg", "is_group", "group_chat", "groupChat"):
            value = payload.get(key)
            if value is True or (isinstance(value, str) and value.strip().lower() in ("true", "1")):
                return True
        chat = payload.get("chat")
        if isinstance(chat, Mapping) and self.is_group_chat(chat):
            return True
        for key in ("chat_id", "chatId", "conversation_id", "from", "to", "id"):
            value = payload.get(key)
            if isinstance(value, str) and ("@g.us" in value or value.endswith("@temp")):
                return True
        return False

    def should_filter(self, payload: Mapping[str, Any]) -> bool:
        return self.is_self_sent(payload) or self.is_receipt_only(payload) or self.is_group_chat(payload)

    is_filtered = should_filter

    def normalize(self, payload: Mapping[str, Any]) -> RawMessage:
        if self.is_self_sent(payload):
            raise ValueError("OpenWA adapter ignores self-sent messages per PRD §17")
        if self.is_receipt_only(payload):
            raise ValueError("OpenWA adapter ignores receipt-only events per PRD §17")
        if self.is_group_chat(payload):
            raise ValueError("OpenWA adapter ignores group chat events per PRD §17")
        source_message_id = self._required(payload, "source_message_id", "id")
        tenant_id = self._required(payload, "tenant_id")
        conversation_id = self._required(payload, "conversation_id", "chat_id", "chatId", "from")
        text = self._required(payload, "text", "body")
        received_at = payload.get("received_at", payload.get("timestamp"))
        if isinstance(received_at, str):
            received_at = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
        if not isinstance(received_at, datetime):
            raise ValueError("OpenWA payload requires received_at")
        return RawMessage(
            message_id=build_normalized_message_id(tenant_id, self._connector_id, self._account_id, source_message_id),
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            text=text,
            received_at=received_at,
        )

    def callback(self, payload: Mapping[str, Any]) -> bool:
        if self.should_filter(payload):
            return False
        message = self.normalize(payload)
        return self.receive(message, self._quoted_source_message_id(payload), payload.get("case_key"))

    def receive(self, message: RawMessage, quoted_source_message_id: str | None = None, case_key: str | None = None) -> bool:
        if self._intake_service is None:
            raise RuntimeError("OpenWA intake_service is not configured")
        return self._intake_service.accept(message, quoted_source_message_id, case_key, self._connector_id, self._account_id)

    def health_check(self) -> bool:
        return True

    @staticmethod
    def _required(payload: Mapping[str, Any], *keys: str) -> str:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        raise ValueError(f"OpenWA payload requires one of: {', '.join(keys)}")

    @staticmethod
    def _quoted_source_message_id(payload: Mapping[str, Any]) -> str | None:
        for key in ("quoted_source_message_id", "quoted_message_id", "quotedMessageId", "quotedMsgId"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        quoted = payload.get("quotedMessage")
        if isinstance(quoted, Mapping):
            for key in ("source_message_id", "id", "_serialized"):
                value = quoted.get(key)
                if isinstance(value, str) and value:
                    return value
        return None
