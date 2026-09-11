from collections.abc import Mapping
from datetime import datetime
import hashlib
from typing import TYPE_CHECKING, Any

from contracts.models import RawMessage

if TYPE_CHECKING:
    from services.intake.service import IntakeService


def _compact_prefix(
    tenant_id: str, connector_id: str, account_id: str, max_len: int = 63
) -> str:
    candidate = f"{tenant_id}:{connector_id}:{account_id}"
    if len(candidate) <= max_len:
        return candidate

    avail = max_len - 2  # 2 colons
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

    res = f"{tenant_id[:t_len]}:{connector_id[:c_len]}:{account_id[:a_len]}"
    return res[:max_len]


def build_normalized_message_id(
    tenant_id: str, connector_id: str, account_id: str, source_message_id: str
) -> str:
    raw_id = f"{tenant_id}:{connector_id}:{account_id}:{source_message_id}"
    if len(raw_id) <= 128:
        return raw_id

    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    prefix = _compact_prefix(tenant_id, connector_id, account_id, max_len=63)
    message_id = f"{prefix}:{digest}"
    return message_id[:128]


class OpenWAConnector:
    def __init__(
        self, intake_service: Any, connector_id: str, account_id: str
    ) -> None:
        self._intake_service = intake_service
        self._connector_id = connector_id
        self._account_id = account_id

    @property
    def connector_id(self) -> str:
        return self._connector_id

    @property
    def account_id(self) -> str:
        return self._account_id

    def is_self_sent(self, payload: Mapping[str, Any]) -> bool:
        for key in (
            "fromMe",
            "from_me",
            "isMe",
            "is_me",
            "self_sent",
            "selfSent",
            "is_self",
        ):
            val = payload.get(key)
            if val is True:
                return True
            if isinstance(val, str) and val.strip().lower() in ("true", "1"):
                return True

        sender = payload.get("sender")
        if isinstance(sender, Mapping):
            for key in ("isMe", "is_me"):
                val = sender.get(key)
                if val is True:
                    return True
                if isinstance(val, str) and val.strip().lower() in ("true", "1"):
                    return True
            sender_id = sender.get("id") or sender.get("jid")
            if (
                isinstance(sender_id, str)
                and self._account_id
                and sender_id == self._account_id
            ):
                return True

        for key in ("from", "sender_id", "author"):
            val = payload.get(key)
            if isinstance(val, str) and self._account_id and val == self._account_id:
                return True

        msg_id = payload.get("id") or payload.get("source_message_id")
        if isinstance(msg_id, str) and msg_id.startswith("true_"):
            return True

        return False

    def is_receipt_only(self, payload: Mapping[str, Any]) -> bool:
        for key in ("receipt_only", "receiptOnly", "isReceipt", "is_receipt"):
            val = payload.get(key)
            if val is True:
                return True
            if isinstance(val, str) and val.strip().lower() in ("true", "1"):
                return True

        event = str(payload.get("event", "")).strip().lower()
        if event in (
            "ack",
            "onack",
            "message_ack",
            "receipt",
            "delivery",
            "delivery_receipt",
            "read",
            "read_receipt",
        ):
            return True

        msg_type = str(payload.get("type", "")).strip().lower()
        if msg_type in (
            "ack",
            "receipt",
            "delivery",
            "delivery_receipt",
            "read",
            "read_receipt",
            "status_receipt",
        ):
            return True

        subtype = str(payload.get("subtype", "")).strip().lower()
        if subtype in ("ack", "receipt", "delivery", "read"):
            return True

        has_text = bool(payload.get("text") or payload.get("body"))
        if not has_text:
            if payload.get("ack") is not None:
                return True
            status = str(payload.get("status", "")).strip().lower()
            if status in ("delivered", "read", "viewed", "played", "sent", "ack"):
                return True

        return False

    def is_group_chat(self, payload: Mapping[str, Any]) -> bool:
        for key in (
            "isGroupMsg",
            "isGroup",
            "is_group_msg",
            "is_group",
            "group_chat",
            "groupChat",
        ):
            val = payload.get(key)
            if val is True:
                return True
            if isinstance(val, str) and val.strip().lower() in ("true", "1"):
                return True

        for key in ("chat_id", "chatId", "conversation_id", "from", "to"):
            val = payload.get(key)
            if isinstance(val, str) and ("@g.us" in val or val.endswith("@temp")):
                return True

        msg_id = payload.get("id") or payload.get("source_message_id")
        if isinstance(msg_id, str) and "@g.us" in msg_id:
            return True

        chat = payload.get("chat")
        if isinstance(chat, Mapping):
            for key in ("isGroup", "is_group"):
                val = chat.get(key)
                if val is True:
                    return True
                if isinstance(val, str) and val.strip().lower() in ("true", "1"):
                    return True
            chat_id = chat.get("id") or chat.get("jid")
            if isinstance(chat_id, str) and "@g.us" in chat_id:
                return True

        return False

    def should_filter(self, payload: Mapping[str, Any]) -> bool:
        return (
            self.is_self_sent(payload)
            or self.is_receipt_only(payload)
            or self.is_group_chat(payload)
        )

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
        conversation_id = self._required(payload, "conversation_id", "chat_id", "from")
        text = self._required(payload, "text", "body")
        received_at = payload.get("received_at", payload.get("timestamp"))
        if isinstance(received_at, str):
            received_at = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
        if not isinstance(received_at, datetime):
            raise ValueError("OpenWA payload requires received_at")

        message_id = build_normalized_message_id(
            tenant_id, self._connector_id, self._account_id, source_message_id
        )

        return RawMessage(
            message_id=message_id,
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
        return self.receive(
            message,
            self._quoted_source_message_id(payload),
            payload.get("case_key"),
        )

    def receive(
        self,
        message: RawMessage,
        quoted_source_message_id: str | None = None,
        case_key: str | None = None,
    ) -> bool:
        return self._intake_service.accept(
            message,
            quoted_source_message_id,
            case_key,
            self._connector_id,
            self._account_id,
        )

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
        for key in (
            "quoted_source_message_id",
            "quoted_message_id",
            "quotedMessageId",
            "quotedMsgId",
        ):
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
