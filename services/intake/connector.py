from datetime import datetime
from typing import Protocol

from contracts.models import RawMessage


class IntakeConnector(Protocol):
    def normalize(self, payload: object) -> RawMessage: ...

    def receive(
        self,
        message: RawMessage,
        quoted_source_message_id: str | None = None,
        case_key: str | None = None,
    ) -> bool: ...

    def health_check(self) -> bool: ...


class OfflineReplayConnector:
    def message(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        source_message_id: str,
        text: str,
        received_at: datetime,
    ) -> RawMessage:
        return RawMessage(
            message_id=f"{tenant_id}:{source_message_id}",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            text=text,
            received_at=received_at,
        )

    def health_check(self) -> bool:
        return True
