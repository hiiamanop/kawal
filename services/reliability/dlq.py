from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DeadLetterEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_id: str
    source_ref: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    attempt_count: int = Field(ge=1)
    quarantined_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeadLetterQueue:
    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: dict[str, DeadLetterEntry] = {}

    def enqueue(
        self,
        source_ref: str,
        case_id: str,
        tenant_id: str,
        reason: str,
        error_type: str,
        error_message: str,
        payload: dict[str, Any] | None = None,
        attempt_count: int = 1,
    ) -> DeadLetterEntry:
        with self._lock:
            entry_id = str(uuid4())
            entry = DeadLetterEntry(
                entry_id=entry_id,
                source_ref=source_ref,
                case_id=case_id,
                tenant_id=tenant_id,
                reason=reason,
                error_type=error_type,
                error_message=error_message,
                payload=payload or {},
                attempt_count=attempt_count,
            )
            self._entries[entry_id] = entry
            return entry

    def get(self, entry_id: str) -> DeadLetterEntry | None:
        with self._lock:
            return self._entries.get(entry_id)

    def list_entries(self) -> tuple[DeadLetterEntry, ...]:
        with self._lock:
            return tuple(self._entries.values())

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
