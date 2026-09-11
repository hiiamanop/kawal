from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Category(StrEnum):
    ROAD = "ROAD"


class RiskLevel(StrEnum):
    MEDIUM = "MEDIUM"


class Sensitivity(StrEnum):
    NORMAL = "NORMAL"
    RESTRICTED = "RESTRICTED"


class ProcessingState(StrEnum):
    ASSEMBLING = "ASSEMBLING"
    READY = "READY"
    ANALYZING = "ANALYZING"
    EXECUTING = "EXECUTING"
    TICKETED = "TICKETED"


class DecisionMode(StrEnum):
    EXECUTE = "EXECUTE"


class TicketStatus(StrEnum):
    SUBMITTED = "SUBMITTED"


class TicketPriority(StrEnum):
    NORMAL = "NORMAL"


class TicketVisibility(StrEnum):
    NORMAL = "NORMAL"
    RESTRICTED = "RESTRICTED"


class MessageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_message_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=64_000)
    offset_seconds: int = Field(ge=0)


class ReplayFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=128)
    bubbles: tuple[MessageInput, ...] = Field(min_length=1, max_length=3)


class RawMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    tenant_id: str
    conversation_id: str
    source_message_id: str
    text: str
    received_at: datetime


class CaseSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    tenant_id: str
    conversation_id: str
    revision: int = Field(ge=1)
    state: ProcessingState
    messages: tuple[RawMessage, ...]
    evidence_hash: str
    created_at: datetime


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: Category
    risk: RiskLevel
    sensitivity: Sensitivity
    jurisdiction_id: str
    authority_unit_id: str
    missing_fields: tuple[str, ...]
    has_mandatory_evidence: bool


class PolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    case_id: str
    revision: int = Field(ge=1)
    category: Category
    jurisdiction_id: str
    authority_unit_id: str
    missing_fields: tuple[str, ...]
    has_mandatory_evidence: bool
    idempotency_key: str


class PolicyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["ALLOW", "DENY"]
    rule_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    bundle_version: str = "m1.v1"
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    case_id: str
    revision: int = Field(ge=1)
    mode: DecisionMode
    reason_codes: tuple[str, ...]
    created_at: datetime


class TicketCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    case_id: str
    case_revision: int = Field(ge=1)
    category: Category
    priority: TicketPriority
    visibility: TicketVisibility
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=64_000)
    jurisdiction_id: str
    authority_unit_id: str
    source_decision_id: str
    payload_hash: str


class TicketReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket_id: str
    external_id: str
    status: TicketStatus
    idempotency_key: str
    payload_hash: str
    created_at: datetime


class OperationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["COMPLETED"]
    receipt: TicketReceipt


class TicketCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: DecisionRecord
    request: TicketCreateRequest
    idempotency_key: str

    @model_validator(mode="after")
    def stable_key_matches_request(self) -> TicketCommand:
        expected = f"{self.request.tenant_id}:{self.request.case_id}:ticket:create:v1"
        if self.idempotency_key != expected:
            raise ValueError("idempotency_key must use the canonical ticket-create format")
        return self
