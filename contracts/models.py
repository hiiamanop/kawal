from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema, core_schema


class Category(StrEnum):
    ROAD = "ROAD"
    DRAINAGE_FLOOD = "DRAINAGE_FLOOD"
    WASTE = "WASTE"
    CLEAN_WATER = "CLEAN_WATER"
    CIVIL_ADMIN = "CIVIL_ADMIN"
    HEALTH_SERVICE = "HEALTH_SERVICE"
    PUBLIC_ORDER = "PUBLIC_ORDER"
    TRANSPORTATION = "TRANSPORTATION"
    FIRE_RESCUE = "FIRE_RESCUE"
    SOCIAL_AFFAIRS = "SOCIAL_AFFAIRS"
    EDUCATION = "EDUCATION"
    PARKS_HOUSING = "PARKS_HOUSING"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class Completeness(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    INCOMPLETE = "INCOMPLETE"
    AMBIGUOUS = "AMBIGUOUS"


CompletenessLevel = Completeness


class ResultStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    TIMED_OUT = "TIMED_OUT"


AgentResultStatus = ResultStatus


class UncertaintyMethod(StrEnum):
    ENTROPY = "ENTROPY"
    MARGIN = "MARGIN"
    INTERVAL = "INTERVAL"
    RULE = "RULE"
    UNKNOWN = "UNKNOWN"


class EvidenceKind(StrEnum):
    TEXT_SPAN = "TEXT_SPAN"
    IMAGE = "IMAGE"
    DATABASE = "DATABASE"
    RULE = "RULE"


class CostKind(StrEnum):
    ACTUAL = "ACTUAL"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


TelemetryCostKind = CostKind


class Sensitivity(StrEnum):
    NORMAL = "NORMAL"
    RESTRICTED = "RESTRICTED"


class ProcessingState(StrEnum):
    ASSEMBLING = "ASSEMBLING"
    READY = "READY"
    ANALYZING = "ANALYZING"
    WAITING_RESULTS = "WAITING_RESULTS"
    WAITING_CLARIFICATION = "WAITING_CLARIFICATION"
    EXECUTING = "EXECUTING"
    TICKETED = "TICKETED"
    REJECTED = "REJECTED"
    WAITING_DEPENDENCY = "WAITING_DEPENDENCY"
    BLOCKED = "BLOCKED"
    UNRESOLVED = "UNRESOLVED"


class DecisionMode(StrEnum):
    EXECUTE = "EXECUTE"
    RE_EVALUATE = "RE_EVALUATE"
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    REJECT_IGNORE = "REJECT_IGNORE"


class TicketStatus(StrEnum):
    SUBMITTED = "SUBMITTED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class TicketPriority(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"


class TicketVisibility(StrEnum):
    NORMAL = "NORMAL"
    RESTRICTED = "RESTRICTED"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class CanonicalSpanLabel(StrEnum):
    OBJ = "OBJ"
    LOC = "LOC"
    TIME = "TIME"


class CanonicalSpan(tuple):
    _fields = ("start", "end", "label")

    def __new__(cls, start: Any = None, end: Any = None, label: Any = None, **kwargs: Any) -> CanonicalSpan:
        if kwargs:
            extra = set(kwargs.keys()) - {"start", "end", "label"}
            if extra:
                raise ValueError(f"Extra fields not permitted in CanonicalSpan: {sorted(extra)}")
            if "start" in kwargs:
                start = kwargs["start"]
            if "end" in kwargs:
                end = kwargs["end"]
            if "label" in kwargs:
                label = kwargs["label"]

        if isinstance(start, (list, tuple)) and end is None and label is None:
            if len(start) != 3:
                raise ValueError(
                    f"Canonical span sequence must have exactly 3 elements [start, end, label], got {len(start)}"
                )
            start, end, label = start[0], start[1], start[2]
        elif isinstance(start, dict) and end is None and label is None:
            extra = set(start.keys()) - {"start", "end", "label"}
            if extra:
                raise ValueError(f"Extra fields not permitted in CanonicalSpan: {sorted(extra)}")
            if "start" not in start or "end" not in start or "label" not in start:
                raise ValueError(f"Canonical span dict must contain start, end, label: {start}")
            start, end, label = start["start"], start["end"], start["label"]

        if start is None or end is None or label is None:
            raise ValueError("CanonicalSpan requires start, end, and label")

        if not isinstance(start, int) or isinstance(start, bool):
            try:
                start = int(start)
            except Exception:
                raise ValueError(f"start offset must be an integer, got {start!r}")
        if not isinstance(end, int) or isinstance(end, bool):
            try:
                end = int(end)
            except Exception:
                raise ValueError(f"end offset must be an integer, got {end!r}")

        label_str = label.value if hasattr(label, "value") else str(label)
        if label_str not in ("OBJ", "LOC", "TIME"):
            raise ValueError(f"label must be one of OBJ, LOC, TIME; got {label!r}")

        if start < 0:
            raise ValueError(f"start offset must be non-negative, got {start}")
        if end < 0:
            raise ValueError(f"end offset must be non-negative, got {end}")
        if start >= end:
            raise ValueError(f"start ({start}) must be strictly less than end ({end})")

        return super().__new__(cls, (start, end, CanonicalSpanLabel(label_str)))

    @property
    def start(self) -> int:
        return self[0]

    @property
    def end(self) -> int:
        return self[1]

    @property
    def label(self) -> CanonicalSpanLabel:
        return self[2]

    def _asdict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "label": self.label.value}

    def __repr__(self) -> str:
        return f"CanonicalSpan(start={self.start}, end={self.end}, label={self.label.value!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def validate(value: Any) -> CanonicalSpan:
            if isinstance(value, CanonicalSpan):
                return value
            if isinstance(value, (list, tuple)):
                if len(value) != 3:
                    raise ValueError(
                        f"Canonical span sequence must have exactly 3 elements [start, end, label], got {len(value)}"
                    )
                return cls(value[0], value[1], value[2])
            if isinstance(value, dict):
                return cls(value)
            raise ValueError(f"Unsupported canonical span input: {value!r}")

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: [v.start, v.end, v.label.value] if isinstance(v, CanonicalSpan) else list(v)
            ),
        )


class TrajectoryBubble(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_message_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=64_000)
    offset_seconds: int = Field(default=0, ge=0)
    canonical_spans: tuple[CanonicalSpan, ...] = Field(default_factory=tuple)


class TurnExpectedAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn: int = Field(ge=1)
    allowed_actions: tuple[DecisionMode, ...]
    missing: tuple[str, ...] = Field(default_factory=tuple)
    strategy: str | None = None


class TrajectoryTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn: int = Field(ge=1)
    bubbles: tuple[TrajectoryBubble, ...]
    observable_facts: tuple[str, ...] = Field(default_factory=tuple)
    hidden_facts: tuple[str, ...] = Field(default_factory=tuple)
    expected_action: TurnExpectedAction


class ComplaintTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1, max_length=128)
    family_id: str = Field(min_length=1, max_length=128)
    split: DatasetSplit
    category: Category
    world_truth: dict[str, Any] = Field(default_factory=dict)
    observable_facts: tuple[str, ...] = Field(default_factory=tuple)
    hidden_facts: tuple[str, ...] = Field(default_factory=tuple)
    location_completeness: str = "COMPLETE"
    duration: str = "UNKNOWN"
    claim_certainty: str = "FIRST_HAND"
    persona: str = "STANDARD"
    noise: dict[str, Any] = Field(default_factory=dict)
    attachment_role: str = "NONE"
    turns: tuple[TrajectoryTurn, ...]
    expected_action_by_turn: tuple[TurnExpectedAction, ...] = Field(default_factory=tuple)
    canonical_spans: tuple[CanonicalSpan, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def sync_expected_action_by_turn(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "expected_action_by_turn" not in data and "turns" in data:
                turns = data["turns"]
                actions = []
                for t in turns:
                    if isinstance(t, dict) and "expected_action" in t:
                        actions.append(t["expected_action"])
                    elif hasattr(t, "expected_action"):
                        actions.append(t.expected_action)
                data["expected_action_by_turn"] = tuple(actions)
        return data


class FamilySplitAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    total_trajectories: int = Field(ge=0)
    total_families: int = Field(ge=0)
    split_trajectories: dict[str, int] = Field(default_factory=dict)
    split_families: dict[str, int] = Field(default_factory=dict)
    family_overlap: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    oracle_leakages: tuple[str, ...] = Field(default_factory=tuple)
    cross_split_text_duplicates: tuple[str, ...] = Field(default_factory=tuple)
    violations: tuple[str, ...] = Field(default_factory=tuple)


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


class AttachmentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    object_key: str = Field(min_length=1, max_length=512)
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    size_bytes: int = Field(gt=0, le=10 * 1024 * 1024)
    content_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    received_at: datetime
    storage_bucket: Literal["attachments"] = "attachments"
    is_private: Literal[True] = True
    retention_class: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_received_at(self) -> "AttachmentMetadata":
        if self.received_at.tzinfo is None:
            raise ValueError("received_at must include a timezone")
        return self


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


class UncertaintyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: UncertaintyMethod
    value: float | None = None
    calibrated: bool


ResultUncertainty = UncertaintyInfo
AgentUncertainty = UncertaintyInfo


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: EvidenceKind
    ref: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> EvidenceItem:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end offset must be greater than or equal to start offset")
        return self


ResultEvidence = EvidenceItem
AgentEvidence = EvidenceItem


class VersionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    preprocess: str = Field(min_length=1)
    calibration: str | None = None
    prompt: str | None = None


ResultVersions = VersionInfo
AgentVersions = VersionInfo


class TelemetryInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_ms: float = Field(ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    cost_kind: CostKind = CostKind.UNKNOWN
    attempt: int = Field(default=1, ge=1)


ResultTelemetry = TelemetryInfo
AgentTelemetry = TelemetryInfo


class AgentResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    input_revision: int = Field(ge=1)
    input_hash: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    task: str = Field(min_length=1)
    status: ResultStatus
    data: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: UncertaintyInfo
    evidence: tuple[EvidenceItem, ...] = Field(default_factory=tuple)
    versions: VersionInfo
    telemetry: TelemetryInfo
    error: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_status_data_error(self) -> AgentResultV1:
        if self.status == ResultStatus.SUCCEEDED:
            if self.data is None:
                raise ValueError("data must be non-null when status is SUCCEEDED")
            if self.error is not None:
                raise ValueError("error must be null when status is SUCCEEDED")
        else:
            if self.data is not None:
                raise ValueError("data must be null when status is not SUCCEEDED")
            if self.confidence is not None:
                raise ValueError("confidence must be null when status is not SUCCEEDED")
            if self.error is None:
                raise ValueError("error must be non-null when status is not SUCCEEDED")
        return self


AgentResult = AgentResultV1


class DeliveryStatus(StrEnum):
    PENDING = "PENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    DELIVERY_UNKNOWN = "DELIVERY_UNKNOWN"
    FAILED = "FAILED"


class OutboundPurpose(StrEnum):
    CLARIFICATION = "CLARIFICATION"
    STATUS_UPDATE = "STATUS_UPDATE"
    RECEIPT = "RECEIPT"


class OutboundMessageCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    case_id: str | None = None
    recipient_phone: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4096)
    quoted_source_message_id: str | None = None
    idempotency_key: str = Field(min_length=1)
    purpose: OutboundPurpose = OutboundPurpose.CLARIFICATION
    payload_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class OutboundSendReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    send_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    source_message_id: str | None = None
    status: DeliveryStatus
    sent_at: datetime
    payload_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
