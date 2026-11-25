from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import Category, RiskLevel

LocationQuality = Literal["MISSING", "VAGUE", "SPECIFIC"]
SourceIntent = Literal[
    "CITIZEN_COMPLAINT",
    "NEWS_REPORT",
    "OPINION",
    "QUESTION",
    "ANNOUNCEMENT",
    "UNKNOWN",
]


class SocialPost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    post_id: str = Field(min_length=1)
    platform: Literal["x"] = "x"
    url: str = ""
    author_handle: str = ""
    text: str
    media_urls: tuple[str, ...] = ()
    posted_at: datetime | None = None
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw: dict[str, Any] = Field(default_factory=dict)


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_complaint: bool
    category: Category
    risk: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: tuple[str, ...] = ()


class ExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    location: str | None = None
    object: str | None = None
    time: str | None = None
    evidence_spans: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class VerificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_complaint: bool
    category: Category
    evidence_supported: bool
    issues: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class ArbitrationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_complaint: bool
    category: Category
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class AgentUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    provider: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    cost_source: Literal["provider", "estimated", "local"]
    latency_ms: float = Field(default=0.0, ge=0.0)


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    status: Literal["OK", "INVALID", "UNAVAILABLE", "BUDGET_EXCEEDED"]
    output: dict[str, Any] | None = None
    usage: AgentUsage
    prompt_version: str = Field(min_length=1)
    error: str | None = None


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str = Field(min_length=1)
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class CostReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    calls: int = Field(default=0, ge=0)
    arbiter_calls: int = Field(default=0, ge=0)
    by_role: dict[str, float] = Field(default_factory=dict)
    budget_status: Literal["OK", "EXCEEDED"] = "OK"


class ComplaintCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    post: SocialPost
    triage: TriageOutput
    extraction: ExtractionOutput
    verification: VerificationOutput
    arbitration: ArbitrationOutput | None = None
    location_quality: LocationQuality | None = None
    source_intent: SourceIntent | None = None
    trust_decision: Literal["AGREED", "NEEDS_REVIEW", "REJECTED"]
    status: Literal["ACCEPTED", "NEEDS_REVIEW", "REJECTED"]
    reason_codes: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HarnessRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    post: SocialPost | None = None
    status: Literal[
        "COMPLETED", "NEEDS_REVIEW", "REJECTED", "BUDGET_EXCEEDED", "FAILED"
    ]
    case: ComplaintCase | None = None
    trace: tuple[TraceEvent, ...] = ()
    cost: CostReport


class RunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)


def deterministic_case_id(post: SocialPost) -> str:
    digest = hashlib.sha256(
        f"{post.platform}:{post.post_id}".encode("utf-8")
    ).hexdigest()
    return f"case-x-{digest[:16]}"
