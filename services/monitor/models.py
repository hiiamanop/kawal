from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contracts.harness import ComplaintCase, HarnessRun, SocialPost


OperatorStatus = Literal[
    "NEW", "MONITORING", "READY_TO_FORWARD", "FORWARDED", "DISMISSED"
]
AnalysisStatus = Literal["OK", "FAILED", "BUDGET_EXCEEDED"]


class MonitorCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    post: SocialPost
    run: HarnessRun
    viral_score: int | None = Field(default=None, ge=0, le=100)
    department: str | None = None
    operator_status: OperatorStatus = "NEW"
    analysis_status: AnalysisStatus = "OK"
    notes: str = ""
    first_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def case(self) -> ComplaintCase | None:
        return self.run.case


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    source_mode: Literal["live", "replay"] = "live"
    analysis_mode: Literal["rule", "llm"] = "rule"
    fixture_dir: str | None = None


class ScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scan_id: str
    query: str
    source_mode: Literal["live", "replay"]
    analysis_mode: Literal["rule", "llm"]
    new_cases: int = Field(default=0, ge=0)
    updated_cases: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    cases: tuple[MonitorCase, ...] = ()
    errors: tuple[str, ...] = ()
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CaseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: OperatorStatus | None = None
    note: str | None = None


class ForwardPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    markdown_path: str
    json_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MonitorStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = 0
    complaints: int = 0
    needs_review: int = 0
    ready_to_forward: int = 0
    forwarded: int = 0
    last_scan_at: datetime | None = None
