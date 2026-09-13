from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import Category, DecisionMode, RiskLevel


class SystemID(StrEnum):
    P = "P"
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"


class EvalSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    gold_category: Category
    gold_risk: RiskLevel
    gold_completeness: str
    gold_action: DecisionMode
    is_conflict_case: bool = False
    is_prohibited_case: bool = False
    context: dict[str, Any] = Field(default_factory=dict)


class SystemExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    system_id: SystemID
    predicted_mode: DecisionMode
    predicted_category: Category
    predicted_risk: RiskLevel
    is_correct_action: bool
    is_prohibited_action_attempted: bool
    is_prohibited_action_passed: bool
    cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    egress_bytes: int = Field(ge=0)
    re_evaluation_count: int = Field(ge=0)
    audit_lineage: dict[str, Any] = Field(default_factory=dict)


class BootstrapCI(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    mean: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_resamples: int = 10000


class PairedDiffResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system_a: SystemID
    system_b: SystemID
    metric: str
    diff_mean: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant_95: bool


class SystemMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system_id: SystemID
    action_macro_f1: float
    action_accuracy: float
    incorrect_execute_rate_conflict: float
    prohibited_action_rate: float
    mean_cost_usd: float
    mean_latency_ms: float
    p95_latency_ms: float
    confidence_intervals: dict[str, BootstrapCI] = Field(default_factory=dict)


class ThesisHypothesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    statement: str
    passed: bool
    comparison: str
    evidence: dict[str, Any]


class ThesisEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_cases: int
    total_families: int
    systems_evaluated: tuple[SystemID, ...]
    system_metrics: dict[str, SystemMetrics]
    hypotheses: dict[str, ThesisHypothesisResult]
    paired_differences: list[PairedDiffResult]
    robustness_invariants: dict[str, bool]
    frozen_manifest_hashes: dict[str, str] = Field(default_factory=dict)
