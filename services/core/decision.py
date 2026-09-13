from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.models import DecisionMode


class ConflictType(StrEnum):
    CLASSIFICATION = "CLASSIFICATION"
    ENTITY = "ENTITY"
    LOCATION = "LOCATION"
    RISK = "RISK"
    EVIDENCE = "EVIDENCE"
    TEXT_IMAGE = "TEXT_IMAGE"
    AUTHORITY = "AUTHORITY"


class ConflictSeverity(StrEnum):
    MINOR = "MINOR"
    MATERIAL = "MATERIAL"
    CRITICAL = "CRITICAL"
    HARD = "HARD"


class ResolutionCapability(StrEnum):
    NONE = "NONE"
    LOCAL_VERIFICATION = "LOCAL_VERIFICATION"
    CLARIFICATION = "CLARIFICATION"
    POLICY_CORRECTION = "POLICY_CORRECTION"


class TrustContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_version: str = Field(min_length=1)
    task: str = Field(min_length=1)
    category: str = Field(min_length=1)
    language_style: str = Field(min_length=1)
    risk_band: str = Field(min_length=1)


class TrustObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context: TrustContext
    successes: int = Field(ge=0)
    total: int = Field(ge=0)
    ece: float = Field(ge=0.0, le=1.0)
    drift_penalty: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def successes_do_not_exceed_total(self) -> TrustObservation:
        if self.successes > self.total:
            raise ValueError("successes must not exceed total")
        return self


class TrustResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    posterior_mean: float = Field(ge=0.0, le=1.0)
    lower_bound: float = Field(ge=0.0, le=1.0)
    reliability_weight: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=0)
    used_parent: bool


class ConflictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ConflictType
    field: str = Field(min_length=1)
    left_value: str = Field(min_length=1)
    right_value: str = Field(min_length=1)
    left_trust: float = Field(ge=0.0, le=1.0)
    right_trust: float = Field(ge=0.0, le=1.0)
    consequence_weight: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[str, ...] = Field(min_length=2)
    route_changes: bool = False
    visual_required: bool = False


class ConflictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ConflictType
    severity: ConflictSeverity
    field: str
    score: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[str, ...]
    supported_values: tuple[str, str]
    resolvable_by: ResolutionCapability


class EscalationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    expected_utility_gain: float
    cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    egress_bytes: int = Field(ge=0)
    queue_delay_ms: float = Field(ge=0.0)
    policy_allowed: bool


class BudgetState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    remaining_usd: float = Field(ge=0.0)
    remaining_latency_ms: float = Field(ge=0.0)
    remaining_egress_bytes: int = Field(ge=0)


class DecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    required_fields_complete: bool
    evidence_covered: bool
    authority_valid: bool
    policy_allowed: bool
    is_complaint: bool
    conflicts: tuple[ConflictRecord, ...] = ()
    escalation_candidates: tuple[EscalationCandidate, ...] = ()
    budget: BudgetState
    clarification_available: bool
    voi_threshold: float = 0.0
    completed_task_ids: frozenset[str] = frozenset()


class DecisionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: DecisionMode
    reason_codes: tuple[str, ...]
    selected_task_id: str | None = None
    next_state: str
    reserved_cost_usd: float = Field(ge=0.0)


def _normal_quantile(probability: float) -> float:
    if probability == 0.10:
        return -1.2815515655446004
    raise ValueError("only the frozen 10% lower credible bound is supported")


def estimate_contextual_trust(
    observation: TrustObservation | None,
    parent_successes: int,
    parent_total: int,
    evidence_quality: float,
    prior_strength: float = 20.0,
    min_context_support: int = 30,
) -> TrustResult:
    """Compute conservative Beta-shrinkage trust with deterministic normal approximation."""
    if parent_successes < 0 or parent_total < 0 or parent_successes > parent_total:
        raise ValueError("invalid parent success counts")
    if not 0.0 <= evidence_quality <= 1.0:
        raise ValueError("evidence_quality must be between 0 and 1")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")

    parent_mean = (parent_successes + 1.0) / (parent_total + 2.0)
    support = observation.total if observation is not None else 0
    used_parent = observation is None or support < min_context_support
    successes = observation.successes if observation is not None else 0
    ece = observation.ece if observation is not None else 0.0
    drift = observation.drift_penalty if observation is not None else 0.0
    alpha = successes + prior_strength * parent_mean
    beta = support - successes + prior_strength * (1.0 - parent_mean)
    posterior_mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    lower_bound = max(0.0, min(1.0, posterior_mean + _normal_quantile(0.10) * variance**0.5))
    reliability = max(0.0, min(1.0, lower_bound * (1.0 - ece) * evidence_quality * (1.0 - drift)))
    return TrustResult(
        posterior_mean=posterior_mean,
        lower_bound=lower_bound,
        reliability_weight=reliability,
        support=support,
        used_parent=used_parent,
    )


def diagnose_conflict(input_: ConflictInput) -> ConflictRecord | None:
    if input_.left_value.strip().casefold() == input_.right_value.strip().casefold():
        return None
    if input_.type == ConflictType.AUTHORITY:
        severity = ConflictSeverity.HARD
        resolver = ResolutionCapability.POLICY_CORRECTION
        score = 1.0
    else:
        score = min(1.0, min(input_.left_trust, input_.right_trust) * input_.consequence_weight)
        if input_.type == ConflictType.RISK and input_.consequence_weight >= 0.8:
            severity = ConflictSeverity.CRITICAL
            resolver = ResolutionCapability.LOCAL_VERIFICATION
        elif input_.type == ConflictType.TEXT_IMAGE and input_.visual_required:
            severity = ConflictSeverity.CRITICAL
            resolver = ResolutionCapability.CLARIFICATION
        elif input_.type in (ConflictType.LOCATION, ConflictType.ENTITY) or input_.route_changes:
            severity = ConflictSeverity.MATERIAL
            resolver = ResolutionCapability.CLARIFICATION
        elif input_.type == ConflictType.EVIDENCE:
            severity = ConflictSeverity.MATERIAL
            resolver = ResolutionCapability.CLARIFICATION
        else:
            severity = ConflictSeverity.MINOR
            resolver = ResolutionCapability.LOCAL_VERIFICATION
    return ConflictRecord(
        type=input_.type,
        severity=severity,
        field=input_.field,
        score=score,
        evidence_refs=input_.evidence_refs,
        supported_values=(input_.left_value, input_.right_value),
        resolvable_by=resolver,
    )


def expected_value_of_information(candidate: EscalationCandidate) -> float:
    if not candidate.policy_allowed:
        return float("-inf")
    resource_penalty = candidate.cost_usd + candidate.latency_ms / 1000.0 + candidate.egress_bytes / 1_000_000.0 + candidate.queue_delay_ms / 1000.0
    return candidate.expected_utility_gain - resource_penalty


def decide(input_: DecisionInput) -> DecisionPlan:
    """Return a pure, stable decision plan without side effects or wall-clock inputs."""
    hard_conflict = any(conflict.severity == ConflictSeverity.HARD for conflict in input_.conflicts)
    blocking_conflict = any(
        conflict.severity in (ConflictSeverity.MATERIAL, ConflictSeverity.CRITICAL)
        for conflict in input_.conflicts
    )
    if not input_.is_complaint:
        return DecisionPlan(mode=DecisionMode.REJECT_IGNORE, reason_codes=("OUT_OF_SCOPE_OR_NON_COMPLAINT",), next_state="REJECTED", reserved_cost_usd=0.0)
    if not input_.policy_allowed or not input_.authority_valid or hard_conflict:
        return DecisionPlan(mode=DecisionMode.RE_EVALUATE, reason_codes=("HARD_POLICY_OR_AUTHORITY_BLOCK",), next_state="WAITING_RESULTS", reserved_cost_usd=0.0)
    feasible = [
        candidate for candidate in input_.escalation_candidates
        if candidate.policy_allowed
        and candidate.task_id not in input_.completed_task_ids
        and candidate.cost_usd <= input_.budget.remaining_usd
        and candidate.latency_ms + candidate.queue_delay_ms <= input_.budget.remaining_latency_ms
        and candidate.egress_bytes <= input_.budget.remaining_egress_bytes
    ]
    ranked = sorted(feasible, key=lambda candidate: (-expected_value_of_information(candidate), candidate.task_id))
    if blocking_conflict or not input_.required_fields_complete or not input_.evidence_covered:
        if ranked and expected_value_of_information(ranked[0]) > input_.voi_threshold:
            candidate = ranked[0]
            return DecisionPlan(mode=DecisionMode.RE_EVALUATE, reason_codes=("POSITIVE_VOI",), selected_task_id=candidate.task_id, next_state="WAITING_RESULTS", reserved_cost_usd=candidate.cost_usd)
        if input_.clarification_available:
            return DecisionPlan(mode=DecisionMode.REQUEST_CLARIFICATION, reason_codes=("MANDATORY_FACT_OR_CONFLICT_UNRESOLVED",), next_state="WAITING_CLARIFICATION", reserved_cost_usd=0.0)
        return DecisionPlan(mode=DecisionMode.RE_EVALUATE, reason_codes=("BLOCKED_WITHOUT_CLARIFICATION_CAPABILITY",), next_state="WAITING_RESULTS", reserved_cost_usd=0.0)
    return DecisionPlan(mode=DecisionMode.EXECUTE, reason_codes=("EVIDENCE_ROUTE_POLICY_READY",), next_state="EXECUTING", reserved_cost_usd=0.0)


ALLOWED_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic", "google", "local"})
ALLOWED_DATA_CLASSES: frozenset[str] = frozenset({"public", "synthetic", "internal_proxy"})
BLOCKED_MIME_TYPES: frozenset[str] = frozenset({"application/octet-stream", "application/x-executable"})
IMAGE_MIME_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/png", "image/webp"})


class EgressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    data_class: str = Field(min_length=1)
    has_pii: bool = False
    attachment_mime: str | None = None
    attachment_role: str | None = None
    category_sensitive: bool = False
    entity_flag_sensitive: bool = False
    budget: BudgetState | None = None
    cost_usd: float = Field(ge=0.0, default=0.0)


class EgressDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    rule_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


def evaluate_egress(request: EgressRequest) -> EgressDecision:
    """Deterministic, fail-closed model gateway. No side effects."""
    if request.provider not in ALLOWED_PROVIDERS:
        return EgressDecision(allowed=False, rule_ids=("POL-01",), reason_codes=("PROVIDER_NOT_ON_ALLOWLIST",))
    if request.data_class not in ALLOWED_DATA_CLASSES:
        return EgressDecision(allowed=False, rule_ids=("POL-01",), reason_codes=("DATA_CLASS_NOT_ALLOWED",))
    if request.has_pii:
        return EgressDecision(allowed=False, rule_ids=("POL-01",), reason_codes=("PII_EGRESS_BLOCKED",))
    if request.attachment_mime is not None:
        if request.attachment_mime in BLOCKED_MIME_TYPES:
            return EgressDecision(allowed=False, rule_ids=("POL-07",), reason_codes=("BLOCKED_MIME_TYPE",))
        if request.attachment_mime in IMAGE_MIME_TYPES:
            if request.category_sensitive or request.entity_flag_sensitive:
                return EgressDecision(allowed=False, rule_ids=("POL-01",), reason_codes=("SENSITIVE_IMAGE_EGRESS_DENIED",))
    if request.budget is not None and request.cost_usd > request.budget.remaining_usd:
        return EgressDecision(allowed=False, rule_ids=("POL-08",), reason_codes=("BUDGET_EXCEEDED",))
    return EgressDecision(allowed=True, rule_ids=("POL-01",), reason_codes=("EGRESS_ALLOWED",))


class TaskResultCache:
    """Per-cycle cache preventing duplicate task invocations."""

    def __init__(self) -> None:
        self._results: dict[str, Any] = {}

    def has(self, task_id: str) -> bool:
        return task_id in self._results

    def get(self, task_id: str) -> Any | None:
        return self._results.get(task_id)

    def put(self, task_id: str, result: Any) -> None:
        self._results[task_id] = result

    @property
    def completed_task_ids(self) -> frozenset[str]:
        return frozenset(self._results)

    def clear(self) -> None:
        self._results.clear()
