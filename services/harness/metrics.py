from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contracts.harness import LocationQuality
from contracts.models import Category
from services.harness.compare import ComparisonItem, ComparisonReport


class FixtureLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_complaint: bool
    category: Category | None = None
    location_quality: LocationQuality
    acceptable_statuses: tuple[str, ...] = Field(min_length=1)
    notes: str = ""


class LabelSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    labels: dict[str, FixtureLabel]


class ModeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["rule", "llm"]
    total: int = Field(ge=0)
    predicted_complaints: int = Field(ge=0)
    actual_complaints: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    complaint_precision: float = Field(ge=0.0, le=1.0)
    complaint_recall: float = Field(ge=0.0, le=1.0)
    category_correct: int = Field(ge=0)
    category_evaluated: int = Field(ge=0)
    category_accuracy: float = Field(ge=0.0, le=1.0)
    decision_correct: int = Field(ge=0)
    decision_accuracy: float = Field(ge=0.0, le=1.0)
    location_correct: int = Field(ge=0)
    location_evaluated: int = Field(ge=0)
    location_accuracy: float = Field(ge=0.0, le=1.0)
    review_rate: float = Field(ge=0.0, le=1.0)
    false_accepts: int = Field(ge=0)
    false_rejects: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)
    cost_per_correct_usd: float = Field(ge=0.0)


class MetricsReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_id: str
    labels_count: int = Field(ge=0)
    rule: ModeMetrics
    llm: ModeMetrics
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def load_labels(path: str | Path) -> LabelSet:
    return LabelSet.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _score_mode(
    mode: Literal["rule", "llm"],
    items: list[tuple[ComparisonItem, FixtureLabel]],
) -> ModeMetrics:
    predicted_complaints = 0
    actual_complaints = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    category_correct = 0
    category_evaluated = 0
    decision_correct = 0
    location_correct = 0
    location_evaluated = 0
    review_count = 0
    false_accepts = 0
    false_rejects = 0
    total_cost = 0.0

    for item, label in items:
        if mode == "rule":
            status = item.rule_status
            category = item.rule_category
            is_complaint = item.rule_is_complaint
            location_quality = item.rule_location_quality
            cost = item.rule_cost_usd
        else:
            status = item.llm_status
            category = item.llm_category
            is_complaint = item.llm_is_complaint
            location_quality = item.llm_location_quality
            cost = item.llm_cost_usd

        total_cost += cost
        predicted = is_complaint is True
        predicted_complaints += int(predicted)
        actual_complaints += int(label.is_complaint)
        if predicted and label.is_complaint:
            true_positives += 1
        elif predicted and not label.is_complaint:
            false_positives += 1
        elif not predicted and label.is_complaint:
            false_negatives += 1

        if label.category is not None and category is not None and predicted:
            category_evaluated += 1
            if category == label.category.value:
                category_correct += 1

        if status in label.acceptable_statuses:
            decision_correct += 1
        if status == "COMPLETED" and "COMPLETED" not in label.acceptable_statuses:
            false_accepts += 1
        if status == "REJECTED" and label.is_complaint:
            false_rejects += 1
        if status == "NEEDS_REVIEW":
            review_count += 1

        if location_quality is not None:
            location_evaluated += 1
            if location_quality == label.location_quality:
                location_correct += 1

    total = len(items)
    precision = true_positives / predicted_complaints if predicted_complaints else 0.0
    recall = true_positives / actual_complaints if actual_complaints else 0.0
    return ModeMetrics(
        mode=mode,
        total=total,
        predicted_complaints=predicted_complaints,
        actual_complaints=actual_complaints,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        complaint_precision=precision,
        complaint_recall=recall,
        category_correct=category_correct,
        category_evaluated=category_evaluated,
        category_accuracy=(category_correct / category_evaluated)
        if category_evaluated
        else 0.0,
        decision_correct=decision_correct,
        decision_accuracy=(decision_correct / total) if total else 0.0,
        location_correct=location_correct,
        location_evaluated=location_evaluated,
        location_accuracy=(location_correct / location_evaluated)
        if location_evaluated
        else 0.0,
        review_rate=(review_count / total) if total else 0.0,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        total_cost_usd=total_cost,
        cost_per_correct_usd=(total_cost / decision_correct)
        if decision_correct
        else 0.0,
    )


def score_comparison(report: ComparisonReport, labels: LabelSet) -> MetricsReport:
    items: list[tuple[ComparisonItem, FixtureLabel]] = []
    for item in report.items:
        label = labels.labels.get(item.post_id)
        if label is not None:
            items.append((item, label))
    return MetricsReport(
        comparison_id=report.comparison_id,
        labels_count=len(items),
        rule=_score_mode("rule", items),
        llm=_score_mode("llm", items),
    )
