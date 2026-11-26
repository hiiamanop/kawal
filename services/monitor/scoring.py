from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from contracts.harness import ComplaintCase, SocialPost
from contracts.models import RiskLevel
from services.monitor.config import (
    ENGAGEMENT_WEIGHT,
    RECENCY_WINDOW_HOURS,
    RECENCY_WEIGHT,
    RISK_WEIGHT,
    TRUST_WEIGHT,
)

_RISK_SCORES = {
    RiskLevel.URGENT: 1.0,
    RiskLevel.HIGH: 0.75,
    RiskLevel.MEDIUM: 0.5,
    RiskLevel.LOW: 0.25,
}
_TRUST_SCORES = {"ACCEPTED": 1.0, "NEEDS_REVIEW": 0.6, "REJECTED": 0.0}


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        try:
            return max(0.0, float(value.replace(",", "")))
        except ValueError:
            return 0.0
    return 0.0


def _engagement_norm(post: SocialPost) -> float:
    likes = _number(post.raw.get("likes"))
    views = _number(post.raw.get("views"))
    raw = likes + 0.05 * views
    return min(1.0, math.log1p(raw) / math.log1p(10000))


def _recency_norm(post: SocialPost, now: datetime) -> float:
    if post.posted_at is None:
        return 0.0
    posted = post.posted_at
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=timezone.utc)
    hours = max(0.0, (now - posted.astimezone(timezone.utc)).total_seconds() / 3600.0)
    return max(0.0, 1.0 - hours / RECENCY_WINDOW_HOURS)


def viral_score(
    post: SocialPost,
    case: ComplaintCase | None,
    now: datetime,
) -> int:
    if case is None:
        return 0
    engagement = _engagement_norm(post)
    recency = _recency_norm(post, now)
    risk = _RISK_SCORES.get(case.triage.risk, 0.5)
    trust = _TRUST_SCORES.get(case.status, 0.0)
    score = 100 * (
        ENGAGEMENT_WEIGHT * engagement
        + RECENCY_WEIGHT * recency
        + RISK_WEIGHT * risk
        + TRUST_WEIGHT * trust
    )
    return max(0, min(100, round(score)))
