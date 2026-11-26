from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts.models import Category
from services.monitor.routing import recommend_department
from services.monitor.scoring import viral_score

from tests.test_monitor_helpers import _complaint_case, _post


def test_viral_score_uses_engagement_recency_risk_and_trust() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    post = _post(likes=100, views=1000, posted_at=now - timedelta(hours=1))
    case = _complaint_case(risk="HIGH", status="ACCEPTED")
    assert viral_score(post, case, now) == 74


def test_missing_engagement_is_zero_not_an_error() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    assert (
        viral_score(
            _post(likes=None, views=None, posted_at=now), _complaint_case(), now
        )
        >= 0
    )


def test_department_routing_uses_category_mapping() -> None:
    assert (
        recommend_department(_complaint_case(category=Category.WASTE))
        == "Dinas Lingkungan Hidup (DLH)"
    )
    assert (
        recommend_department(_complaint_case(category=Category.ROAD))
        == "Dinas Sumber Daya Air dan Bina Marga (DSDABM)"
    )
