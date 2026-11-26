from __future__ import annotations

from contracts.harness import ComplaintCase
from services.core.policy import resolve_department_name


def recommend_department(case: ComplaintCase) -> str:
    return resolve_department_name(case.triage.category)
