from __future__ import annotations

from datetime import timedelta

import pytest

from services.monitor.models import CaseUpdate
from services.monitor.store import InvalidTransition, MonitorStore

from tests.test_monitor_helpers import _monitor_case, _run, _store_with_case


def test_upsert_case_creates_once_and_preserves_operator_state(tmp_path) -> None:
    store = MonitorStore(tmp_path / "monitor.db")
    store.create_schema()
    post, run = _run("Min, jalan di Jl. Kopo rusak parah.")
    case = _monitor_case(post, run)
    assert store.upsert_case(case, "jalan rusak") is True
    store.update_case(case.case_id, CaseUpdate(status="MONITORING", note="Cek lokasi"))
    case2 = case.model_copy(
        update={"last_seen_at": case.last_seen_at + timedelta(hours=1)}
    )
    assert store.upsert_case(case2, "trotoar rusak") is False
    saved = store.get_case(case.case_id)
    assert saved is not None
    assert saved.operator_status == "MONITORING"
    assert saved.notes == "Cek lokasi"
    assert {row["query"] for row in store.list_case_queries(case.case_id)} == {
        "jalan rusak",
        "trotoar rusak",
    }


def test_invalid_status_transition_is_rejected(tmp_path) -> None:
    store = _store_with_case(tmp_path)
    with pytest.raises(InvalidTransition):
        store.update_case("case-x-1", CaseUpdate(status="FORWARDED"))


def test_failed_analysis_is_hidden_from_default_feed(tmp_path) -> None:
    store = _store_with_case(
        tmp_path, analysis_status="FAILED", viral_score=None, department=None
    )
    assert store.list_cases() == []
    assert len(store.list_cases(analysis_status="FAILED")) == 1


def test_stats_and_actions(tmp_path) -> None:
    store = _store_with_case(tmp_path)
    store.update_case("case-x-1", CaseUpdate(status="READY_TO_FORWARD", note="Siap"))
    assert store.stats().ready_to_forward == 1
    assert [a["new_status"] for a in store.list_actions("case-x-1")][-1] == (
        "READY_TO_FORWARD"
    )
