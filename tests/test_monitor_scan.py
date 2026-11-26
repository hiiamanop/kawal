from __future__ import annotations

import pytest

from contracts.harness import CostReport, HarnessRun, SocialPost
from services.harness.sources import SourceError
from services.monitor.models import CaseUpdate
from services.monitor.scan import run_scan
from services.monitor.store import MonitorStore

from tests.test_monitor_helpers import _request, _write_fixture


class FailingSource:
    def fetch(self, query: str, limit: int) -> list[SocialPost]:
        raise SourceError("UPSTREAM_BROKEN", "source down")


class FailingOrchestrator:
    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun:
        return HarnessRun(
            run_id=run_id or "failed-run",
            post=post,
            status="FAILED",
            cost=CostReport(),
        )


class RaisingOrchestrator:
    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun:
        raise RuntimeError("boom")


class MissingCaseOrchestrator:
    def run(self, post: SocialPost, run_id: str | None = None) -> HarnessRun:
        return HarnessRun(
            run_id=run_id or "missing-case-run",
            post=post,
            status="NEEDS_REVIEW",
            cost=CostReport(),
        )


def _store(tmp_path) -> MonitorStore:
    store = MonitorStore(tmp_path / "monitor.db")
    store.create_schema()
    return store


def test_replay_scan_stores_and_ranks_cases(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    result = run_scan(_request(fixture_dir, "jalan rusak"), _store(tmp_path))
    assert result.new_cases == 1
    assert result.cases[0].department == "Dinas Sumber Daya Air dan Bina Marga (DSDABM)"
    assert result.cases[0].viral_score is not None


def test_second_scan_updates_without_duplicating(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    store = _store(tmp_path)
    first = run_scan(_request(fixture_dir, "jalan rusak"), store)
    store.update_case(
        first.cases[0].case_id, CaseUpdate(status="MONITORING", note="Pantau")
    )
    second = run_scan(_request(fixture_dir, "trotoar rusak"), store)
    assert second.updated_cases == 1
    saved = store.get_case(first.cases[0].case_id)
    assert saved is not None
    assert saved.operator_status == "MONITORING"
    assert saved.notes == "Pantau"


def test_failed_analysis_is_stored_as_error(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    result = run_scan(
        _request(fixture_dir, "jalan rusak"),
        _store(tmp_path),
        orchestrator=FailingOrchestrator(),
    )
    assert result.failed == 1
    assert result.cases[0].analysis_status == "FAILED"
    assert result.cases[0].viral_score is None


def test_source_failure_returns_error_without_cases(tmp_path) -> None:
    with pytest.raises(SourceError):
        run_scan(
            _request(tmp_path, "jalan rusak"),
            _store(tmp_path),
            source=FailingSource(),
        )


def test_failed_duplicate_scan_preserves_existing_analysis(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    store = _store(tmp_path)
    first = run_scan(_request(fixture_dir, "jalan rusak"), store)
    original = store.get_case(first.cases[0].case_id)
    assert original is not None
    assert original.analysis_status == "OK"
    second = run_scan(
        _request(fixture_dir, "trotoar rusak"),
        store,
        orchestrator=FailingOrchestrator(),
    )
    saved = store.get_case(first.cases[0].case_id)
    assert saved is not None
    assert second.failed == 0
    assert saved.analysis_status == "OK"
    assert saved.run == original.run
    assert saved.operator_status == original.operator_status


def test_orchestrator_exception_counts_failure_once(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    result = run_scan(
        _request(fixture_dir, "jalan rusak"),
        _store(tmp_path),
        orchestrator=RaisingOrchestrator(),
    )
    assert result.failed == 1
    assert result.cases[0].analysis_status == "FAILED"
    assert len(result.errors) == 1


def test_needs_review_without_case_is_a_failed_analysis(tmp_path) -> None:
    fixture_dir = _write_fixture(
        tmp_path / "fixtures", "post-1", "Min, jalan di Jl. Kopo rusak parah."
    )
    store = _store(tmp_path)
    result = run_scan(
        _request(fixture_dir, "jalan rusak"),
        store,
        orchestrator=MissingCaseOrchestrator(),
    )
    assert result.failed == 1
    assert result.cases[0].analysis_status == "FAILED"
    assert store.list_cases() == []
    assert len(store.list_cases(analysis_status="FAILED")) == 1
