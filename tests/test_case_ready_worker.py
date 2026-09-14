from __future__ import annotations

from services.core.worker import CaseReadyWorker, MissingCaseSnapshotError
from services.outbox.broker import BrokerMessage


def _event() -> BrokerMessage:
    return BrokerMessage(
        event_id="11111111-1111-1111-1111-111111111111",
        topic="cases.ready.v1",
        partition_key="tenant:case",
        payload={"case_id": "case"},
    )


def test_missing_snapshot_is_considered_unprocessable_not_retryable() -> None:
    assert CaseReadyWorker._handle_unprocessable_event(
        _event(), MissingCaseSnapshotError("missing")
    ) is True


def test_unexpected_worker_error_remains_retryable() -> None:
    assert CaseReadyWorker._handle_unprocessable_event(
        _event(), RuntimeError("database unavailable")
    ) is False
