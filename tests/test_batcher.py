from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from services.core.batcher import BatchItem, DynamicBatcher, compute_priority
from services.core.decision import EgressDecision, EgressRequest
from services.core.model_gateway import EscalationResult, ModelGateway


def test_priority_sorting() -> None:
    items = [
        BatchItem(
            item_id="case-low",
            prompt="p1",
            egress_request=EgressRequest(provider="local", model_id="m", data_class="public"),
            risk="LOW",
            priority=compute_priority("LOW"),
        ),
        BatchItem(
            item_id="case-urgent",
            prompt="p2",
            egress_request=EgressRequest(provider="local", model_id="m", data_class="public"),
            risk="URGENT",
            priority=compute_priority("URGENT"),
        ),
        BatchItem(
            item_id="case-high",
            prompt="p3",
            egress_request=EgressRequest(provider="local", model_id="m", data_class="public"),
            risk="HIGH",
            priority=compute_priority("HIGH"),
        ),
    ]

    mock_gateway = MagicMock()
    batcher = DynamicBatcher(mock_gateway)
    ordered = batcher.sort_by_priority(items)
    assert [x.item_id for x in ordered] == ["case-urgent", "case-high", "case-low"]


def test_fail_closed_pii_filtered_before_batching() -> None:
    mock_gateway = MagicMock()
    mock_gateway.provider = "local"
    mock_gateway.model_id = "m"
    batcher = DynamicBatcher(mock_gateway)

    items = [
        BatchItem(
            item_id="case-pii",
            prompt="lapor orang namanya Budi KTP 3273...",
            egress_request=EgressRequest(
                provider="local", model_id="m", data_class="public", has_pii=True
            ),
        )
    ]

    results = batcher.execute_batch(items)
    assert len(results) == 1
    gate, res = results["case-pii"]
    assert gate.allowed is False
    assert res.status == "DENIED"
    assert res.reason == "POLICY_DENIED"
    # Gateway was never called for PII violation
    assert not mock_gateway.escalate.called


def test_single_flight_lock_enforces_one_call_at_a_time() -> None:
    mock_gateway = MagicMock()
    mock_gateway.provider = "local"
    mock_gateway.model_id = "m"

    active_calls = 0
    max_observed_active = 0
    lock = threading.Lock()

    def slow_escalate(prompt: str, req: EgressRequest) -> tuple[EgressDecision, EscalationResult]:
        nonlocal active_calls, max_observed_active
        with lock:
            active_calls += 1
            if active_calls > max_observed_active:
                max_observed_active = active_calls
        time.sleep(0.05)
        with lock:
            active_calls -= 1
        return (
            EgressDecision(allowed=True, rule_ids=(), reason_codes=()),
            EscalationResult(status="SUCCEEDED", provider="local", model_id="m", content="OK"),
        )

    mock_gateway.escalate.side_effect = slow_escalate
    batcher = DynamicBatcher(mock_gateway, max_batch_size=1)

    items = [
        BatchItem(
            item_id=f"item-{i}",
            prompt=f"prompt-{i}",
            egress_request=EgressRequest(provider="local", model_id="m", data_class="public"),
        )
        for i in range(4)
    ]

    threads = []
    for item in items:
        t = threading.Thread(target=lambda it=item: batcher.execute_batch([it]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Crucial assertion: never more than 1 in flight!
    assert max_observed_active == 1


def test_micro_batch_combines_multiple_items_into_single_prompt() -> None:
    mock_gateway = MagicMock()
    mock_gateway.provider = "local"
    mock_gateway.model_id = "m"

    batch_output = [
        {"id": "item-1", "decision": "KLARIFIKASI", "reasoning": "Kurang lokasi"},
        {"id": "item-2", "decision": "CUKUP", "reasoning": "Lokasi lengkap"},
    ]
    mock_gateway.escalate.return_value = (
        EgressDecision(allowed=True, rule_ids=(), reason_codes=()),
        EscalationResult(
            status="SUCCEEDED",
            provider="local",
            model_id="m",
            content=json.dumps(batch_output),
        ),
    )

    batcher = DynamicBatcher(mock_gateway, max_batch_size=5)

    items = [
        BatchItem(
            item_id="item-1",
            prompt="aduan 1",
            egress_request=EgressRequest(provider="local", model_id="m", data_class="public"),
        ),
        BatchItem(
            item_id="item-2",
            prompt="aduan 2",
            egress_request=EgressRequest(provider="local", model_id="m", data_class="public"),
        ),
    ]

    results = batcher.execute_batch(items)

    # EXACTLY 1 call made to model gateway for 2 items!
    assert mock_gateway.escalate.call_count == 1
    assert len(results) == 2
    assert "item-1" in results
    assert "item-2" in results
    assert "KLARIFIKASI" in results["item-1"][1].content
    assert "CUKUP" in results["item-2"][1].content
