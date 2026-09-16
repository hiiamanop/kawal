from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import RiskLevel
from services.core.decision import EgressDecision, EgressRequest, evaluate_egress
from services.core.model_gateway import EscalationResult, ModelGateway

logger = logging.getLogger("dynamic-batcher")


class BatchItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_id: str
    prompt: str
    egress_request: EgressRequest
    risk: str = "MEDIUM"
    category: str = "ROAD"
    priority: int = 2  # 0: URGENT, 1: HIGH, 2: MEDIUM, 3: LOW


def compute_priority(risk: str) -> int:
    clean = risk.upper().strip()
    if clean == "URGENT":
        return 0
    if clean == "HIGH":
        return 1
    if clean == "MEDIUM":
        return 2
    return 3


class DynamicBatcher:
    """Combines concurrent escalation requests into single-flight micro-batches.

    Guarantees:
    1. Single-Flight Concurrency: strictly <= 1 request to OmniRoute at any time.
    2. Priority Ordering: URGENT and HIGH risk complaints jump to the front.
    3. High Throughput: 1 LLM request resolves up to `max_batch_size` complaints at once.
    4. Fail-closed PII gating: unauthorized requests are excluded before network egress.
    """

    _global_lock = threading.Lock()

    def __init__(
        self,
        model_gateway: ModelGateway,
        max_batch_size: int = 5,
        lock: threading.Lock | None = None,
    ) -> None:
        self._model_gateway = model_gateway
        self._max_batch_size = max(1, int(max_batch_size))
        self._lock = lock if lock is not None else self._global_lock

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def sort_by_priority(self, items: Sequence[BatchItem]) -> list[BatchItem]:
        """Order items by urgency: URGENT (0) -> HIGH (1) -> MEDIUM (2) -> LOW (3)."""
        return sorted(items, key=lambda x: x.priority)

    def execute_batch(
        self, items: Sequence[BatchItem]
    ) -> dict[str, tuple[EgressDecision, EscalationResult]]:
        """Process a collection of escalation items with priority ordering and micro-batching."""
        if not items:
            return {}

        results: dict[str, tuple[EgressDecision, EscalationResult]] = {}
        sorted_items = self.sort_by_priority(items)

        # 1. First pass: evaluate policy for all items (fail-closed)
        allowed_items: list[BatchItem] = []
        for item in sorted_items:
            gate = evaluate_egress(item.egress_request)
            if not gate.allowed:
                results[item.item_id] = (
                    gate,
                    EscalationResult(
                        status="DENIED",
                        provider=self._model_gateway.provider,
                        model_id=self._model_gateway.model_id,
                        reason="POLICY_DENIED",
                    ),
                )
            else:
                allowed_items.append(item)

        if not allowed_items:
            return results

        # 2. Chunk allowed items into micro-batches of size <= max_batch_size
        chunks = [
            allowed_items[i : i + self._max_batch_size]
            for i in range(0, len(allowed_items), self._max_batch_size)
        ]

        # 3. Process each micro-batch through the Single-Flight Concurrency Lock
        for chunk in chunks:
            if len(chunk) == 1:
                item = chunk[0]
                gate, esc_res = self._execute_single_with_lock(item)
                results[item.item_id] = (gate, esc_res)
            else:
                batch_results = self._execute_multi_with_lock(chunk)
                results.update(batch_results)

        return results

    def _execute_single_with_lock(
        self, item: BatchItem
    ) -> tuple[EgressDecision, EscalationResult]:
        with self._lock:
            return self._model_gateway.escalate(item.prompt, item.egress_request)

    def _execute_multi_with_lock(
        self, chunk: Sequence[BatchItem]
    ) -> dict[str, tuple[EgressDecision, EscalationResult]]:
        """Run multiple items in a single LLM prompt call while holding the concurrency lock."""
        with self._lock:
            t0 = time.perf_counter()
            items_payload = [
                {
                    "id": item.item_id,
                    "category": item.category,
                    "risk": item.risk,
                    "prompt": item.prompt,
                }
                for item in chunk
            ]

            batch_prompt = (
                "Tugas: Anda adalah sistem evaluasi cerdas KAWAL untuk aduan publik warga kota.\n"
                "Proses SEMUA aduan di bawah sekaligus. Tentukan apakah masing-masing aduan memerlukan "
                "KLARIFIKASI lokasi atau CUKUP.\n\n"
                f"Daftar Aduan:\n{json.dumps(items_payload, indent=2)}\n\n"
                "Keluarkan HANYA JSON array murni tanpa format markdown tambahan:\n"
                "[\n"
                '  {"id": "...", "decision": "KLARIFIKASI" | "CUKUP", "reasoning": "..."}\n'
                "]"
            )

            # Use the first item's egress request as the envelope
            sample_egress = chunk[0].egress_request
            gate, raw_result = self._model_gateway.escalate(batch_prompt, sample_egress)

            if raw_result.status not in ("SUCCEEDED", "APPROVED") or not raw_result.content:
                logger.warning("Micro-batch request returned %s; falling back to sequential single calls", raw_result.status)
                return self._fallback_sequential(chunk)

            parsed_array = self._parse_json_array(raw_result.content)
            if not parsed_array:
                logger.warning("Failed to parse micro-batch JSON array; falling back to sequential single calls")
                return self._fallback_sequential(chunk)

            mapped_results: dict[str, tuple[EgressDecision, EscalationResult]] = {}
            for res_obj in parsed_array:
                cid = str(res_obj.get("id", ""))
                decision = str(res_obj.get("decision", "KLARIFIKASI")).strip().upper()
                content = f"{decision}: {res_obj.get('reasoning', '')}"
                mapped_results[cid] = (
                    gate,
                    EscalationResult(
                        status="APPROVED",
                        provider=raw_result.provider,
                        model_id=raw_result.model_id,
                        content=content,
                    ),
                )

            # Check if any items in the chunk were missed by the model output
            for item in chunk:
                if item.item_id not in mapped_results:
                    mapped_results[item.item_id] = (
                        gate,
                        EscalationResult(
                            status="APPROVED",
                            provider=raw_result.provider,
                            model_id=raw_result.model_id,
                            content="KLARIFIKASI: Evaluasi batch otomatis",
                        ),
                    )

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Executed micro-batch of %d items in %.1f ms (1 LLM call)", len(chunk), elapsed)
            return mapped_results

    def _fallback_sequential(
        self, chunk: Sequence[BatchItem]
    ) -> dict[str, tuple[EgressDecision, EscalationResult]]:
        res: dict[str, tuple[EgressDecision, EscalationResult]] = {}
        for item in chunk:
            res[item.item_id] = self._model_gateway.escalate(item.prompt, item.egress_request)
        return res

    @staticmethod
    def _parse_json_array(text: str) -> list[dict[str, Any]] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        match = re.search(r"(\[.*\])", cleaned, flags=re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
            except Exception:
                pass
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception:
            pass
        return None
