from __future__ import annotations

import pytest

from contracts.models import Category
from services.core.decision import EgressRequest
from services.core.model_gateway import ModelGateway, OmniRouteAdapter
from services.intelligence.causality import analyze_causality
from services.ml import LocalMLRuntime


def test_omniroute_adapter_initialization_and_properties() -> None:
    adapter = OmniRouteAdapter(model_id="gpt-4o-mini", base_url="http://localhost:20128/v1")
    assert adapter.provider == "openai"
    assert adapter.model_id == "gpt-4o-mini"
    assert adapter._url == "http://localhost:20128/v1/chat/completions"


def test_omniroute_adapter_under_model_gateway() -> None:
    adapter = OmniRouteAdapter(model_id="gpt-4o-mini", base_url="http://localhost:20128/v1")
    gateway = ModelGateway(adapter)

    # Allowed non-PII internal proxy request
    gate, res = gateway.escalate(
        "Halo, tes koneksi",
        EgressRequest(
            provider="openai",
            model_id="gpt-4o-mini",
            data_class="internal_proxy",
            has_pii=False,
        ),
    )
    assert gate.allowed is True
    assert res.status == "SUCCEEDED"
    assert res.content is not None

    # Blocked PII request
    gate_pii, res_pii = gateway.escalate(
        "Data dengan NIK 3273010101010001",
        EgressRequest(
            provider="openai",
            model_id="gpt-4o-mini",
            data_class="internal_proxy",
            has_pii=True,
        ),
    )
    assert gate_pii.allowed is False
    assert res_pii.status == "DENIED"


def test_semantic_causal_disambiguation_without_lexical_connectors() -> None:
    """Verifies that an ambiguous complaint without words like 'bikin' or 'gara-gara'

    is cleanly resolved using OmniRoute semantic disambiguation.
    """
    runtime = LocalMLRuntime.live_from_artifacts()
    llm = OmniRouteAdapter(model_id="gpt-4o-mini")

    text = "pipa ledeng pecah, jalanan depan rumah jadi danau aspal bolong semua"
    causal = analyze_causality(text, ml_runtime=runtime, llm_adapter=llm)

    assert causal.has_causal_relation is True
    assert causal.root_cause_category == Category.CLEAN_WATER
    assert causal.secondary_category == Category.ROAD
    assert "pipa" in causal.explanation.lower() or "ledeng" in causal.explanation.lower()
