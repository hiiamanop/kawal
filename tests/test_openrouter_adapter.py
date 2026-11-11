from __future__ import annotations

import json
from typing import Any

import pytest

from contracts.models import Category
from services.core.decision import EgressRequest
from services.core.model_gateway import ModelGateway, OmniRouteAdapter, OpenRouterAdapter
from services.intelligence.causality import analyze_causality
from services.ml import LocalMLRuntime


class _FakeResponse:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_openrouter_adapter_uses_openrouter_defaults() -> None:
    adapter = OpenRouterAdapter(model_id="google/gemini-2.5-flash", api_key="sk-or-test")

    assert adapter.provider == "openai"
    assert adapter.model_id == "google/gemini-2.5-flash"
    assert adapter._url == "https://openrouter.ai/api/v1/chat/completions"
    headers = adapter._headers()
    assert headers["Authorization"] == "Bearer sk-or-test"
    assert "HTTP-Referer" in headers
    assert headers["X-Title"] == "KAWAL"


def test_legacy_omniroute_alias_points_at_openrouter_adapter() -> None:
    assert OmniRouteAdapter is OpenRouterAdapter


def test_openrouter_adapter_parses_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "tes berhasil"}}]})

    monkeypatch.setattr("services.core.model_gateway.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(model_id="google/gemini-2.5-flash", api_key="sk-or-test")

    result = adapter.run("Halo")

    assert result.status == "SUCCEEDED"
    assert result.content == "tes berhasil"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["payload"]["model"] == "google/gemini-2.5-flash"


def test_openrouter_adapter_requires_no_pii_egress() -> None:
    adapter = OpenRouterAdapter(model_id="google/gemini-2.5-flash", api_key="sk-or-test")
    gateway = ModelGateway(adapter)

    gate, res = gateway.escalate(
        "Halo, tes koneksi",
        EgressRequest(
            provider="openai",
            model_id="google/gemini-2.5-flash",
            data_class="internal_proxy",
            has_pii=True,
        ),
    )
    assert gate.allowed is False
    assert res.status == "DENIED"


class _StubDisambiguator:
    provider = "openai"
    model_id = "stub-model"

    def run(self, prompt: str):  # pragma: no cover - not used in this test
        raise AssertionError("run() should not be called")

    def disambiguate_complaint(self, text: str) -> dict[str, Any] | None:
        return {
            "is_complaint": True,
            "root_cause_category": "CLEAN_WATER",
            "secondary_category": "ROAD",
            "reasoning": "Pipa ledeng pecah merusak badan jalan.",
        }


def test_semantic_causal_disambiguation_without_lexical_connectors() -> None:
    runtime = LocalMLRuntime.live_from_artifacts()
    llm = _StubDisambiguator()

    text = "pipa ledeng pecah, jalanan depan rumah jadi danau aspal bolong semua"
    causal = analyze_causality(text, ml_runtime=runtime, llm_adapter=llm)

    assert causal.has_causal_relation is True
    assert causal.root_cause_category == Category.CLEAN_WATER
    assert causal.secondary_category == Category.ROAD
    assert "pipa" in causal.explanation.lower() or "ledeng" in causal.explanation.lower()
