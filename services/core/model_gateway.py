from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from services.core.decision import EgressDecision, EgressRequest, evaluate_egress


class EscalationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    provider: str
    model_id: str
    content: str | None = None
    reason: str | None = None


class EscalationAdapter(Protocol):
    provider: str
    model_id: str

    def run(self, prompt: str) -> EscalationResult: ...


class OllamaLocalAdapter:
    provider = "local"

    def __init__(self, model_id: str, base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 15.0) -> None:
        self.model_id = model_id
        self._url = f"{base_url.rstrip('/')}/api/generate"
        self._timeout_seconds = timeout_seconds

    def run(self, prompt: str) -> EscalationResult:
        request = Request(
            self._url,
            data=json.dumps({"model": self.model_id, "prompt": prompt, "stream": False}, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                if response.status != 200:
                    return EscalationResult(status="UNAVAILABLE", provider=self.provider, model_id=self.model_id, reason="OLLAMA_HTTP_ERROR")
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("response")
            if not isinstance(content, str) or not content.strip():
                return EscalationResult(status="UNAVAILABLE", provider=self.provider, model_id=self.model_id, reason="OLLAMA_INVALID_RESPONSE")
            return EscalationResult(status="SUCCEEDED", provider=self.provider, model_id=self.model_id, content=content.strip())
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return EscalationResult(status="UNAVAILABLE", provider=self.provider, model_id=self.model_id, reason="OLLAMA_UNAVAILABLE")


class ModelGateway:
    """Evaluates the deterministic egress gate before invoking an adapter."""

    def __init__(self, adapter: EscalationAdapter) -> None:
        self._adapter = adapter

    def escalate(self, prompt: str, request: EgressRequest) -> tuple[EgressDecision, EscalationResult]:
        gate = evaluate_egress(request)
        if not gate.allowed:
            return gate, EscalationResult(
                status="DENIED",
                provider=request.provider,
                model_id=request.model_id,
                reason=gate.reason_codes[0],
            )
        if request.provider != self._adapter.provider or request.model_id != self._adapter.model_id:
            return EgressDecision(allowed=False, rule_ids=("POL-01",), reason_codes=("ADAPTER_IDENTITY_MISMATCH",)), EscalationResult(
                status="DENIED",
                provider=request.provider,
                model_id=request.model_id,
                reason="ADAPTER_IDENTITY_MISMATCH",
            )
        return gate, self._adapter.run(prompt)
