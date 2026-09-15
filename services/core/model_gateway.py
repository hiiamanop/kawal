from __future__ import annotations

import json
import os
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


class OmniRouteAdapter:
    """Escalation adapter communicating with OmniRoute proxy (OpenAI-compatible) at localhost:20128."""

    provider = "openai"

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        base_url: str = "http://localhost:20128/v1",
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.model_id = model_id
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key or os.environ.get("OMNIROUTE_API_KEY", "sk-496a5f477e68f0e1-1dd25c-1f86deed")
        self._timeout_seconds = timeout_seconds

    def run(self, prompt: str) -> EscalationResult:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        request = Request(
            self._url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                if response.status != 200:
                    return EscalationResult(
                        status="UNAVAILABLE",
                        provider=self.provider,
                        model_id=self.model_id,
                        reason="OMNIROUTE_HTTP_ERROR",
                    )
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                return EscalationResult(
                    status="UNAVAILABLE",
                    provider=self.provider,
                    model_id=self.model_id,
                    reason="OMNIROUTE_INVALID_RESPONSE",
                )
            message = choices[0].get("message", {})
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                return EscalationResult(
                    status="UNAVAILABLE",
                    provider=self.provider,
                    model_id=self.model_id,
                    reason="OMNIROUTE_EMPTY_RESPONSE",
                )
            return EscalationResult(
                status="SUCCEEDED",
                provider=self.provider,
                model_id=self.model_id,
                content=content.strip(),
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return EscalationResult(
                status="UNAVAILABLE",
                provider=self.provider,
                model_id=self.model_id,
                reason="OMNIROUTE_UNAVAILABLE",
            )

    def disambiguate_complaint(self, text: str) -> dict[str, Any] | None:
        """Perform zero-shot semantic root-cause disentanglement on ambiguous complaints."""
        system_prompt = (
            "Kamu adalah sistem klasifikasi aduan warga Kota Bandung.\n"
            "Kategori yang tersedia:\n"
            "ROAD, DRAINAGE_FLOOD, WASTE, CLEAN_WATER, CIVIL_ADMIN, HEALTH_SERVICE, "
            "PUBLIC_ORDER, TRANSPORTATION, FIRE_RESCUE, SOCIAL_AFFAIRS, EDUCATION, PARKS_HOUSING.\n\n"
            "Tentukan:\n"
            "1. is_complaint: boolean (true jika aduan konkret, false jika sekadar opini/salam biasa)\n"
            "2. root_cause_category: kategori dinas yang menjadi AKAR MASALAH (misal pipa bocor bikin aspal ambles -> CLEAN_WATER)\n"
            "3. secondary_category: kategori dampak (misal ROAD) atau null\n\n"
            "Format respon HANYA JSON:\n"
            '{"is_complaint": bool, "root_cause_category": str, "secondary_category": str|null, "reasoning": str}'
        )
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        request = Request(
            self._url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                if response.status != 200:
                    return None
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices")
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content", "")
            return json.loads(content)
        except Exception:
            return None


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
