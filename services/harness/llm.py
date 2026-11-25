from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Literal, Protocol
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from contracts.harness import (
    AgentResult,
    AgentUsage,
    ArbitrationOutput,
    ExtractionOutput,
    TriageOutput,
    VerificationOutput,
)
from contracts.models import Category, RiskLevel
from services.harness.ports import AgentTask
from services.llm_config import (
    openrouter_api_key,
    openrouter_base_url,
    openrouter_headers,
    openrouter_model,
)


class ChatClientError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ChatUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    cost_source: Literal["provider", "estimated", "local"] = "estimated"


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    model: str
    provider: str = "openai"
    usage: ChatUsage = Field(default_factory=ChatUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)


class ChatClient(Protocol):
    model_id: str
    provider: str

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float: ...

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResponse: ...


class OpenRouterChatClient:
    """Small OpenRouter chat-completions adapter for harness agents."""

    provider = "openai"

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
        timeout_seconds: float = 30.0,
        reasoning_enabled: bool | None = None,
    ) -> None:
        self.model_id = openrouter_model(model_id)
        self._base_url = openrouter_base_url(base_url)
        self._api_key = api_key
        self._input_price = input_price_per_million
        self._output_price = output_price_per_million
        self._timeout_seconds = timeout_seconds
        self._reasoning_enabled = reasoning_enabled

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self._input_price + output_tokens * self._output_price
        ) / 1_000_000.0

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResponse:
        if not openrouter_api_key(self._api_key):
            raise ChatClientError(
                "MISSING_API_KEY", "OPENROUTER_API_KEY is not configured"
            )

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": self._reasoning_enabled}
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=openrouter_headers(self._api_key),
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_seconds or self._timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            raise ChatClientError("OPENROUTER_UNAVAILABLE", type(exc).__name__) from exc
        except json.JSONDecodeError as exc:
            raise ChatClientError(
                "OPENROUTER_INVALID_RESPONSE", "Invalid JSON envelope"
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ChatClientError("OPENROUTER_INVALID_RESPONSE", "Missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ChatClientError("OPENROUTER_EMPTY_RESPONSE", "Empty model content")

        usage = body.get("usage") if isinstance(body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        provider_cost = usage.get("cost")
        if provider_cost is None:
            provider_cost = body.get("total_cost") if isinstance(body, dict) else None
        if isinstance(provider_cost, (int, float)):
            cost_usd = float(provider_cost)
            cost_source: Literal["provider", "estimated"] = "provider"
        else:
            cost_usd = self.estimate_cost(input_tokens, output_tokens)
            cost_source = "estimated"

        return ChatResponse(
            content=content.strip(),
            model=self.model_id,
            provider=self.provider,
            usage=ChatUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                cost_source=cost_source,
            ),
            latency_ms=latency_ms,
        )


_PII_PATTERNS = (
    re.compile(r"\b\d{16}\b"),
    re.compile(r"(?:\+62|62|0)8\d{7,12}\b"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
)


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PII_PATTERNS)


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    candidate = match.group(1) if match else cleaned
    try:
        parsed = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _as_confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.0, number))


def _normalize_category(value: Any, is_complaint: bool) -> str:
    if isinstance(value, str):
        clean = value.strip().upper()
        if clean in Category._value2member_map_:
            return clean
    if not is_complaint:
        return Category.ROAD.value
    raise ValueError("invalid category for a complaint")


def _normalize_triage(data: dict[str, Any]) -> dict[str, Any]:
    is_complaint = bool(data.get("is_complaint", False))
    risk_value = str(data.get("risk", "MEDIUM")).strip().upper()
    if risk_value not in RiskLevel._value2member_map_:
        risk_value = RiskLevel.MEDIUM.value
    return {
        "is_complaint": is_complaint,
        "category": _normalize_category(data.get("category"), is_complaint),
        "risk": risk_value,
        "confidence": _as_confidence(data.get("confidence")),
        "reason_codes": _as_string_tuple(data.get("reason_codes")),
    }


def _normalize_extraction(data: dict[str, Any]) -> dict[str, Any]:
    def optional_text(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    return {
        "location": optional_text(data.get("location")),
        "object": optional_text(data.get("object")),
        "time": optional_text(data.get("time")),
        "evidence_spans": _as_string_tuple(data.get("evidence_spans")),
        "confidence": _as_confidence(data.get("confidence")),
    }


def _normalize_verification(data: dict[str, Any]) -> dict[str, Any]:
    is_complaint = bool(data.get("is_complaint", False))
    return {
        "is_complaint": is_complaint,
        "category": _normalize_category(data.get("category"), is_complaint),
        "evidence_supported": bool(data.get("evidence_supported", False)),
        "issues": _as_string_tuple(data.get("issues")),
        "confidence": _as_confidence(data.get("confidence")),
    }


def _normalize_arbitration(data: dict[str, Any]) -> dict[str, Any]:
    is_complaint = bool(data.get("is_complaint", False))
    rationale = data.get("rationale")
    return {
        "is_complaint": is_complaint,
        "category": _normalize_category(data.get("category"), is_complaint),
        "rationale": rationale.strip()
        if isinstance(rationale, str) and rationale.strip()
        else "Arbiter output incomplete.",
        "confidence": _as_confidence(data.get("confidence")),
    }


class LLMAgent:
    """Agent backed by a chat client; the orchestrator sees only AgentResult."""

    def __init__(
        self,
        *,
        agent_id: str,
        role: str,
        client: ChatClient,
        system_prompt: str,
        user_prompt_builder: Callable[[AgentTask], str],
        output_model: type[BaseModel],
        output_normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_input_tokens: int = 2000,
        max_output_tokens: int = 800,
        prompt_version: str = "llm-v1",
        block_pii: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self._client = client
        self._system_prompt = system_prompt
        self._user_prompt_builder = user_prompt_builder
        self._output_model = output_model
        self._output_normalizer = output_normalizer
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._prompt_version = prompt_version
        self._block_pii = block_pii

    def estimate_cost(self, task: AgentTask) -> float:
        del task
        return self._client.estimate_cost(
            self._max_input_tokens, self._max_output_tokens
        )

    def run(self, task: AgentTask) -> AgentResult:
        if self._block_pii and contains_pii(task.source_text):
            return self._unavailable("PII_EGRESS_BLOCKED", latency_ms=0.0)

        started = time.perf_counter()
        try:
            response = self._client.complete(
                self._system_prompt,
                self._user_prompt_builder(task),
                max_tokens=self._max_output_tokens,
                timeout_seconds=30.0,
            )
        except ChatClientError as exc:
            return self._unavailable(
                exc.code, latency_ms=(time.perf_counter() - started) * 1000.0
            )
        except Exception as exc:
            return self._unavailable(
                type(exc).__name__,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        usage = AgentUsage(
            model=response.model,
            provider=response.provider,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=response.usage.cost_usd,
            cost_source=response.usage.cost_source,
            latency_ms=response.latency_ms,
        )
        parsed = _extract_json(response.content)
        if parsed is None:
            return AgentResult(
                agent_id=self.agent_id,
                role=self.role,
                status="INVALID",
                output=None,
                usage=usage,
                prompt_version=self._prompt_version,
                error="INVALID_JSON",
            )
        try:
            if self._output_normalizer is not None:
                parsed = self._output_normalizer(parsed)
            output = self._output_model.model_validate(parsed)
        except Exception:
            return AgentResult(
                agent_id=self.agent_id,
                role=self.role,
                status="INVALID",
                output=None,
                usage=usage,
                prompt_version=self._prompt_version,
                error="INVALID_SCHEMA",
            )
        return AgentResult(
            agent_id=self.agent_id,
            role=self.role,
            status="OK",
            output=output.model_dump(mode="json"),
            usage=usage,
            prompt_version=self._prompt_version,
        )

    def _unavailable(self, error: str, *, latency_ms: float) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            role=self.role,
            status="UNAVAILABLE",
            output=None,
            usage=AgentUsage(
                model=self._client.model_id,
                provider=self._client.provider,
                cost_usd=0.0,
                cost_source="estimated",
                latency_ms=latency_ms,
            ),
            prompt_version=self._prompt_version,
            error=error,
        )


def _category_values() -> str:
    return ", ".join(category.value for category in Category)


def _risk_values() -> str:
    return ", ".join(risk.value for risk in RiskLevel)


class LLMTriageAgent(LLMAgent):
    def __init__(self, client: ChatClient, **kwargs: Any) -> None:
        system_prompt = (
            "You classify Indonesian public complaints for a municipal service. "
            "Return ONLY one JSON object with exactly these keys: is_complaint "
            "(boolean), category, risk, confidence (number 0..1), reason_codes "
            "(array of short strings). Include every key, even when reason_codes is "
            f"empty. category must be one of: {_category_values()}. "
            f"risk must be one of: {_risk_values()}. "
            "No markdown, no extra keys, no explanation outside the JSON."
        )
        super().__init__(
            agent_id="llm-triage-v1",
            role="triage",
            client=client,
            system_prompt=system_prompt,
            user_prompt_builder=lambda task: task.source_text,
            output_model=TriageOutput,
            output_normalizer=_normalize_triage,
            **kwargs,
        )


class LLMVerifierAgent(LLMAgent):
    def __init__(self, client: ChatClient, **kwargs: Any) -> None:
        system_prompt = (
            "You are an independent verifier for Indonesian public complaints. "
            "Decide independently whether the post is a complaint and which category "
            "it belongs to. Check whether the extraction fields are supported by the "
            "source text. Return ONLY one JSON object with exactly these keys: "
            "is_complaint (boolean), category, evidence_supported (boolean), issues "
            "(array of short strings), confidence (number 0..1). Include every key; "
            "use an empty issues array when there are no issues. "
            f"category must be one of: {_category_values()}. "
            "Do not assume another agent's answer. No markdown, no extra keys."
        )

        def build_user_prompt(task: AgentTask) -> str:
            extraction = task.context.get("extraction")
            return (
                f"Source post:\n{task.source_text}\n\n"
                "Extraction to verify:\n"
                f"{json.dumps(extraction, ensure_ascii=False, sort_keys=True)}"
            )

        super().__init__(
            agent_id="llm-verifier-v1",
            role="verifier",
            client=client,
            system_prompt=system_prompt,
            user_prompt_builder=build_user_prompt,
            output_model=VerificationOutput,
            output_normalizer=_normalize_verification,
            **kwargs,
        )


class LLMExtractionAgent(LLMAgent):
    def __init__(self, client: ChatClient, **kwargs: Any) -> None:
        system_prompt = (
            "Extract factual complaint fields from the Indonesian source text. "
            "Return ONLY one JSON object with exactly these keys: location (string "
            "or null), object (string or null), time (string or null), evidence_spans "
            "(array of exact substrings copied from the source), confidence (number "
            "0..1). Include every key; use null when a field is absent and an empty "
            "array when there are no evidence spans. Never invent a location or an "
            "object that is not supported by the text. No markdown, no extra keys."
        )
        super().__init__(
            agent_id="llm-extraction-v1",
            role="extraction",
            client=client,
            system_prompt=system_prompt,
            user_prompt_builder=lambda task: task.source_text,
            output_model=ExtractionOutput,
            output_normalizer=_normalize_extraction,
            **kwargs,
        )


class LLMArbiterAgent(LLMAgent):
    def __init__(self, client: ChatClient, **kwargs: Any) -> None:
        system_prompt = (
            "You are an explanatory arbiter for a complaint harness. Two agents "
            "disagree. Return ONLY one JSON object with exactly these keys: "
            "is_complaint (boolean), category, rationale (short string), confidence "
            "(number 0..1). Include every key. "
            f"category must be one of: {_category_values()}. "
            "You explain the conflict; you do not authorize automatic acceptance. "
            "No markdown, no extra keys."
        )

        def build_user_prompt(task: AgentTask) -> str:
            return (
                f"Source post:\n{task.source_text}\n\n"
                "Triage output:\n"
                f"{json.dumps(task.context.get('triage'), ensure_ascii=False, sort_keys=True)}\n\n"
                "Verification output:\n"
                f"{json.dumps(task.context.get('verification'), ensure_ascii=False, sort_keys=True)}"
            )

        super().__init__(
            agent_id="llm-arbiter-v1",
            role="arbiter",
            client=client,
            system_prompt=system_prompt,
            user_prompt_builder=build_user_prompt,
            output_model=ArbitrationOutput,
            output_normalizer=_normalize_arbitration,
            max_output_tokens=500,
            **kwargs,
        )
