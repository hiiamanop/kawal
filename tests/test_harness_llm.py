from __future__ import annotations

import json

import pytest

from services.harness.llm import (
    ChatClientError,
    ChatResponse,
    ChatUsage,
    LLMArbiterAgent,
    LLMExtractionAgent,
    LLMTriageAgent,
    LLMVerifierAgent,
    OpenRouterChatClient,
)
from services.harness.ports import AgentTask


class FakeChatClient:
    model_id = "fake-model"
    provider = "openai"

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        del input_tokens, output_tokens
        return 0.001

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResponse:
        del max_tokens, timeout_seconds
        self.calls += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return ChatResponse(
            content=self._content,
            model=self.model_id,
            provider=self.provider,
            usage=ChatUsage(
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
                cost_source="estimated",
            ),
            latency_ms=12.0,
        )


def _task(text: str = "Jalan rusak di Jl. Kopo Bandung") -> AgentTask:
    return AgentTask(role="triage", source_text=text)


def test_llm_triage_valid_output() -> None:
    content = json.dumps(
        {
            "is_complaint": True,
            "category": "ROAD",
            "risk": "HIGH",
            "confidence": 0.9,
            "reason_codes": ["ROAD_DAMAGE"],
        }
    )
    client = FakeChatClient(content)
    agent = LLMTriageAgent(client)
    result = agent.run(_task())
    assert result.status == "OK"
    assert result.output is not None
    assert result.output["category"] == "ROAD"
    assert result.usage.cost_usd == 0.001
    assert result.usage.model == "fake-model"
    assert agent.estimate_cost(_task()) == 0.001


def test_llm_triage_invalid_json() -> None:
    result = LLMTriageAgent(FakeChatClient("not json")).run(_task())
    assert result.status == "INVALID"
    assert result.error == "INVALID_JSON"


def test_llm_agent_blocks_pii_before_calling_client() -> None:
    client = FakeChatClient("{}")
    result = LLMTriageAgent(client).run(_task("NIK 3273010101010001, tolong lapor"))
    assert result.status == "UNAVAILABLE"
    assert result.error == "PII_EGRESS_BLOCKED"
    assert client.calls == 0


def test_llm_verifier_prompt_excludes_triage_output() -> None:
    content = json.dumps(
        {
            "is_complaint": True,
            "category": "ROAD",
            "evidence_supported": True,
            "issues": [],
            "confidence": 0.8,
        }
    )
    client = FakeChatClient(content)
    agent = LLMVerifierAgent(client)
    task = AgentTask(
        role="verifier",
        source_text="Jalan rusak di Jl. Kopo Bandung",
        context={
            "extraction": {
                "location": "Kopo",
                "object": "jalan rusak",
                "time": None,
                "evidence_spans": ["Jl. Kopo"],
                "confidence": 0.8,
            },
            "triage": {
                "is_complaint": True,
                "category": "WASTE",
                "risk": "LOW",
                "confidence": 0.9,
                "reason_codes": [],
            },
        },
    )
    result = agent.run(task)
    assert result.status == "OK"
    assert "triage" not in client.last_user_prompt.lower()
    assert "waste" not in client.last_user_prompt.lower()


def test_openrouter_client_requires_key_without_network() -> None:
    client = OpenRouterChatClient(api_key="")
    with pytest.raises(ChatClientError) as exc:
        client.complete("system", "user", max_tokens=10, timeout_seconds=1.0)
    assert exc.value.code == "MISSING_API_KEY"


def test_llm_extraction_valid_output() -> None:
    content = json.dumps(
        {
            "location": "Kopo",
            "object": "lubang",
            "time": None,
            "evidence_spans": ["Jl. Kopo", "lubang"],
            "confidence": 0.85,
        }
    )
    result = LLMExtractionAgent(FakeChatClient(content)).run(_task())
    assert result.status == "OK"
    assert result.output is not None
    assert result.output["location"] == "Kopo"


def test_llm_extraction_normalizes_missing_fields() -> None:
    content = json.dumps({"location": "Kopo"})
    result = LLMExtractionAgent(FakeChatClient(content)).run(_task())
    assert result.status == "OK"
    assert result.output is not None
    assert result.output["object"] is None
    assert result.output["evidence_spans"] == []
    assert result.output["confidence"] == 0.5


def test_llm_verifier_normalizes_missing_issues() -> None:
    content = json.dumps(
        {
            "is_complaint": True,
            "category": "ROAD",
            "evidence_supported": True,
            "confidence": 0.7,
        }
    )
    result = LLMVerifierAgent(FakeChatClient(content)).run(_task())
    assert result.status == "OK"
    assert result.output is not None
    assert result.output["issues"] == []


def test_llm_arbiter_valid_output_and_prompt() -> None:
    content = json.dumps(
        {
            "is_complaint": True,
            "category": "ROAD",
            "rationale": "The road damage is the root cause.",
            "confidence": 0.7,
        }
    )
    client = FakeChatClient(content)
    agent = LLMArbiterAgent(client)
    task = AgentTask(
        role="arbiter",
        source_text="Jalan rusak di Jl. Kopo Bandung",
        context={
            "triage": {"is_complaint": True, "category": "ROAD"},
            "verification": {"is_complaint": True, "category": "WASTE"},
        },
    )
    result = agent.run(task)
    assert result.status == "OK"
    assert result.output is not None
    assert result.output["rationale"]
    assert "Triage output" in client.last_user_prompt
    assert "Verification output" in client.last_user_prompt
