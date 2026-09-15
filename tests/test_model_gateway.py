from __future__ import annotations

from services.core.decision import EgressRequest
from services.core.escalation import build_anonymized_escalation_command
from services.core.model_gateway import EscalationResult, ModelGateway


class RecordingAdapter:
    provider = "local"
    model_id = "qwen2.5:7b"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def run(self, prompt: str) -> EscalationResult:
        self.prompts.append(prompt)
        return EscalationResult(status="SUCCEEDED", provider=self.provider, model_id=self.model_id, content="verified")


def test_anonymized_escalation_command_excludes_citizen_text() -> None:
    command = build_anonymized_escalation_command(
        tenant_id="tenant-1",
        case_id="case-1",
        revision=1,
        task_id="local-clarification-check",
        category="ROAD",
        risk="HIGH",
        completeness="INCOMPLETE",
    )

    assert "citizen" not in command.prompt.lower()
    assert "Jl." not in command.prompt
    assert command.egress_request.has_pii is False
    assert command.idempotency_key == "tenant-1:case-1:escalation:local-clarification-check:v1"


def test_model_gateway_blocks_pii_before_adapter_invocation() -> None:
    adapter = RecordingAdapter()
    gateway = ModelGateway(adapter)

    gate, result = gateway.escalate(
        "contains citizen identity",
        EgressRequest(provider="local", model_id="qwen2.5:7b", data_class="public", has_pii=True),
    )

    assert gate.allowed is False
    assert result.status == "DENIED"
    assert result.reason == "PII_EGRESS_BLOCKED"
    assert adapter.prompts == []


def test_model_gateway_allows_local_non_pii_escalation() -> None:
    adapter = RecordingAdapter()
    gateway = ModelGateway(adapter)

    gate, result = gateway.escalate(
        "Classify whether the anonymized complaint needs clarification.",
        EgressRequest(provider="local", model_id="qwen2.5:7b", data_class="internal_proxy"),
    )

    assert gate.allowed is True
    assert result.status == "SUCCEEDED"
    assert result.content == "verified"
    assert len(adapter.prompts) == 1


def test_model_gateway_rejects_adapter_identity_mismatch() -> None:
    adapter = RecordingAdapter()
    gateway = ModelGateway(adapter)

    gate, result = gateway.escalate(
        "safe prompt",
        EgressRequest(provider="local", model_id="other-model", data_class="public"),
    )

    assert gate.allowed is False
    assert result.reason == "ADAPTER_IDENTITY_MISMATCH"
    assert adapter.prompts == []
