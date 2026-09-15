from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.core.decision import EgressRequest


class EscalationCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=4096)
    egress_request: EgressRequest
    idempotency_key: str = Field(min_length=1)
    payload_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def valid_identity(self) -> EscalationCommand:
        expected = f"{self.tenant_id}:{self.case_id}:escalation:{self.task_id}:v1"
        if self.idempotency_key != expected:
            raise ValueError("idempotency_key must use the canonical escalation format")
        return self


def build_anonymized_escalation_command(
    tenant_id: str,
    case_id: str,
    revision: int,
    task_id: str,
    category: str,
    risk: str,
    completeness: str,
    model_id: str = "qwen2.5:7b",
) -> EscalationCommand:
    prompt = (
        "Tentukan apakah metadata aduan anonim berikut memerlukan klarifikasi lokasi. "
        "Jawab singkat dengan KLARIFIKASI atau CUKUP. "
        f"Kategori={category}; risiko={risk}; kelengkapan_lokasi={completeness}."
    )
    key = f"{tenant_id}:{case_id}:escalation:{task_id}:v1"
    return EscalationCommand(
        tenant_id=tenant_id,
        case_id=case_id,
        revision=revision,
        task_id=task_id,
        prompt=prompt,
        egress_request=EgressRequest(
            provider="local",
            model_id=model_id,
            data_class="internal_proxy",
            has_pii=False,
            cost_usd=0.0,
        ),
        idempotency_key=key,
        payload_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )
