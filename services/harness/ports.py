from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from contracts.harness import AgentResult, HarnessRun, RunReceipt, SocialPost


class AgentTask(BaseModel):
    role: str = Field(min_length=1)
    source_text: str
    source_media: tuple[str, ...] = ()
    context: dict[str, object] = Field(default_factory=dict)
    deadline_ms: int = Field(default=5000, ge=1)
    budget_remaining_usd: float = Field(default=0.0, ge=0.0)


class SourcePort(Protocol):
    def fetch(self, query: str, limit: int) -> list[SocialPost]: ...


class Agent(Protocol):
    agent_id: str
    role: str

    def estimate_cost(self, task: AgentTask) -> float: ...

    def run(self, task: AgentTask) -> AgentResult: ...


class RunSink(Protocol):
    def write(self, run: HarnessRun) -> RunReceipt: ...
