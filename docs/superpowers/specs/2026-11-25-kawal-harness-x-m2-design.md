# KAWAL Harness X — M2 Agent-Agnostic LLM Design

**Status:** draft for implementation
**Date:** 2026-11-25
**Scope:** compare rule-based and LLM-backed Triage/Verifier agents on the same X fixtures, with cost, latency, and agreement reporting.

---

## 1. Intent

M2 proves that the harness is agent-agnostic: the same `Agent` contract can be
filled by deterministic rules or by a remote LLM without changing the
orchestrator, source adapters, sink, or trust rules.

The research question is not "is the LLM smarter?" It is:

> On the same fixtures, how do rule-based and LLM-backed Triage/Verifier agents
> differ in decisions, agreement, cost, and latency?

## 2. Scope

In scope:

- OpenRouter chat-completions client using the existing OpenRouter config.
- LLM-backed Triage and Verifier agents.
- Structured JSON output with Pydantic validation.
- Fail-closed PII guard before remote egress.
- Cost and latency tracking from provider usage or configured prices.
- Rule-vs-LLM comparison runner and CLI.
- Deterministic tests with a fake chat client; live calls are manual only.

Out of scope:

- Prompt optimization or fine-tuning.
- Multiple remote providers in one run.
- LLM-backed Extraction or Arbiter.
- Production deployment, retries beyond a single call, or streaming.
- Live calls in CI.

## 3. Architecture

```text
fixtures
  -> ReplayXSource
  -> two orchestrators
       -> rule orchestrator: RuleTriage / RuleExtraction / RuleVerifier / RuleArbiter
       -> llm orchestrator:  LLMTriage  / RuleExtraction / LLMVerifier / RuleArbiter
  -> two BatchReports
  -> HarnessComparisonRunner
  -> comparison.json + comparison.md
```

The comparison runner only knows `HarnessOrchestrator`; it does not know
whether an agent is rule-based or LLM-backed.

## 4. Chat client contract

```python
class ChatClient(Protocol):
    model_id: str
    provider: str
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResponse: ...
```

`ChatResponse` carries content, model, provider, usage, cost, and latency.
`OpenRouterChatClient` is the only live implementation. Tests use a fake client.

## 5. LLM agent contract

`LLMAgent` implements the existing `Agent` protocol:

```python
class Agent(Protocol):
    agent_id: str
    role: str
    def estimate_cost(self, task: AgentTask) -> float: ...
    def run(self, task: AgentTask) -> AgentResult: ...
```

`LLMTriageAgent` and `LLMVerifierAgent` are thin wrappers that supply:

- a system prompt,
- a user-prompt builder,
- a Pydantic output model.

The Verifier receives the source post and extraction output only. It must not
receive the Triage output.

## 6. Guardrails

- Remote calls are blocked if the source text matches simple PII patterns
  (NIK, phone number, email). The agent returns `UNAVAILABLE` with
  `PII_EGRESS_BLOCKED`; the rule path remains available.
- Output must parse as JSON and validate against the output schema. Invalid
  output returns `INVALID`; the orchestrator turns it into `FAILED` or
  `NEEDS_REVIEW` according to the existing rules.
- Budget is checked before the call using a worst-case token estimate.
- No raw post text is written to application logs.
- No API key is logged or included in reports.

## 7. Cost and comparison

`LLMAgent.estimate_cost` uses configured input/output prices and max tokens.
`AgentUsage` records provider-reported usage when available; otherwise the
client estimates from the configured price.

`HarnessComparisonRunner` runs both orchestrators over the same posts and
reports:

- rule/LLM decision counts,
- category agreement,
- status agreement,
- per-post decisions,
- total and per-post cost,
- total and average latency,
- failures and budget stops.

## 8. Tests

- Fake chat client returns valid JSON: agent status `OK`, output validated,
  usage and cost recorded.
- Fake chat client returns invalid JSON: agent status `INVALID`.
- PII text: agent returns `UNAVAILABLE` without calling the client.
- Verifier prompt excludes Triage output.
- Comparison runner agreement and disagreement reports.
- Comparison CLI rejects a missing OpenRouter key.
- Core CI remains offline and deterministic.

## 9. Acceptance

1. LLM Triage and Verifier implement the existing `Agent` contract.
2. Invalid output and PII fail closed.
3. Rule-vs-LLM comparison writes `comparison.json` and `comparison.md`.
4. Tests run without an API key or network.
5. Local core CI is green.
6. A live OpenRouter comparison is documented but not required for CI.
