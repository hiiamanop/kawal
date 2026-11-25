from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_harness_agents import main
from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.cassette import CassetteChatClient, CassetteError
from services.harness.compare import HarnessComparisonRunner
from services.harness.llm import (
    ChatClientError,
    ChatResponse,
    ChatUsage,
    LLMTriageAgent,
    LLMVerifierAgent,
)
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sources import ReplayXSource


FIXTURE = (
    Path(__file__).parent / "fixtures" / "x_posts" / "synthetic_road_bandung_001.json"
)


class StubChatClient:
    model_id = "stub-model"
    provider = "stub-provider"

    def __init__(
        self, content: str = '{"ok": true}', error: ChatClientError | None = None
    ) -> None:
        self._content = content
        self._error = error
        self.calls = 0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * 2.0 + output_tokens * 3.0) / 1_000_000.0

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResponse:
        del system_prompt, user_prompt, max_tokens, timeout_seconds
        self.calls += 1
        if self._error is not None:
            raise self._error
        return ChatResponse(
            content=self._content,
            model=self.model_id,
            provider=self.provider,
            usage=ChatUsage(
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.0001,
                cost_source="estimated",
            ),
            latency_ms=12.0,
        )


class RoleAwareChatClient(StubChatClient):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResponse:
        if "You classify" in system_prompt:
            self._content = json.dumps(
                {
                    "is_complaint": True,
                    "category": "ROAD",
                    "risk": "HIGH",
                    "confidence": 0.9,
                    "reason_codes": ["ROAD_DAMAGE"],
                }
            )
        elif "independent verifier" in system_prompt:
            self._content = json.dumps(
                {
                    "is_complaint": True,
                    "category": "ROAD",
                    "evidence_supported": True,
                    "issues": [],
                    "confidence": 0.8,
                }
            )
        return super().complete(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )


def test_cassette_records_and_replays_without_raw_prompts(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    inner = StubChatClient('{"category": "ROAD"}')
    recorder = CassetteChatClient(
        path,
        mode="record",
        inner=inner,
        input_price_per_million=2.0,
        output_price_per_million=3.0,
    )
    response = recorder.complete(
        "system instruction",
        "source post: Jalan rusak di Kopo",
        max_tokens=100,
        timeout_seconds=30.0,
    )
    assert response.content == '{"category": "ROAD"}'
    assert inner.calls == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "Jalan rusak" not in json.dumps(payload)
    assert "system instruction" not in json.dumps(payload)
    assert payload["entries"][0]["request_key"]

    replay = CassetteChatClient(path, mode="replay")
    replayed = replay.complete(
        "system instruction",
        "source post: Jalan rusak di Kopo",
        max_tokens=100,
        timeout_seconds=30.0,
    )
    assert replayed.content == '{"category": "ROAD"}'
    assert replayed.usage.cost_usd == 0.0001
    assert replay.estimate_cost(1_000_000, 0) == 2.0


def test_cassette_replay_miss_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    recorder = CassetteChatClient(path, mode="record", inner=StubChatClient())
    recorder.complete("system", "one", max_tokens=10, timeout_seconds=30.0)
    replay = CassetteChatClient(path, mode="replay")

    with pytest.raises(ChatClientError) as exc:
        replay.complete("system", "two", max_tokens=10, timeout_seconds=30.0)
    assert exc.value.code == "CASSETTE_MISS"


def test_cassette_records_and_replays_errors(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    inner = StubChatClient(error=ChatClientError("OPENROUTER_EMPTY_RESPONSE", "empty"))
    recorder = CassetteChatClient(path, mode="record", inner=inner)
    with pytest.raises(ChatClientError) as recorded:
        recorder.complete("system", "user", max_tokens=10, timeout_seconds=30.0)
    assert recorded.value.code == "OPENROUTER_EMPTY_RESPONSE"

    replay = CassetteChatClient(path, mode="replay")
    with pytest.raises(ChatClientError) as replayed:
        replay.complete("system", "user", max_tokens=10, timeout_seconds=30.0)
    assert replayed.value.code == "OPENROUTER_EMPTY_RESPONSE"


def test_cassette_does_not_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    first = CassetteChatClient(path, mode="record", inner=StubChatClient())
    first.complete("system", "user", max_tokens=10, timeout_seconds=30.0)
    with pytest.raises(CassetteError) as exc:
        CassetteChatClient(path, mode="record", inner=StubChatClient())
    assert exc.value.code == "CASSETTE_EXISTS"


def test_cassette_replay_requires_file(tmp_path: Path) -> None:
    with pytest.raises(CassetteError) as exc:
        CassetteChatClient(tmp_path / "missing.json", mode="replay")
    assert exc.value.code == "CASSETTE_NOT_FOUND"


def test_compare_cli_replays_cassette_without_api_key(
    tmp_path: Path, monkeypatch: object
) -> None:
    cassette_path = tmp_path / "compare.json"
    recorder = CassetteChatClient(
        cassette_path, mode="record", inner=RoleAwareChatClient()
    )
    rule_orchestrator = HarnessOrchestrator(
        triage=RuleTriageAgent(),
        extraction=RuleExtractionAgent(),
        verifier=RuleVerifierAgent(),
        arbiter=RuleArbiterAgent(),
        max_cost_usd=0.05,
    )
    llm_orchestrator = HarnessOrchestrator(
        triage=LLMTriageAgent(recorder),
        extraction=RuleExtractionAgent(),
        verifier=LLMVerifierAgent(recorder),
        arbiter=RuleArbiterAgent(),
        max_cost_usd=0.05,
    )
    posts = ReplayXSource(FIXTURE).fetch()
    HarnessComparisonRunner(
        tmp_path / "recorded", rule_orchestrator, llm_orchestrator
    ).run(posts, comparison_id="recorded")
    assert len(recorder.entries) == 2

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("KAWAL_OPENROUTER_API_KEY", raising=False)  # type: ignore[attr-defined]
    output = tmp_path / "replayed"
    exit_code = main(
        [
            "--fixture",
            str(FIXTURE),
            "--out",
            str(output),
            "--comparison-id",
            "replayed",
            "--cassette",
            str(cassette_path),
            "--replay-cassette",
        ]
    )
    assert exit_code == 0
    assert (output / "comparison-replayed" / "comparison.json").is_file()
