from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_cache_scenarios import main
from services.harness.cache_scenarios import (
    CacheScenarioSet,
    evaluate_scenario,
    evaluate_scenarios,
)
from services.intelligence.embedder import TextEmbedder


FIXTURE = Path(__file__).parent.parent / "fixtures" / "x_cache_scenarios.json"


class StubEmbedder(TextEmbedder):
    model_id = "stub-768-v1"
    is_fallback = False

    def embed_text(self, text: str) -> list[float]:
        del text
        return [1.0] + [0.0] * 767

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def _scenarios() -> CacheScenarioSet:
    return CacheScenarioSet.model_validate(
        json.loads(FIXTURE.read_text(encoding="utf-8"))
    )


def test_fixture_loads_all_scenarios() -> None:
    scenarios = _scenarios().scenarios
    assert len(scenarios) == 5
    assert {scenario.scenario_id for scenario in scenarios} == {
        "road-kopo-paraphrase",
        "road-cihampelas-paraphrase",
        "road-kopo-different-location",
        "road-kopo-waste-query",
        "road-kopo-opinion",
    }


def test_positive_scenarios_reuse_with_stub_embedder() -> None:
    scenarios = _scenarios().scenarios
    for scenario in scenarios:
        if not scenario.expected_reuse:
            continue
        result = evaluate_scenario(scenario, embedder=StubEmbedder(), threshold=0.8)
        assert result.seed_stored is True, scenario.scenario_id
        assert result.actual_reuse is True, scenario.scenario_id
        assert result.passed is True, scenario.scenario_id


def test_negative_scenarios_do_not_reuse() -> None:
    scenarios = _scenarios().scenarios
    for scenario in scenarios:
        if scenario.expected_reuse:
            continue
        result = evaluate_scenario(scenario, embedder=StubEmbedder(), threshold=0.8)
        assert result.actual_reuse is False, scenario.scenario_id
        assert result.passed is True, scenario.scenario_id


def test_scenario_cli_writes_report(tmp_path: Path, capsys: object) -> None:
    out = tmp_path / "scenarios"
    exit_code = main(
        [
            "--fixture",
            str(FIXTURE),
            "--out",
            str(out),
            "--embedder",
            "deterministic",
            "--allow-fallback-semantic",
            "--thresholds",
            "0.2,0.6",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["best_threshold"] == 0.6
    assert (out / "scenario_report.json").is_file()
    assert (out / "scenario_report.md").is_file()


def test_threshold_sweep_reports_perfect_routing_for_stub() -> None:
    report = evaluate_scenarios(
        _scenarios().scenarios,
        embedder=StubEmbedder(),
        thresholds=(0.5, 0.8, 0.95),
    )
    assert report.scenarios == 5
    assert report.best_threshold == 0.95
    assert all(metric.accuracy == 1.0 for metric in report.thresholds)
    assert all(metric.precision == 1.0 for metric in report.thresholds)
    assert all(metric.recall == 1.0 for metric in report.thresholds)
