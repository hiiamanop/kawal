from __future__ import annotations

import json
from pathlib import Path

from scripts.run_harness_x import main
from services.harness.agents import (
    RuleArbiterAgent,
    RuleExtractionAgent,
    RuleTriageAgent,
    RuleVerifierAgent,
)
from services.harness.orchestrator import HarnessOrchestrator
from services.harness.sinks import FileRunSink
from services.harness.sources import ReplayXSource


FIXTURE = (
    Path(__file__).parent / "fixtures" / "x_posts" / "synthetic_road_bandung_001.json"
)
REAL_FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "x_posts"
    / "x_mgt_radio_2026159892885832085.json"
)


def _orchestrator() -> HarnessOrchestrator:
    return HarnessOrchestrator(
        triage=RuleTriageAgent(),
        extraction=RuleExtractionAgent(),
        verifier=RuleVerifierAgent(),
        arbiter=RuleArbiterAgent(),
        max_cost_usd=0.05,
    )


def test_e2e_writes_run_case_trace_and_cost(tmp_path: Path) -> None:
    post = ReplayXSource(FIXTURE).fetch()[0]
    run = _orchestrator().run(post, run_id="run-e2e-1")
    receipt = FileRunSink(tmp_path).write(run)

    output_dir = Path(receipt.output_dir)
    assert (output_dir / "run.json").is_file()
    assert (output_dir / "case.json").is_file()
    assert (output_dir / "trace.jsonl").is_file()
    assert (output_dir / "cost.json").is_file()

    case = json.loads((output_dir / "case.json").read_text(encoding="utf-8"))
    assert case["status"] == "ACCEPTED"
    assert case["case_id"] == run.case.case_id if run.case is not None else False
    assert case["verification"]["is_complaint"] is True


def test_e2e_replay_is_idempotent(tmp_path: Path) -> None:
    post = ReplayXSource(FIXTURE).fetch()[0]
    run_1 = _orchestrator().run(post, run_id="run-e2e-2a")
    run_2 = _orchestrator().run(post, run_id="run-e2e-2b")
    FileRunSink(tmp_path).write(run_1)
    FileRunSink(tmp_path).write(run_2)

    assert run_1.case is not None
    assert run_2.case is not None
    assert run_1.case.case_id == run_2.case.case_id
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 2


def test_real_x_fixture_is_accepted() -> None:
    post = ReplayXSource(REAL_FIXTURE).fetch()[0]
    run = _orchestrator().run(post, run_id="run-real-fixture")
    assert run.case is not None
    assert run.case.status == "ACCEPTED"
    assert run.case.extraction.location == "Gatot Subroto"
    assert run.case.triage.category.value == "ROAD"


def test_cli_returns_zero_and_prints_case_id(tmp_path: Path, capsys: object) -> None:
    exit_code = main(
        [
            "--fixture",
            str(FIXTURE),
            "--out",
            str(tmp_path),
            "--run-id",
            "run-cli-1",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload["status"] == "COMPLETED"
    assert payload["case_id"].startswith("case-x-")
    assert (tmp_path / "run-cli-1" / "case.json").is_file()
