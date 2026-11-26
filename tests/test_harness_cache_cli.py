from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_harness_cache import main


FIXTURE = Path(__file__).parent / "fixtures" / "x_posts"


def _write_labels(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "labels": {
                    "synthetic-road-bandung-001": {
                        "is_complaint": True,
                        "category": "ROAD",
                        "location_quality": "SPECIFIC",
                        "acceptable_statuses": ["COMPLETED"],
                        "notes": "synthetic accepted road complaint",
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_cache_cli_rule_two_passes(tmp_path: Path, capsys: object) -> None:
    labels = tmp_path / "labels.json"
    _write_labels(labels)
    out = tmp_path / "reports"

    exit_code = main(
        [
            "--fixture",
            str(FIXTURE),
            "--labels",
            str(labels),
            "--out",
            str(out),
            "--cache-store",
            "memory",
            "--cache-namespace",
            "cache-cli-test",
            "--analysis-version",
            "m6-test-v1",
            "--threshold",
            "0.5",
            "--passes",
            "2",
            "--inner",
            "rule",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["cache_eligible"] >= 1
    assert payload["exact_hits"] >= 1
    assert payload["false_reuse"] == 0
    assert (out / "cache_report.json").is_file()
    assert (out / "cache_report.md").is_file()

    report = json.loads((out / "cache_report.json").read_text(encoding="utf-8"))
    assert report["baseline"]["total"] == 1
    assert report["cached"]["total"] == 1
    assert report["cache_eligible_hit_rate"] == 1.0
