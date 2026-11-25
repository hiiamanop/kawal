from __future__ import annotations

import json
from pathlib import Path

from scripts.run_harness_x_batch import main


def _write_fixture(path: Path, post_id: str, text: str) -> None:
    path.write_text(
        json.dumps(
            {
                "post": {
                    "post_id": post_id,
                    "platform": "x",
                    "url": f"https://x.com/example/status/{post_id}",
                    "author_handle": "warga",
                    "text": text,
                    "collected_at": "2026-11-25T00:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )


def test_batch_cli_processes_fixture_directory(tmp_path: Path, capsys: object) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    _write_fixture(
        fixtures / "accepted.json",
        "cli-accepted",
        "Min, jalan di Jl. Kopo Bandung rusak parah, ada lubang besar.",
    )
    _write_fixture(fixtures / "rejected.json", "cli-rejected", "Selamat pagi min")

    out = tmp_path / "runs"
    captured_fixtures = tmp_path / "captured"
    exit_code = main(
        [
            "--fixture",
            str(fixtures),
            "--out",
            str(out),
            "--save-fixtures",
            str(captured_fixtures),
            "--batch-id",
            "cli-batch-1",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["batch_id"] == "cli-batch-1"
    assert output["total_posts"] == 2
    assert output["accepted"] == 1
    assert output["rejected"] == 1
    assert (out / "batch-cli-batch-1" / "summary.json").is_file()
    assert len(list(captured_fixtures.glob("*.json"))) == 2
