from __future__ import annotations

import json
from pathlib import Path

from contracts.harness import HarnessRun, RunReceipt


class FileRunSink:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def write(self, run: HarnessRun) -> RunReceipt:
        output_dir = self._root / run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "run.json").write_text(
            json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if run.case is not None:
            (output_dir / "case.json").write_text(
                json.dumps(run.case.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
            for event in run.trace:
                handle.write(
                    json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n"
                )
        (output_dir / "cost.json").write_text(
            json.dumps(run.cost.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return RunReceipt(run_id=run.run_id, output_dir=str(output_dir))
