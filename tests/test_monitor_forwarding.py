from __future__ import annotations

import json
from pathlib import Path

from services.monitor.forwarding import write_forward_packet

from tests.test_monitor_helpers import _monitor_case_with_note


def test_forward_packet_is_a_local_draft(tmp_path: Path) -> None:
    case = _monitor_case_with_note()
    packet = write_forward_packet(case, "Mohon diteruskan ke DSDABM.", tmp_path)
    md = Path(packet.markdown_path).read_text()
    data = json.loads(Path(packet.json_path).read_text())
    assert "DRAFT — NOT SENT" in md
    assert case.case_id in md
    assert data["case_id"] == case.case_id
    assert data["department"] == case.department
    assert data["trust_decision"] == case.run.case.trust_decision
    assert data["evidence_spans"] == list(case.run.case.extraction.evidence_spans)
    assert "Evidence spans" in md
    assert "Trust decision" in md
