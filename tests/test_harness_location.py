from __future__ import annotations

import pytest

from services.harness.location import assess_location


@pytest.mark.parametrize(
    ("location", "evidence", "source", "expected"),
    [
        (
            "Gatot Subroto",
            ["Jl. Gatot Subroto"],
            "Jl. Gatot Subroto Bandung",
            "SPECIFIC",
        ),
        ("Kopo", ["Jl. Kopo"], "Jl. Kopo Bandung", "SPECIFIC"),
        (
            "stasiun selatan",
            ["jl stasiun selatan"],
            "jl stasiun selatan area stasiun bandung",
            "SPECIFIC",
        ),
        (
            "Jalan Cihampelas, di depan Hotel Grand Tjokro",
            ["Jalan Cihampelas"],
            "Lokasi banjir di Jalan Cihampelas, depan Hotel Grand Tjokro",
            "SPECIFIC",
        ),
        (
            "jalan raya di daerah abi seer",
            ["ieu jalan raya di daerah abi seer nu rusak"],
            "mau tanya upami lapor jalan rusak/berlubang ka mana nya? ieu jalan raya di daerah abi seer nu rusak",
            "VAGUE",
        ),
        ("Bandung", ["Bandung"], "bandung ni ujannya gak berhenti", "VAGUE"),
        (None, [], "Jalan rusak", "MISSING"),
        ("", [], "Jalan rusak", "MISSING"),
        (
            "area TSM ke arah Binong",
            ["area TSM"],
            "area TSM ke arah Binong",
            "SPECIFIC",
        ),
    ],
)
def test_assess_location(
    location: str | None,
    evidence: list[str],
    source: str,
    expected: str,
) -> None:
    assessment = assess_location(location, evidence, source)
    assert assessment.quality == expected
    assert assessment.reason_codes
