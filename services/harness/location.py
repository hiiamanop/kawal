from __future__ import annotations

import re
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from contracts.harness import LocationQuality


GENERIC_TERMS = {
    "jalan",
    "raya",
    "daerah",
    "area",
    "lokasi",
    "sekitar",
    "sini",
    "sana",
    "itu",
    "ini",
    "depan",
    "belakang",
    "dekat",
    "sisi",
    "arah",
}
GENERIC_STREET_WORDS = {
    "raya",
    "umum",
    "tol",
    "utama",
    "lingkungan",
    "besar",
    "kecil",
    "baru",
    "lama",
}
CITY_TERMS = {
    "bandung",
    "jakarta",
    "surabaya",
    "semarang",
    "medan",
    "bekasi",
    "depok",
    "tangerang",
    "bogor",
    "indonesia",
    "jawa",
    "barat",
    "timur",
    "tengah",
}
LANDMARK_TERMS = (
    "stasiun",
    "terminal",
    "pasar",
    "hotel",
    "mall",
    "gedung",
    "kampus",
    "sekolah",
    "puskesmas",
    "rumah sakit",
    "masjid",
    "gereja",
    "taman",
    "jembatan",
    "simpang",
    "pertigaan",
    "perempatan",
    "kantor",
    "bandara",
    "pelabuhan",
)
STREET_RE = re.compile(r"\b(?:jl\.?|jalan)\s+([A-Za-z0-9.]+)", re.IGNORECASE)
ADMIN_RE = re.compile(
    r"\b(?:kelurahan|kecamatan|rt|rw|gang|gg|nomor|no)\b", re.IGNORECASE
)
PROPER_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")


class LocationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quality: LocationQuality
    reason_codes: tuple[str, ...] = ()


def assess_location(
    location: str | None,
    evidence_spans: Sequence[str] = (),
    source_text: str = "",
) -> LocationAssessment:
    if location is None or not location.strip():
        return LocationAssessment(
            quality="MISSING",
            reason_codes=("LOCATION_MISSING",),
        )

    loc = location.strip().lower()
    evidence = " ".join(evidence_spans).lower()

    if ADMIN_RE.search(loc) or ADMIN_RE.search(evidence):
        return LocationAssessment(
            quality="SPECIFIC",
            reason_codes=("LOCATION_ADMIN_MARKER",),
        )

    for match in STREET_RE.finditer(source_text):
        name = match.group(1).lower().strip(".")
        if name and name not in GENERIC_STREET_WORDS and name in loc:
            return LocationAssessment(
                quality="SPECIFIC",
                reason_codes=("LOCATION_STREET_MARKER",),
            )

    if any(term in loc for term in LANDMARK_TERMS):
        return LocationAssessment(
            quality="SPECIFIC",
            reason_codes=("LOCATION_LANDMARK",),
        )

    for token in PROPER_RE.findall(source_text):
        normalized = token.lower()
        if (
            normalized in loc
            and normalized not in GENERIC_TERMS
            and normalized not in CITY_TERMS
        ):
            return LocationAssessment(
                quality="SPECIFIC",
                reason_codes=("LOCATION_PROPER_NAME",),
            )

    if any(term in loc for term in GENERIC_TERMS):
        return LocationAssessment(
            quality="VAGUE",
            reason_codes=("LOCATION_VAGUE",),
        )

    tokens = {token for token in re.split(r"[^a-z0-9]+", loc) if token}
    if not tokens or tokens <= (GENERIC_TERMS | CITY_TERMS):
        return LocationAssessment(
            quality="VAGUE",
            reason_codes=("LOCATION_VAGUE",),
        )

    return LocationAssessment(
        quality="VAGUE",
        reason_codes=("LOCATION_NO_SPECIFIC_MARKER",),
    )
