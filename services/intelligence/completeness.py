from __future__ import annotations

import re
from typing import Iterable

from services.intelligence.ner import EntitySpan

COMPLETENESS_LABELS = ("SUFFICIENT", "INCOMPLETE", "AMBIGUOUS")

_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:acuan|alamat|arah|keberadaan|letak|lokasi|patokan|posisi|titik|koordinat)"
    r"(?:\s+\w+){0,5}\s+(?:(?:masih\s+)?(?:belum|tidak)|kurang)\s+(?:jelas|pasti|rinci|definitif|"
    r"diketahui|spesifik|terinci|terkonfirmasi|terverifikasi)\b",
    re.IGNORECASE,
)
_AMBIGUITY_PHRASES = (
    "patokannya belum jelas",
    "titiknya belum pasti",
    "titiknya kurang spesifik",
    "letak rincinya belum diketahui",
    "titik koordinat spesifik belum terkonfirmasi",
    "acuan titik tepatnya masih belum terinci",
    "keberadaan letak rincinya belum definitif",
    "posisi bidang pastinya belum terverifikasi",
    "petunjuk tempatnya belum lengkap",
    "informasi lokasinya belum jelas",
    "titiknya masih ambigu membingungkan",
    "posisi lapangannya belum teridentifikasi",
)
_ADDRESS_SIGNALS = (
    re.compile(r"\b(?:jl\.?|jalan)\s+[\w.\-]+", re.IGNORECASE),
    re.compile(r"\brt\s*\d+\s*rw\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bkelurahan\s+[\w.\-]+", re.IGNORECASE),
    re.compile(r"\bkecamatan\s+[\w.\-]+", re.IGNORECASE),
    re.compile(r"\b(?:kota|kabupaten)\s+[\w.\-]+", re.IGNORECASE),
)


def resolve_location_completeness(
    text: str,
    entities: Iterable[EntitySpan] = (),
) -> str:
    """Resolve location completeness from explicit ambiguity and address evidence."""
    normalized = " ".join(text.split())
    normalized_lower = normalized.lower()
    if _AMBIGUITY_PATTERN.search(normalized) or any(
        phrase in normalized_lower for phrase in _AMBIGUITY_PHRASES
    ):
        return "AMBIGUOUS"

    entity_text = " ".join(
        entity.text for entity in entities if entity.label.upper() in {"LOC", "LOCATION", "LANDMARK"}
    )
    evidence = f"{normalized} {entity_text}"
    if all(pattern.search(evidence) for pattern in _ADDRESS_SIGNALS):
        return "SUFFICIENT"
    return "INCOMPLETE"
