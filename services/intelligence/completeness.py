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
_FORMAL_ADDRESS_SIGNALS = (
    re.compile(r"\b(?:jl\.?|jalan)\s+[\w.\-]+", re.IGNORECASE),
    re.compile(r"\brt\s*\d+\s*rw\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bkelurahan\s+[\w.\-]+", re.IGNORECASE),
    re.compile(r"\bkecamatan\s+[\w.\-]+", re.IGNORECASE),
    re.compile(r"\b(?:kota|kabupaten)\s+[\w.\-]+", re.IGNORECASE),
)

_STREET_PATTERN = re.compile(
    r"\b(?:jl\.?|jalan)\s+(?!(?:rusak|berlubang|kecil|sempit|dekat|depan|sekitar|menuju|arah)\b)[a-zA-Z0-9\-]+",
    re.IGNORECASE,
)

_SPECIFIC_LANDMARKS = re.compile(
    r"\b(?:miko mall|borma|trans studio|ciwalk|pvj|alun-alun|alun alun|gedung sate|rs immanuel|rs hasan sadikin|rshs|pasar baru|pasar caringin|pasar suci|stasiun bandung|stasiun kiaracondong|tegalega|tegallega|tegalega park|taman tegalega|lapangan tegalega|monumen bandung lautan api|gasibu|lapangan gasibu|monumen perjuangan|monju|bip|bandung indah plaza|becky|balai kota|babakan siliwangi|baksil|taman lansia|itb|unpad|upi|uin|telkom university|stt telkom|kebun binatang|bandung zoo)\b"
    r"|\b(?:mall|borma|pasar|masjid|rs|rumah sakit|spbu|pom bensin|stasiun|terminal|taman|lapangan|monumen|tugu)\s+[A-Za-z0-9\-]+\b",
    re.IGNORECASE,
)

_AREA_SIGNALS = re.compile(
    r"\b(?:kopo|pasteur|dago|buah batu|buahbatu|cibadak|asia afrika|riau|merdeka|supratman|gedebage|antapani|arcamanik|ciroyom|setiabudi|lembang|sukajadi|astanaanyar|lengkong|cikutra|cicadas|kiaracondong|batununggal|soekarno hatta|sukarno hatta|serayu|tegalega|tegallega|peta|bkr|mohkamad toha|moh toha|otista|otto iskandardinata|pasir koja|pasirkoja|sudirman|gardujati|andir|bojongloa|regol|sukasari|cidadap|coblong|cibeunying|bandung wetan|bandung kulon|sumur bandung)\b",
    re.IGNORECASE,
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

    # 1. Full formal administrative address
    if all(pattern.search(evidence) for pattern in _FORMAL_ADDRESS_SIGNALS):
        return "SUFFICIENT"

    # 2. Street + Street Number
    has_street = bool(_STREET_PATTERN.search(evidence))
    has_number = bool(re.search(r"\b(?:no\.?|nomor)\s*\d+\b", evidence, re.IGNORECASE))
    if has_street and has_number:
        return "SUFFICIENT"

    # 3. Recognizable Specific Landmark + Area or Street
    has_landmark = bool(_SPECIFIC_LANDMARKS.search(evidence))
    has_area = bool(_AREA_SIGNALS.search(evidence))
    if has_landmark and (has_area or has_street):
        return "SUFFICIENT"

    if has_street and (has_landmark or has_area):
        return "SUFFICIENT"

    # 4. Two or more distinct Location entities from NER with street and number or landmark
    loc_ents = [e for e in entities if getattr(e, "label", "").upper() in {"LOC", "LOCATION"}]
    if len(loc_ents) >= 2 and (has_landmark or (has_street and has_number)):
        return "SUFFICIENT"

    return "INCOMPLETE"
