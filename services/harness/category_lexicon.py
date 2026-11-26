from __future__ import annotations

import re

from contracts.models import Category


TRIAGE_CUES: dict[Category, tuple[str, ...]] = {
    Category.ROAD: (
        "jalan rusak",
        "jalan berlubang",
        "jalan",
        "lubang",
        "aspal",
        "trotoar",
        "bahu jalan",
        "jembatan",
    ),
    Category.DRAINAGE_FLOOD: (
        "banjir",
        "drainase",
        "gorong-gorong",
        "selokan",
        "parit",
        "genangan",
    ),
    Category.WASTE: (
        "sampah",
        "limbah",
        "tps",
        "tumpukan sampah",
    ),
    Category.CLEAN_WATER: (
        "pdam",
        "pipa air",
        "pipa bocor",
        "air keruh",
        "kran mati",
        "leding",
    ),
    Category.CIVIL_ADMIN: (
        "ktp",
        "kartu keluarga",
        "akta",
        "kependudukan",
        "disdukcapil",
    ),
    Category.HEALTH_SERVICE: (
        "puskesmas",
        "rumah sakit",
        "dokter",
        "obat",
        "bidan",
        "ambulans",
    ),
    Category.PUBLIC_ORDER: (
        "pungli",
        "parkir liar",
        "pedagang kaki lima",
        "ketertiban",
        "balap liar",
    ),
    Category.TRANSPORTATION: (
        "angkot",
        "bus",
        "halte",
        "lampu lalu lintas",
        "rambu",
        "zebra cross",
    ),
    Category.FIRE_RESCUE: (
        "kebakaran",
        "pemadam",
        "pohon tumbang",
        "evakuasi",
        "tabung gas",
    ),
    Category.SOCIAL_AFFAIRS: (
        "bansos",
        "bantuan sosial",
        "lansia",
        "anak terlantar",
        "panti",
    ),
    Category.EDUCATION: (
        "sekolah",
        "guru",
        "kelas",
        "buku",
        "seragam",
        "laboratorium",
    ),
    Category.PARKS_HOUSING: (
        "taman",
        "bangku",
        "lampu taman",
        "rumah susun",
        "rusun",
        "bangunan",
    ),
}

OBJECT_CUES: dict[Category, tuple[str, ...]] = {
    Category.ROAD: ("jalan rusak", "jalan berlubang", "lubang", "aspal", "trotoar"),
    Category.DRAINAGE_FLOOD: ("gorong-gorong", "drainase", "selokan", "banjir"),
    Category.WASTE: ("tumpukan sampah", "sampah", "limbah", "tps"),
    Category.CLEAN_WATER: ("pipa bocor", "pipa air", "air keruh", "pdam", "leding"),
    Category.CIVIL_ADMIN: ("kartu keluarga", "ktp", "akta", "berkas kependudukan"),
    Category.HEALTH_SERVICE: ("puskesmas", "rumah sakit", "ambulans", "obat"),
    Category.PUBLIC_ORDER: (
        "parkir liar",
        "pungli",
        "pedagang kaki lima",
        "balap liar",
    ),
    Category.TRANSPORTATION: ("lampu lalu lintas", "angkot", "halte", "rambu"),
    Category.FIRE_RESCUE: ("kebakaran", "pohon tumbang", "tabung gas", "pemadam"),
    Category.SOCIAL_AFFAIRS: ("bantuan sosial", "bansos", "lansia", "anak terlantar"),
    Category.EDUCATION: ("sekolah", "seragam", "laboratorium", "buku"),
    Category.PARKS_HOUSING: ("lampu taman", "taman", "bangku", "rumah susun"),
}


def _contains(lower: str, cue: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", lower) is not None


def _matches(lower: str, cues: tuple[str, ...]) -> list[str]:
    return [cue for cue in cues if _contains(lower, cue)]


def match_category(text: str) -> tuple[Category | None, bool]:
    """Return the best category and whether multiple categories matched.

    Ambiguity means cues from more than one category appear. The conservative
    trust model sends those cases to review instead of letting one keyword win.
    """

    lower = text.strip().lower()
    if not lower:
        return None, False

    scores: dict[Category, int] = {}
    for category, cues in TRIAGE_CUES.items():
        matched = _matches(lower, cues)
        if matched:
            scores[category] = sum(len(cue) for cue in matched)

    if not scores:
        return None, False

    ranked = sorted(scores.values(), reverse=True)
    best = max(scores, key=lambda category: scores[category])
    ambiguous = len(ranked) > 1 and ranked[1] >= 0.60 * ranked[0]
    return best, ambiguous


def match_object(text: str) -> tuple[str | None, Category | None]:
    """Return the longest recognised object phrase and its category."""

    lower = text.strip().lower()
    if not lower:
        return None, None

    matches: list[tuple[int, str, Category]] = []
    for category, cues in OBJECT_CUES.items():
        for cue in cues:
            if _contains(lower, cue):
                matches.append((len(cue), cue, category))
    if not matches:
        return None, None

    _, cue, category = max(matches, key=lambda item: item[0])
    return cue, category
