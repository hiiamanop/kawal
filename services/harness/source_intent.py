from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict

from contracts.harness import SourceIntent


CITIZEN_CUES = (
    "lapor",
    "aduan",
    "keluhan",
    "tolong",
    "mohon",
    "bantu",
    "perbaiki",
    "benerin atuh",
    "diabaikan",
    "pungli merajalela",
    "segera",
)
OPINION_CUES = (
    "menurut saya",
    "seharusnya",
    "saya rasa",
    "gada perubahan",
    "malah bagusan",
    "berasa kayak",
    "kalau menurut",
    "salah assessment",
)
ANNOUNCEMENT_CUES = (
    "jadwal",
    "info gempa",
    "pemadaman",
    "peringatan",
    "pengumuman",
    "imbauan",
    "ditata",
    "penataan trotoar",
)
QUESTION_CUES = (
    "mau nanya",
    "mau tanya",
    "ada yang tau",
    "ada yang tahu",
    "euy kalau",
    "gimana",
    "dimana",
    "kemana",
    "angkot ga ya",
    "parkir motor",
)
NEWS_CUES = (
    "dilansir",
    "kata ",
    "ujar",
    "sebanyak",
    "akibatnya",
    "berdasarkan",
    "dalam kejadian",
    "kapolsek",
    "petugas",
    "pendataan",
    "selengkapnya",
    "warga terdampak",
    "korban jiwa",
    "program",
    "kegiatan",
    "rapat koordinasi",
    "hanguskan",
    "terjang",
    "hantam",
    "roboh",
    "penutupan jalur",
    "memaksa warga",
    "pengendara diuji",
    "antrean kendaraan",
    "tewas terperosok",
    "satpol pp",
    "warga minta pemerintah",
)
PROBLEM_CUES = (
    "rusak",
    "lubang",
    "banjir",
    "sampah",
    "macet",
    "mati",
    "keruh",
    "tumbang",
    "mampet",
    "kebakaran",
)
CACHE_NON_ACTIONABLE_CUES = (
    "dilansir",
    "kata ",
    "ujar",
    "kapolsek",
    "korban jiwa",
    "warga terdampak",
    "menurut saya",
    "seharusnya",
    "saya rasa",
    "gada perubahan",
    "ada yang tahu",
    "ada yang tau",
    "mau nanya",
    "mau tanya",
    "pengumuman",
    "imbauan",
    "pemadaman",
    "jadwal",
    "info gempa",
)


class SourceIntentAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: SourceIntent
    reason_codes: tuple[str, ...] = ()


def _contains_any(text: str, cues: Sequence[str]) -> bool:
    return any(cue in text for cue in cues)


def has_non_actionable_cue(text: str) -> bool:
    """Return True when any non-citizen cue appears, regardless of citizen cues."""

    return _contains_any(text.strip().lower(), CACHE_NON_ACTIONABLE_CUES)


def assess_source_intent(
    text: str,
    author_handle: str = "",
) -> SourceIntentAssessment:
    del author_handle
    lower = text.strip().lower()

    if _contains_any(lower, CITIZEN_CUES):
        return SourceIntentAssessment(
            intent="CITIZEN_COMPLAINT",
            reason_codes=("SOURCE_CITIZEN_REQUEST",),
        )
    if _contains_any(lower, OPINION_CUES):
        return SourceIntentAssessment(
            intent="OPINION",
            reason_codes=("SOURCE_OPINION",),
        )
    if _contains_any(lower, ANNOUNCEMENT_CUES):
        return SourceIntentAssessment(
            intent="ANNOUNCEMENT",
            reason_codes=("SOURCE_ANNOUNCEMENT",),
        )
    if _contains_any(lower, QUESTION_CUES):
        if _contains_any(lower, PROBLEM_CUES):
            return SourceIntentAssessment(
                intent="UNKNOWN",
                reason_codes=("SOURCE_QUESTION_WITH_PROBLEM",),
            )
        return SourceIntentAssessment(
            intent="QUESTION",
            reason_codes=("SOURCE_QUESTION",),
        )
    if _contains_any(lower, NEWS_CUES):
        return SourceIntentAssessment(
            intent="NEWS_REPORT",
            reason_codes=("SOURCE_NEWS_REPORT",),
        )
    if _contains_any(lower, PROBLEM_CUES):
        return SourceIntentAssessment(
            intent="UNKNOWN",
            reason_codes=("SOURCE_PROBLEM_UNCLEAR",),
        )
    return SourceIntentAssessment(
        intent="UNKNOWN",
        reason_codes=("SOURCE_UNKNOWN",),
    )
