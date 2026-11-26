from __future__ import annotations

import pytest

from contracts.models import Category
from services.harness.category_lexicon import match_category, match_object


CASES = [
    ("Min, jalan di Jl. Kopo rusak parah, ada lubang besar.", Category.ROAD),
    ("Banjir dan drainase mampet di Cihampelas.", Category.DRAINAGE_FLOOD),
    ("Sampah menumpuk di TPS Cijambe.", Category.WASTE),
    ("Pipa PDAM bocor, air keruh.", Category.CLEAN_WATER),
    ("Pengurusan kartu keluarga terhambat di kelurahan.", Category.CIVIL_ADMIN),
    ("Puskesmas kekurangan obat, antrean panjang.", Category.HEALTH_SERVICE),
    ("Pungli parkir liar di pasar.", Category.PUBLIC_ORDER),
    ("Angkot ngetem dan lampu lalu lintas mati.", Category.TRANSPORTATION),
    ("Kebakaran rumah dan pohon tumbang.", Category.FIRE_RESCUE),
    ("Bansos belum cair untuk lansia terlantar.", Category.SOCIAL_AFFAIRS),
    ("Atap sekolah rusak dan biaya seragam mahal.", Category.EDUCATION),
    ("Lampu taman mati dan bangku rusak.", Category.PARKS_HOUSING),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_match_category_covers_all_categories(text: str, expected: Category) -> None:
    category, ambiguous = match_category(text)
    assert category == expected
    assert ambiguous is False


def test_match_category_flags_ambiguity() -> None:
    category, ambiguous = match_category("Pungli parkir liar di trotoar yang rusak.")
    assert category in {Category.PUBLIC_ORDER, Category.ROAD}
    assert ambiguous is True


def test_match_category_returns_none_for_unknown_text() -> None:
    assert match_category("Tolong tindak lanjuti laporan warga.") == (None, False)


def test_match_object_returns_longest_object_and_category() -> None:
    assert match_object("Ada lubang besar di jalan") == ("lubang", Category.ROAD)
    assert match_object("Sampah menumpuk di TPS") == ("sampah", Category.WASTE)
    assert match_object("Pipa bocor di depan rumah") == (
        "pipa bocor",
        Category.CLEAN_WATER,
    )
