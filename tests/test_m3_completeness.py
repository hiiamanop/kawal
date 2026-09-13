from __future__ import annotations

from services.intelligence.completeness import resolve_location_completeness


def test_resolve_location_completeness_sufficient_address() -> None:
    text = (
        "Lokasinya Jl. Raya Kopo No. 50, RT 41 RW 31, Kelurahan Babakan Asih, "
        "Kecamatan Bojongloa Kaler, Kota Bandung."
    )
    assert resolve_location_completeness(text) == "SUFFICIENT"


def test_resolve_location_completeness_incomplete_address() -> None:
    assert resolve_location_completeness("Lokasinya Kelurahan Suka Asih.") == "INCOMPLETE"


def test_resolve_location_completeness_ambiguous_address() -> None:
    text = "Lokasi dekat Gerbang Tol Moch Toha, acuan titik tepatnya masih belum terinci."
    assert resolve_location_completeness(text) == "AMBIGUOUS"


def test_resolve_location_completeness_ambiguous_internal_wording() -> None:
    text = "Lokasi dekat Pasar Suci, titiknya masih ambigu membingungkan."
    assert resolve_location_completeness(text) == "AMBIGUOUS"
