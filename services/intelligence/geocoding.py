from __future__ import annotations

import json
import logging
import math
from typing import Any
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("geocoding")


class GeocodingResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    formatted_address: str = ""
    street: str = ""
    subdistrict: str = ""  # Kelurahan
    district: str = ""  # Kecamatan
    city: str = "Kota Bandung"
    latitude: float
    longitude: float
    is_fallback: bool = False


# Offline centroids for Bandung subdistricts for resilient offline matching
BANDUNG_CENTROIDS: list[dict[str, Any]] = [
    {"name": "Sumur Bandung", "subdistrict": "Braga", "lat": -6.9175, "lon": 107.6111},
    {"name": "Coblong", "subdistrict": "Dago", "lat": -6.8856, "lon": 107.6144},
    {"name": "Sukajadi", "subdistrict": "Pasteur", "lat": -6.8867, "lon": 107.5922},
    {"name": "Bojongloa Kaler", "subdistrict": "Kopo", "lat": -6.9389, "lon": 107.5889},
    {"name": "Bandung Kidul", "subdistrict": "Kujangsari", "lat": -6.9556, "lon": 107.6333},
    {"name": "Lengkong", "subdistrict": "Malabar", "lat": -6.9306, "lon": 107.6222},
    {"name": "Cicendo", "subdistrict": "Pasirkaliki", "lat": -6.9056, "lon": 107.5944},
    {"name": "Astanaanyar", "subdistrict": "Cibadak", "lat": -6.9300, "lon": 107.6000},
    {"name": "Regol", "subdistrict": "Ciateul", "lat": -6.9333, "lon": 107.6111},
    {"name": "Batununggal", "subdistrict": "Gumuruh", "lat": -6.9389, "lon": 107.6389},
    {"name": "Cibeunying Kaler", "subdistrict": "Cikutra", "lat": -6.8944, "lon": 107.6333},
    {"name": "Cibeunying Kidul", "subdistrict": "Cicadas", "lat": -6.9083, "lon": 107.6444},
    {"name": "Kiaracondong", "subdistrict": "Babakansari", "lat": -6.9278, "lon": 107.6500},
    {"name": "Buahbatu", "subdistrict": "Margasari", "lat": -6.9500, "lon": 107.6556},
    {"name": "Andir", "subdistrict": "Garuda", "lat": -6.9111, "lon": 107.5778},
    {"name": "Antapani", "subdistrict": "Antapani Wetan", "lat": -6.9167, "lon": 107.6611},
    {"name": "Arcamanik", "subdistrict": "Sukamiskin", "lat": -6.9139, "lon": 107.6806},
    {"name": "Ujungberung", "subdistrict": "Pasirwangi", "lat": -6.9111, "lon": 107.7000},
    {"name": "Gedebage", "subdistrict": "Cisaranten Rancasari", "lat": -6.9667, "lon": 107.6833},
    {"name": "Rancasari", "subdistrict": "Manjahlega", "lat": -6.9556, "lon": 107.6722},
]


class ReverseGeocoder:
    """Converts geographic coordinates into actionable administrative addresses in Indonesia."""

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self.timeout_seconds = timeout_seconds

    def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        address_hint: str = "",
    ) -> GeocodingResult:
        """Reverse-geocodes coordinates to a full Indonesian street address with district and city."""
        if address_hint and len(address_hint.strip()) > 10:
            return GeocodingResult(
                formatted_address=address_hint.strip(),
                latitude=latitude,
                longitude=longitude,
                is_fallback=False,
            )

        # 1. Try online reverse-geocoding via OpenStreetMap Nominatim
        try:
            url = (
                f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}"
                f"&lon={longitude}&zoom=18&addressdetails=1"
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "KAWAL-Citizen-Intake/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            addr = data.get("address", {})
            road = addr.get("road") or addr.get("pedestrian") or addr.get("neighbourhood") or ""
            subdistrict = addr.get("suburb") or addr.get("village") or addr.get("quarter") or ""
            district = addr.get("district") or addr.get("city_district") or ""
            city = addr.get("city") or addr.get("municipality") or "Kota Bandung"

            parts = []
            if road:
                parts.append(f"Jl. {road}")
            if subdistrict:
                parts.append(f"Kelurahan {subdistrict}")
            if district:
                parts.append(f"Kecamatan {district}")
            if city:
                parts.append(city)

            formatted = ", ".join(parts) if parts else data.get("display_name", "")
            return GeocodingResult(
                formatted_address=formatted,
                street=road,
                subdistrict=subdistrict,
                district=district,
                city=city,
                latitude=latitude,
                longitude=longitude,
                is_fallback=False,
            )
        except Exception as exc:
            logger.debug("Online reverse geocoding unavailable (%s); using offline centroid matching", exc)

        # 2. Resilient Offline Centroid Fallback
        closest = self._find_closest_centroid(latitude, longitude)
        formatted = (
            f"Jl. Wilayah {closest['subdistrict']}, Kelurahan {closest['subdistrict']}, "
            f"Kecamatan {closest['name']}, Kota Bandung (Koordinat: {latitude:.4f}, {longitude:.4f})"
        )
        return GeocodingResult(
            formatted_address=formatted,
            street=f"Wilayah {closest['subdistrict']}",
            subdistrict=closest["subdistrict"],
            district=closest["name"],
            city="Kota Bandung",
            latitude=latitude,
            longitude=longitude,
            is_fallback=True,
        )

    @staticmethod
    def _find_closest_centroid(lat: float, lon: float) -> dict[str, Any]:
        """Haversine distance lookup for closest Bandung subdistrict."""
        best_cand = BANDUNG_CENTROIDS[0]
        min_dist = float("inf")

        for c in BANDUNG_CENTROIDS:
            dlat = math.radians(c["lat"] - lat)
            dlon = math.radians(c["lon"] - lon)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(lat))
                * math.cos(math.radians(c["lat"]))
                * math.sin(dlon / 2) ** 2
            )
            dist = 2 * math.asin(math.sqrt(a)) * 6371  # km
            if dist < min_dist:
                min_dist = dist
                best_cand = c

        return best_cand
