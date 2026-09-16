from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Literal
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import Category

logger = logging.getLogger("vision-analyzer")


class VisionAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    visual_description: str = Field(default="")
    detected_category: Category | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_quality: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    hazard_detected: bool = False
    hazard_summary: str = ""
    raw_response: dict[str, Any] = Field(default_factory=dict)


class VisionAnalyzer:
    """Multimodal Vision-Language Model analyzer for citizen complaint images.

    Connects via OmniRoute / OpenAI-compatible endpoint using models like
    antigravity/gemini-3-flash or auto/gemini.
    """

    def __init__(
        self,
        model_id: str = "antigravity/gemini-3-flash",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = (base_url or os.getenv("KAWAL_OMNIROUTE_URL", "http://localhost:20128/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OMNIROUTE_API_KEY", "sk-496a5f477e68f0e1-1dd25c-1f86deed")
        self.timeout_seconds = timeout_seconds

    def analyze_image(
        self,
        image_data: str | bytes,
        citizen_text: str = "",
        mime_type: str = "image/jpeg",
    ) -> VisionAnalysisResult:
        """Analyze an image with optional citizen complaint context."""
        # Normalize image to base64 data URL
        if isinstance(image_data, bytes):
            b64 = base64.b64encode(image_data).decode("utf-8")
            data_url = f"data:{mime_type};base64,{b64}"
        elif isinstance(image_data, str):
            if image_data.startswith("data:"):
                data_url = image_data
            else:
                data_url = f"data:{mime_type};base64,{image_data}"
        else:
            raise ValueError("image_data must be raw bytes or base64 string")

        prompt = (
            "Tugas: Anda adalah asisten VLM KAWAL untuk menganalisis foto aduan publik warga kota di Indonesia.\n"
            "Analisis gambar terlampir dan teks pengirim (jika ada).\n"
            "Kategori yang tersedia: ROAD, DRAINAGE_FLOOD, WASTE, CLEAN_WATER, TRANSPORTATION, PUBLIC_ORDER, "
            "FIRE_RESCUE, PARKS_HOUSING, CIVIL_ADMIN, HEALTH_SERVICE, OTHER.\n\n"
            "Berikan output strictly JSON murni (dimulai dengan { dan diakhiri dengan }):\n"
            "{\n"
            '  "visual_description": "Deskripsi singkat dan faktual apa yang terlihat pada foto (1-2 kalimat).",\n'
            '  "detected_category": "NAMA_KATEGORI_TERDEKAT",\n'
            '  "confidence": 0.90,\n'
            '  "evidence_quality": "HIGH" | "MEDIUM" | "LOW",\n'
            '  "hazard_detected": true | false,\n'
            '  "hazard_summary": "penjelasan bahaya darurat jika ada"\n'
            "}"
        )

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"{prompt}\n\nTeks warga: \"{citizen_text.strip()}\"" if citizen_text.strip() else prompt,
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
        ]

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": 1500,
            "temperature": 0.1,
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Vision analysis failed via model %s: %s", self.model_id, exc)
            return VisionAnalysisResult(
                visual_description="Gagal memproses gambar aduan secara otomatis.",
                evidence_quality="LOW",
                raw_response={"error": str(exc)},
            )

        choices = data.get("choices", [])
        if not choices:
            return VisionAnalysisResult(visual_description="Tidak ada respon dari model visual.")

        content = choices[0].get("message", {}).get("content", "").strip()
        parsed = self._extract_json(content)
        if not parsed:
            return VisionAnalysisResult(
                visual_description=content[:200] if content else "Format respon visual tidak terstruktur.",
                raw_response=data,
            )

        cat_val = parsed.get("detected_category", "").upper().strip()
        matched_cat = None
        for cat in Category:
            if cat.value == cat_val:
                matched_cat = cat
                break

        eq = str(parsed.get("evidence_quality", "MEDIUM")).upper().strip()
        if eq not in ("HIGH", "MEDIUM", "LOW"):
            eq = "MEDIUM"

        return VisionAnalysisResult(
            visual_description=str(parsed.get("visual_description", "")),
            detected_category=matched_cat,
            confidence=float(parsed.get("confidence", 0.5)),
            evidence_quality=eq,
            hazard_detected=bool(parsed.get("hazard_detected", False)),
            hazard_summary=str(parsed.get("hazard_summary", "")),
            raw_response=parsed,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Robust JSON extractor that handles markdown wrappers or stray text."""
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

        # Match outer braces
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(text)
        except Exception:
            return None
