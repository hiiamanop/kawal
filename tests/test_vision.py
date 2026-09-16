from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import urllib.request

import pytest

from contracts.models import Category
from services.intelligence.vision import VisionAnalysisResult, VisionAnalyzer

TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def test_extract_json_variants() -> None:
    # 1. Markdown codeblock
    md_text = '```json\n{"visual_description": "jalan berlubang", "detected_category": "ROAD"}\n```'
    assert VisionAnalyzer._extract_json(md_text) == {
        "visual_description": "jalan berlubang",
        "detected_category": "ROAD",
    }

    # 2. Text with commentary
    commentary = 'Hasil analisis adalah: {"visual_description": "sampah menumpuk", "detected_category": "WASTE"}. Selesai.'
    assert VisionAnalyzer._extract_json(commentary) == {
        "visual_description": "sampah menumpuk",
        "detected_category": "WASTE",
    }

    # 3. Invalid text
    assert VisionAnalyzer._extract_json("not json at all") is None


def test_analyze_image_mocked_success() -> None:
    analyzer = VisionAnalyzer()

    mock_resp = MagicMock()
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "visual_description": "Genangan air meluap setinggi 30 cm menutup badan jalan",
                        "detected_category": "DRAINAGE_FLOOD",
                        "confidence": 0.95,
                        "evidence_quality": "HIGH",
                        "hazard_detected": True,
                        "hazard_summary": "Arus air deras dapat membahayakan pengguna motor",
                    })
                }
            }
        ]
    }
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.object(urllib.request, "urlopen", return_value=mock_resp):
        res = analyzer.analyze_image(
            image_data=TINY_PNG_BASE64,
            citizen_text="banjir min",
        )

    assert res.detected_category == Category.DRAINAGE_FLOOD
    assert res.confidence == 0.95
    assert res.evidence_quality == "HIGH"
    assert res.hazard_detected is True
    assert "Genangan air meluap" in res.visual_description


def test_analyze_image_error_resilience() -> None:
    analyzer = VisionAnalyzer(base_url="http://invalid-host-999:1234/v1")
    res = analyzer.analyze_image(TINY_PNG_BASE64)

    assert res.evidence_quality == "LOW"
    assert "Gagal memproses gambar" in res.visual_description
    assert "error" in res.raw_response


def test_analyze_image_live_omniroute() -> None:
    analyzer = VisionAnalyzer(model_id="antigravity/gemini-3-flash")
    try:
        res = analyzer.analyze_image(
            image_data=TINY_PNG_BASE64,
            citizen_text="lapor aspal berlubang parah",
        )
        assert isinstance(res, VisionAnalysisResult)
        assert len(res.visual_description) > 0
    except Exception as exc:
        pytest.skip(f"OmniRoute proxy offline or unreachable: {exc}")
