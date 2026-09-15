from __future__ import annotations

from pathlib import Path

from scripts.generate_thesis_tables import (
    generate_latex_tables,
    generate_markdown_tables,
    load_stress_report,
    load_thesis_report,
)


def test_generate_thesis_tables_content() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    thesis_file = repo_root / "artifacts" / "thesis_evaluation_report.json"
    stress_file = repo_root / "artifacts" / "stress_benchmark_report.json"

    assert thesis_file.is_file(), "thesis_evaluation_report.json must exist"

    thesis_data = load_thesis_report(thesis_file)
    stress_data = load_stress_report(stress_file)

    md = generate_markdown_tables(thesis_data, stress_data)
    tex = generate_latex_tables(thesis_data, stress_data)

    # Check Markdown sections
    assert "Tabel 1: Perbandingan Performa Sistem" in md
    assert "KAWAL (Proposed Full)" in md
    assert "Always-LLM (Generative)" in md
    assert "Tabel 2: Hasil Pengujian Hipotesis Formal (H1–H5)" in md
    assert "Tabel 3: Pengujian 5 Invarian Ketahanan Sistem" in md
    assert "Zero Duplicate Tickets under Crash/Retry" in md
    assert "Tabel 4: Karakteristik Throughput & Latensi Beban" in md

    # Check LaTeX booktabs markup
    assert "\\begin{table*}" in tex
    assert "\\toprule" in tex
    assert "\\midrule" in tex
    assert "\\bottomrule" in tex
    assert "\\caption{Perbandingan Performa KAWAL" in tex
    assert "\\caption{Hasil Pengujian Hipotesis Formal H1--H5" in tex
