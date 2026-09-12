from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from contracts.models import (
    CanonicalSpan,
    CanonicalSpanLabel,
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
)
from services.ml.manifest import compute_file_sha256

PROVENANCE_HIFI_SYNTHETIC = "hifi_synthetic"
PROVENANCE_SYNTHETIC_INDEPENDENT = "synthetic_independent"
CATEGORIES = list(Category)
INTENTS = ("COMPLAINT", "INQUIRY", "FEEDBACK")
RISKS = ("LOW", "MEDIUM", "HIGH", "URGENT")
COMPLETENESS = ("SUFFICIENT", "INCOMPLETE", "AMBIGUOUS")

ISSUE_POOLS: dict[str, dict[Category, tuple[str, ...]]] = {
    "train": {
        Category.ROAD: ("aspal ruas utama berlubang ringan", "cat marka lajur mulai pudar", "ubin trotoar mengalami keretakan"),
        Category.DRAINAGE_FLOOD: ("parit lingkungan dipenuhi lumpur", "genangan muncul seusai hujan", "tembok selokan menunjukkan retakan"),
        Category.WASTE: ("tumpukan limbah rumah warga meninggi", "bak sampah kawasan sudah penuh", "jadwal pengambilan sampah berubah"),
        Category.CLEAN_WATER: ("debit leding rumah berkurang", "sambungan pipa mengeluarkan rembesan", "warna air keran terlihat keruh"),
        Category.CIVIL_ADMIN: ("pengajuan surat belum memiliki kabar", "meja layanan menunda penerimaan dokumen", "rincian kartu keluarga masih memakai data lama"),
        Category.HEALTH_SERVICE: ("barisan pasien di ruang periksa memanjang", "persediaan obat umum menipis", "ruang antre puskesmas penuh sesak"),
    },
    "dev": {
        Category.ROAD: ("badan jalan retak pada beberapa bagian", "pembatas jalur sudah sulit terlihat", "jalur pejalan kaki pecah permukaannya"),
        Category.DRAINAGE_FLOOD: ("saluran tepi jalan tertutup pasir", "air hujan tertahan di badan jalan", "dinding kanal lingkungan terkikis"),
        Category.WASTE: ("kantong sampah warga berserakan", "kontainer pembuangan berisi sampai meluber", "ritase pengangkut sampah bergeser"),
        Category.CLEAN_WATER: ("tekanan air jaringan rumah menurun", "pipa layanan mengalami kebocoran kecil", "air dari keran tampak kecokelatan"),
        Category.CIVIL_ADMIN: ("berkas permintaan administrasi masih diproses", "petugas loket menunda penerimaan berkas", "rincian keluarga pada kartu memerlukan pembaruan"),
        Category.HEALTH_SERVICE: ("waktu tunggu konsultasi pasien bertambah", "obat resep dasar tidak tersedia", "kapasitas ruang tunggu klinik terlampaui"),
    },
    "test": {
        Category.ROAD: ("permukaan lintasan kendaraan bergelombang", "tanda pembagi jalan sulit dibaca", "lantai pedestrian terkelupas"),
        Category.DRAINAGE_FLOOD: ("gorong-gorong tertutup material tanah", "air hujan mengisi cekungan jalan", "dinding parit mengalami abrasi"),
        Category.WASTE: ("limbah domestik menggunung di sudut kampung", "bak penampung melampaui kapasitas muatan", "truk kebersihan datang bergantian"),
        Category.CLEAN_WATER: ("pasokan air pipa mengalir sangat kecil", "jaringan distribusi menetes pada sambungan", "air minum rumah tampak keruh"),
        Category.CIVIL_ADMIN: ("permintaan dokumen masih menunggu keputusan", "loket kantor menghentikan layanan berkas", "susunan data keluarga memerlukan perbaikan"),
        Category.HEALTH_SERVICE: ("jumlah pasien membuat antrean memadat", "stok farmasi untuk obat ringan kosong", "ruang tunggu fasilitas kesehatan sesak"),
    },
    "ood": {
        Category.ROAD: ("lapisan jalan penghubung tergerus", "simbol lalu lintas di ruas itu memudar", "paving jalur kaki terangkat"),
        Category.DRAINAGE_FLOOD: ("aliran parit desa tersendat oleh sedimen", "air limpasan berkumpul di sekitar akses", "bibir saluran air retak memanjang"),
        Category.WASTE: ("buangan rumah tangga menumpuk dekat akses", "tempat buang sementara melampaui daya tampung", "jadwal armada angkut bergeser tanpa kepastian"),
        Category.CLEAN_WATER: ("arus sumber air permukiman melemah", "pipa penyalur basah di titik sambungan", "air keran memiliki warna kekuningan"),
        Category.CIVIL_ADMIN: ("permohonan keterangan warga masih menunggu jawaban", "gerai administrasi menunda berkas sementara", "rincian kartu keluarga masih memakai data lama"),
        Category.HEALTH_SERVICE: ("rombongan pasien memenuhi antrean pemeriksaan", "obat pertolongan dasar habis di fasilitas", "ruang layanan kesehatan terasa sempit"),
    },
}

LOCATION_POOLS = {
    "train": (("Jalan Anggrek", "RT 01 RW 02", "Kelurahan Cibadak", "Kota Bandung"), ("Jalan Kenanga", "RT 03 RW 04", "Kelurahan Wonokromo", "Kota Surabaya"), ("Jalan Teratai", "RT 05 RW 06", "Kelurahan Simpang", "Kota Semarang"), ("Jalan Melati", "RT 07 RW 08", "Kelurahan Sukajadi", "Kota Medan"), ("Jalan Kamboja", "RT 09 RW 10", "Kelurahan Babakan", "Kota Palembang"), ("Jalan Anyelir", "RT 11 RW 12", "Kelurahan Margasari", "Kabupaten Bogor")),
    "dev": (("Jalan Soka", "RT 13 RW 14", "Kelurahan Cempaka", "Kota Tasikmalaya"), ("Jalan Pinus", "RT 15 RW 16", "Kelurahan Purnama", "Kota Malang"), ("Jalan Flamboyan", "RT 17 RW 18", "Kelurahan Mulyasari", "Kota Depok"), ("Jalan Dahlia", "RT 19 RW 20", "Kelurahan Harapan", "Kota Bekasi"), ("Jalan Nusaindah", "RT 21 RW 22", "Kelurahan Sukamaju", "Kota Bogor"), ("Jalan Cendana", "RT 23 RW 24", "Kelurahan Mekarjaya", "Kota Cimahi")),
    "test": (("Jalan Cemara", "RT 25 RW 26", "Kelurahan Sinarjati", "Kota Sukabumi"), ("Jalan Puspa", "RT 27 RW 28", "Kelurahan Wanasari", "Kota Banjar"), ("Jalan Karet", "RT 29 RW 30", "Kelurahan Giriharja", "Kabupaten Cianjur"), ("Jalan Wijaya", "RT 31 RW 32", "Kelurahan Sindang", "Kabupaten Majalengka"), ("Jalan Beringin", "RT 33 RW 34", "Kelurahan Cisayong", "Kabupaten Tasikmalaya"), ("Jalan Merbau", "RT 35 RW 36", "Kelurahan Sukarasa", "Kabupaten Kuningan")),
}
OOD_LOCATIONS = (
    ("Akses Bukit Soka", "Blok 2", "Dusun Pagersari", "Kabupaten Garut"),
    ("Koridor Pasar Ikan", "Segmen Utara", "Kampung Muara", "Kota Cirebon"),
    ("Lorong Lontar", "Gang 7", "Desa Karangmulya", "Kabupaten Subang"),
    ("Jalur Kebun Teh", "Petak 14", "Dusun Mekarsari", "Kabupaten Sukabumi"),
    ("Pelantar Sungai", "Pintu Air 3", "Kampung Tanjung", "Kota Pontianak"),
    ("Kompleks Bukit Kapur", "Sektor C", "Desa Sumberrejo", "Kabupaten Gresik"),
)
TIME_POOLS: dict[str, tuple[str, ...]] = {
    "train": (
        "sejak kemarin pagi", "dalam tiga hari belakangan", "pada hari rabu lalu",
        "sejak tadi malam", "sudah beberapa hari terakhir", "mulai pekan lalu",
    ),
    "dev": (
        "sejak seminggu sebelumnya", "selama empat hari terakhir", "pada waktu fajar",
        "mulai bulan kemarin", "dari sabtu petang", "hingga awal pekan ini",
    ),
    "test": (
        "sejak senin sore", "selama lima hari berjalan", "pada jumat petang",
        "mulai selasa subuh", "dari minggu siang", "hingga rabu petang",
    ),
    "ood": (
        "sejak akhir pekan kemarin", "kurun waktu sepekan", "pada selasa siang",
        "sejak permulaan bulan ini", "mulai pertengahan kuartal", "dari penghujung musim hujan",
    ),
}

CONNECTORS: dict[str, tuple[str, ...]] = {
    "train": ("terjadi pada area", "posisi di", "waktu insiden"),
    "dev": ("terpantau di kawasan", "letak pada", "rentang waktu"),
    "test": ("ditemukan di lokasi", "bertempat di", "saat kejadian"),
    "ood": ("tercatat di zona", "titik koordinat di", "periode peristiwa"),
}

AMBIGUOUS_LOCATION_PREFIXES = {
    "train": "seputar area",
    "dev": "radius sekeliling",
    "test": "lingkungan dekat",
    "ood": "zona berdekatan",
}
SPLIT_DATA = {
    "train": {
        "opening": "Kepada dinas terkait",
        "intents": {"COMPLAINT": "Laporan ini disampaikan terkait", "INQUIRY": "Pertanyaan warga perihal", "FEEDBACK": "Usulan perbaikan mengenai"},
        "risks": {"LOW": "berdampak ringan bagi warga", "MEDIUM": "cukup mengganggu aktivitas", "HIGH": "berisiko membahayakan keselamatan", "URGENT": "memerlukan penanganan segera"},
        "completeness": {"INCOMPLETE": "rincian nomor bangunan dan blok belum disertakan", "AMBIGUOUS": "area sekitar patokan kurang spesifik"},
    },
    "dev": {
        "opening": "Mohon atensi instansi",
        "intents": {"COMPLAINT": "Keluhan warga menyangkut", "INQUIRY": "Permintaan keterangan mengenai", "FEEDBACK": "Catatan evaluasi seputar"},
        "risks": {"LOW": "tingkat dampak minimal", "MEDIUM": "berpotensi menghambat mobilitas", "HIGH": "dapat mengakibatkan kecelakaan", "URGENT": "kondisi darurat mendesak"},
        "completeness": {"INCOMPLETE": "keterangan patokan jalan tidak tertera", "AMBIGUOUS": "titik lokasi membingungkan pemantau"},
    },
    "test": {
        "opening": "Kepada pihak berwenang",
        "intents": {"COMPLAINT": "Aduan masyarakat mengenai", "INQUIRY": "Konfirmasi informasi perihal", "FEEDBACK": "Saran penyempurnaan terkait"},
        "risks": {"LOW": "efek gangguan tergolong kecil", "MEDIUM": "menyulitkan kegiatan harian", "HIGH": "mengancam keamanan pengguna", "URGENT": "situasi gawat darurat"},
        "completeness": {"INCOMPLETE": "informasi penunjuk arah belum lengkap", "AMBIGUOUS": "posisi persis tidak dapat dipastikan"},
    },
    "ood": {
        "opening": "Yth tim lapangan",
        "intents": {"COMPLAINT": "Pemberitahuan masyarakat tentang", "INQUIRY": "Permohonan kejelasan seputar", "FEEDBACK": "Pandangan konstruktif atas"},
        "risks": {"LOW": "konsekuensi relatif rendah", "MEDIUM": "mengakibatkan kendala operasional", "HIGH": "menimbulkan ancaman fatal", "URGENT": "tindakan kilat wajib"},
        "completeness": {"INCOMPLETE": "detail denah tempat belum terdata", "AMBIGUOUS": "gambaran wilayah masih rancu"},
    },
}


def _choose(values: tuple[str, ...], seed: int) -> str:
    return values[seed % len(values)]


def _find_spans(text: str, values: list[tuple[str, CanonicalSpanLabel]]) -> tuple[CanonicalSpan, ...]:
    found: list[CanonicalSpan] = []
    occupied: list[tuple[int, int]] = []
    for value, label in sorted(values, key=lambda item: len(item[0]), reverse=True):
        start = 0
        while value:
            pos = text.find(value, start)
            if pos < 0:
                break
            end = pos + len(value)
            if not any(pos < right and end > left for left, right in occupied):
                found.append(CanonicalSpan(start=pos, end=end, label=label))
                occupied.append((pos, end))
            start = end
    return tuple(sorted(found, key=lambda span: (span.start, span.end)))


def _location(completeness: str, parts: tuple[str, ...], split_name: str) -> tuple[str, list[str], tuple[str, ...]]:
    street, rt, kel, city = parts
    framing = SPLIT_DATA[split_name]["completeness"]
    if completeness == "SUFFICIENT":
        return f"{street}, {rt}, {kel}, {city}", [street, rt, kel, city], ()
    if completeness == "INCOMPLETE":
        return f"{kel}, {city}; {framing['INCOMPLETE']}", [kel, city], ("location_detail",)
    prefix = AMBIGUOUS_LOCATION_PREFIXES[split_name]
    return f"{prefix} {kel} atau {city}; {framing['AMBIGUOUS']}", [kel, city], ("ambiguous_context",)


def _semantic_text(intent: str, risk: str, split_name: str) -> tuple[str, str]:
    data = SPLIT_DATA[split_name]
    return data["intents"][intent], data["risks"][risk]



def _build(
    scenario_id: str, family_id: str, split: DatasetSplit, category: Category, intent: str,
    risk: str, completeness: str, variant: int, family_index: int, independent: bool = False,
) -> tuple[ComplaintTrajectory, dict[str, Any]]:
    locations = OOD_LOCATIONS if independent else LOCATION_POOLS[split.value]
    issue = _choose(ISSUE_POOLS["ood" if independent else split.value][category], family_index * 7 + variant * 3 + (19 if independent else 0))
    parts = locations[family_index % len(locations)]
    split_name = "ood" if independent else split.value
    loc, span_locations, missing = _location(completeness, parts, split_name)
    time_text = _choose(TIME_POOLS[split_name], family_index * 5 + variant * 2)
    intent_text, risk_text = _semantic_text(intent, risk, split_name)
    opening = SPLIT_DATA[split_name]["opening"]
    style = opening
    connector = _choose(CONNECTORS[split_name], family_index + variant)
    if variant % 4 == 0:
        text = f"{opening}: {intent_text} {issue} {connector} {loc}, {time_text}; {risk_text}."
    elif variant % 4 == 1:
        text = f"{opening}: {risk_text}; {intent_text} {issue} {connector} {loc} {time_text}."
    elif variant % 4 == 2:
        text = f"{opening}: {intent_text} {issue} {connector} {loc}, {time_text}; {risk_text}."
    else:
        text = f"{opening}: {risk_text} {connector} {time_text}; {intent_text} {issue} {loc}."
    spans = _find_spans(text, [(issue, CanonicalSpanLabel.OBJ), *[(x, CanonicalSpanLabel.LOC) for x in span_locations], (time_text, CanonicalSpanLabel.TIME)])
    message_id = f"msg_{'ood' if independent else split.value}_{category.value.lower()}_{family_index:02d}_{variant:02d}"
    bubble = TrajectoryBubble(source_message_id=message_id, text=text, offset_seconds=0, canonical_spans=spans)
    action = DecisionMode.EXECUTE if completeness == "SUFFICIENT" else DecisionMode.REQUEST_CLARIFICATION if completeness == "INCOMPLETE" else DecisionMode.RE_EVALUATE
    expected = TurnExpectedAction(turn=1, allowed_actions=(action,), missing=missing, strategy=action.value)
    turn = TrajectoryTurn(turn=1, bubbles=(bubble,), observable_facts=(issue, *span_locations, time_text), hidden_facts=(), expected_action=expected)
    provenance = PROVENANCE_SYNTHETIC_INDEPENDENT if independent else PROVENANCE_HIFI_SYNTHETIC
    truth = {
        "intent": intent, "risk": risk, "completeness": completeness, "category": category.value,
        "domain": category.value, "provenance": provenance, "dataset_origin": provenance,
        "is_synthetic": True, "human_written": False, "real_world": False,
        "canonical_spans": [list(span) for span in spans],
        "bubble_canonical_spans": {message_id: [list(span) for span in spans]},
        "discourse_structure": f"semantic_{(family_index + variant) % 6}",
        "discourse_genre": ("field_survey_log", "citizen_consultation", "community_narrative", "administrative_petition", "alert_dispatch")[variant % 5],
        "internal_framing": True,
    }
    trajectory = ComplaintTrajectory(
        scenario_id=scenario_id, family_id=family_id, split=split, category=category,
        world_truth=truth, observable_facts=(issue, *span_locations, time_text), hidden_facts=(),
        location_completeness=completeness, duration="OBSERVED", claim_certainty="FIRST_HAND",
        persona="STANDARD", noise={"style": style}, attachment_role="NONE", turns=(turn,),
        expected_action_by_turn=(expected,), canonical_spans=spans,
    )
    raw = trajectory.model_dump(mode="json")
    return trajectory, raw


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records), encoding="utf-8")


def generate_hifi_dataset(output_dir: Path | str, seed: int = 5101, families_per_category: int = 12, variants_per_family: int = 12, **_: Any) -> dict[str, Any]:
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_counts = ((8, 2, 2), (8, 2, 2), (8, 2, 2), (8, 2, 2), (9, 2, 1), (9, 1, 2))
    buckets: dict[str, list[ComplaintTrajectory]] = {"train": [], "dev": [], "test": []}
    raw: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for ci, category in enumerate(CATEGORIES):
        ntrain, ndev, _ = split_counts[ci]
        for fi in range(families_per_category):
            split = "train" if fi < ntrain else "dev" if fi < ntrain + ndev else "test"
            enum = DatasetSplit(split)
            for variant in range(variants_per_family):
                labels = ((fi + variant + ci) % 3, (fi * 3 + variant + ci) % 4, (fi * 5 + variant + ci) % 3)
                trajectory, record = _build(f"hifi_{split}_{ci:02d}_{fi:02d}_{variant:02d}", f"hifi-family-{ci:02d}-{fi:02d}", enum, category, INTENTS[labels[0]], RISKS[labels[1]], COMPLETENESS[labels[2]], variant, fi)
                buckets[split].append(trajectory)
                raw[split].append(record)
    ood: list[ComplaintTrajectory] = []
    ood_raw: list[dict[str, Any]] = []
    for i in range(30):
        for rep in range(12):
            ci = (i + rep * 2) % len(CATEGORIES)
            trajectory, record = _build(f"ood_{ci:02d}_{i:02d}_{rep:02d}", f"ood-family-{ci:02d}-{i:02d}", DatasetSplit.TEST, CATEGORIES[ci], INTENTS[(i + rep) % 3], RISKS[(i * 2 + rep) % 4], COMPLETENESS[(i * 3 + rep) % 3], rep, i, True)
            ood.append(trajectory)
            ood_raw.append(record)
    all_records = buckets["train"] + buckets["dev"] + buckets["test"]
    for name, values in buckets.items():
        _write_jsonl(out / f"{name}.jsonl", [x.model_dump(mode="json") for x in values])
    _write_jsonl(out / "trajectories.jsonl", [x.model_dump(mode="json") for x in all_records])
    _write_jsonl(out / "ood_canary.jsonl", [x.model_dump(mode="json") for x in ood])
    corpus = "\n".join(b.text for t in buckets["train"] for turn in t.turns for b in turn.bubbles) + "\n"
    (out / "corpus.txt").write_text(corpus, encoding="utf-8")
    (out / "corpus_train.txt").write_text(corpus, encoding="utf-8")
    files = ["train.jsonl", "dev.jsonl", "test.jsonl", "trajectories.jsonl", "ood_canary.jsonl", "corpus.txt", "corpus_train.txt"]
    checksums = "".join(f"{compute_file_sha256(out / name)}  {name}\n" for name in files)
    (out / "sha256sums.txt").write_text(checksums, encoding="utf-8")
    ood_sha256 = compute_file_sha256(out / "ood_canary.jsonl")
    hifi_manifest = {
        "manifest_version": "1.0.0",
        "human_written": False,
        "is_human_written": False,
        "real_world": False,
        "is_real_world": False,
        "provenance": PROVENANCE_HIFI_SYNTHETIC,
        "artifacts": {
            "ood_canary": {
                "name": "ood_canary",
                "path": "ood_canary.jsonl",
                "sha256": ood_sha256,
                "task": "holdout",
                "split": "test",
                "expected_count": len(ood),
                "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            }
        },
        "datasets": {
            "ood_canary": {
                "name": "ood_canary",
                "path": "ood_canary.jsonl",
                "sha256": ood_sha256,
                "task": "holdout",
                "split": "test",
                "expected_count": len(ood),
                "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            }
        },
    }
    (out / "hifi_manifest.json").write_text(json.dumps(hifi_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "manifest_version": "2.0.0", "environment": "local-cpu",
        "splits": {"train": {"path": "train.jsonl", "count": len(buckets["train"])}, "dev": {"path": "dev.jsonl", "count": len(buckets["dev"])}, "test": {"path": "test.jsonl", "count": len(buckets["test"]) }},
        "artifacts": {name: {"name": name.rsplit('.', 1)[0], "version": "2.0.0", "path": name, "sha256": compute_file_sha256(out / name), "size_bytes": (out / name).stat().st_size, "task": "general", "precision": "fp32"} for name in files},
        "ood": {"path": "ood_canary.jsonl", "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT, "withheld_from_train": True, "expected_count": len(ood)},
    }
    for item in manifest["artifacts"].values():
        item.pop("expected_count", None)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"output_dir": out, "train": buckets["train"], "dev": buckets["dev"], "test": buckets["test"], "ood": ood, "all": all_records, "raw": {**raw, "ood": ood_raw}, "splits": {"train": {"count": len(buckets["train"])}, "dev": {"count": len(buckets["dev"])}, "test": {"count": len(buckets["test"])}}}


generate_m3_hifi = generate_hifi_dataset
generate_and_export_m3_hifi = generate_hifi_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate M3 HiFi dataset.")
    parser.add_argument("--output-dir", default="artifacts/m3_hifi")
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--families-per-category", type=int, default=12)
    parser.add_argument("--variants-per-family", type=int, default=12)
    args = parser.parse_args(argv)
    generate_hifi_dataset(args.output_dir, args.seed, args.families_per_category, args.variants_per_family)
    return 0


if __name__ == "__main__":
    sys.exit(main())
