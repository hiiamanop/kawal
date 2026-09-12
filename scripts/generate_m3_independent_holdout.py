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
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
)
from services.dataset.splits import (
    _extract_ngrams,
    _is_common_language_ngram,
)

PROVENANCE_SYNTHETIC_INDEPENDENT = "synthetic_independent"

CATEGORIES: list[Category] = [
    Category.ROAD, Category.DRAINAGE_FLOOD, Category.WASTE,
    Category.CLEAN_WATER, Category.CIVIL_ADMIN, Category.HEALTH_SERVICE,
]
INTENTS = ["COMPLAINT", "INQUIRY", "FEEDBACK"]
RISKS = ["LOW", "MEDIUM", "HIGH", "URGENT"]
COMPLETENESS_LEVELS = ["SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"]

INDEPENDENT_OBJECTS: dict[Category, list[str]] = {
    Category.ROAD: [
        "aspal ambles cukup dalam", "trotoar patah membahayakan pejalan kaki",
        "lampu jalan mati total", "jalan berlubang besar yang mengganggu arus",
        "bahu jalan amblas parah di tikungan tajam", "pembatas jalan roboh miring ke jalur",
    ],
    Category.DRAINAGE_FLOOD: [
        "gorong-gorong air tersumbat endapan lumpur pekat",
        "konstruksi gorong-gorong ambrol menyumbat aliran",
        "titik tanggul penahan air jebol tergerus arus deras",
        "genangan air cukup tinggi menutup akses permukiman",
        "parit saluran air meluap kotoran limbah",
        "dinding saluran drainase runtuh menimbun got",
    ],
    Category.WASTE: [
        "buangan sampah kantong plastik mencemari lapangan terbuka",
        "kondisi sampah menumpuk mengeluarkan bau busuk menyengat",
        "pembuangan limbah beracun tanpa izin di lahan terbuka",
        "timbunan sampah plastik basah menutupi badan jalan",
        "air limbah domestik meluber dari tempat pembuangan liar",
        "bak sampah rusak parah hingga sampah berceceran",
    ],
    Category.CLEAN_WATER: [
        "kejadian pipa bocor bertekanan tinggi menyembur",
        "aliran air keran mati total sudah berhari-hari",
        "pipa saluran air minum retak dan rembes deras",
        "suplai air keran padam mendadak tanpa ada pengumuman",
        "meteran pipa air warga patah mengeluarkan luapan",
        "mutu air leding warga keruh kecokelatan dan berpasir",
    ],
    Category.CIVIL_ADMIN: [
        "layanan perekaman ktp elektronik mengalami gangguan sistem antrean",
        "keterlambatan penerbitan kartu keluarga baru warga pindahan",
        "kendala verifikasi administrasi akta kelahiran anak",
        "masalah pencairan kuota bansos sembako belum tersalurkan",
        "proses legalisir dokumen ktp tersendat di loket pelayanan",
        "permohonan perbaikan kartu keluarga terhambat validasi data",
    ],
    Category.HEALTH_SERVICE: [
        "pelayanan rawat inap puskesmas kekurangan tenaga perawat",
        "antrean dokter umum poli puskesmas menumpuk sejak pagi",
        "ketersediaan stok obat generik di apotek puskesmas kosong",
        "fasilitas ruang ugd puskesmas minim perlengkapan darurat",
        "jadwal pemeriksaan dokter jaga puskesmas sering tertunda",
        "layanan rujukan medis puskesmas ke rumah sakit dipersulit",
    ],
}

STREETS = [
    "Jalan Mahoni Rindang", "Jalan Meranti Teduh", "Jalan Gaharu Sentosa",
    "Jalan Ebony Harmoni", "Jalan Trembesi Lestari", "Jalan Damar Sejuk",
    "Jalan Rasamala Wibawa", "Jalan Akasia Teduh", "Jalan Kemuning Rindang",
    "Jalan Cendana Sentosa",
]
RT_RW = ["RT 01 RW 02", "RT 03 RW 04", "RT 05 RW 06", "RT 07 RW 08", "RT 09 RW 10"]
KELURAHANS = [
    "Kelurahan Sukadamai", "Kelurahan Harapanjaya", "Kelurahan Sambiroto",
    "Kelurahan Tegalmulyo", "Kelurahan Sidomukti", "Kelurahan Sumberagung",
    "Kelurahan Candisari", "Kelurahan Kertomulyo", "Kelurahan Sindumartani",
    "Kelurahan Widyasari",
]
KECAMATANS = [
    "Kecamatan Argomulyo", "Kecamatan Tingkir", "Kecamatan Sidorejo",
    "Kecamatan Banyudono", "Kecamatan Sawit", "Kecamatan Teras",
    "Kecamatan Mojosongo", "Kecamatan Delanggu", "Kecamatan Ceper",
    "Kecamatan Bayat",
]
CITIES = [
    "Kota Salatiga", "Kota Magelang", "Kota Bukittinggi", "Kota Singkawang",
    "Kota Palopo", "Kabupaten Purworejo", "Kabupaten Boyolali",
    "Kabupaten Wonosobo", "Kabupaten Sragen", "Kabupaten Klaten",
]
LANDMARKS = [
    "dekat gardu pos ronda", "seberang balai warga", "samping gapura masuk",
    "depan lapangan olahraga", "dekat ruko perbelanjaan", "seberang pos keamanan",
    "samping taman bermain", "depan gerbang perumahan",
]
TIMES = [
    "sejak kemarin pagi", "pagi ini sekitar jam 08:00", "saat ini kondisinya mendesak",
    "sudah 3 hari ini", "kemarin sore sekitar jam 16:00", "siang ini pada pukul 13:00",
    "tadi pagi sekitar jam 07:00", "hari ini dari pagi", "sejak 2 hari yang lalu",
]
GREETINGS = [
    "Selamat pagi", "Selamat siang", "Selamat sore",
    "Halo admin", "Mohon perhatian", "Lapor petugas",
]
INTENT_TEMPLATES: dict[str, list[str]] = {
    "COMPLAINT": [
        "mohon segera diperbaiki",
        "kami melaporkan keluhan terkait",
        "tolong segera ditindaklanjuti persoalan",
        "warga sangat terganggu oleh",
        "kami menyampaikan keluhan warga perihal",
    ],
    "INQUIRY": [
        "mohon informasi jadwal penanganan untuk",
        "ingin menanyakan kepastian perbaikan",
        "apakah ada tindak lanjut mengenai",
        "bagaimana prosedur pelaporan atas masalah",
        "kapan estimasi tim teknis meninjau",
    ],
    "FEEDBACK": [
        "memberikan evaluasi dan catatan penting mengenai",
        "saran pemeliharaan fasilitas terkait",
        "umpan balik penataan infrastruktur untuk",
        "masukan masyarakat perihal perbaikan",
        "catatan pengawasan lingkungan terkait",
    ],
}

INDEPENDENT_RISK_EVIDENCE: dict[str, list[str]] = {
    "LOW": [
        "kondisi saat ini masih ringan, belum parah dan tidak ada bahaya mendesak",
        "keadaan di lokasi masih tergolong aman, belum parah dan tidak darurat",
        "situasi saat ini masih berstatus aman terkendali, belum parah dan tidak membahayakan",
        "laporan ini bersifat pencegahan awal, dampaknya belum parah dan tidak mendesak",
    ],
    "MEDIUM": [
        "kondisi makin rusak bertahap dan mengganggu kenyamanan tetapi belum darurat",
        "keadaan semakin rusak dan membutuhkan jadwal perbaikan berkala kendati belum sangat parah",
        "situasi di lapangan semakin rusak bertahap walau saat ini belum darurat",
        "masalah infrastruktur ini makin rusak berulang dan perlu perawatan terencana namun belum darurat",
    ],
    "HIGH": [
        "kondisi sangat parah dan bahaya sehingga mendesak untuk segera ditangani",
        "keadaan sangat parah dengan potensi ancaman bahaya tinggi yang butuh penanganan mendesak",
        "situasi sangat parah meluas hingga menimbulkan bahaya nyata yang mendesak diselesaikan",
        "kerusakan sangat parah dan rawan kecelakaan bahaya sehingga mendesak ditindaklanjuti segera",
    ],
    "URGENT": [
        "keadaan darurat sangat mendesak dan bahaya sekali tolong segera kirim bantuan sekarang",
        "situasi sangat darurat dan bahaya sekali butuh tindakan penanganan darurat sekarang juga",
        "kondisi bencana sangat darurat ancaman bahaya sekali tolong datang ke lokasi sekarang",
        "keadaan darurat parah bahaya sekali tolong tindakan cepat di lokasi sekarang juga",
    ],
}

INCOMPLETE_LOCATION_TEMPLATES = [
    "Patokan lokasi hanya berpatokan pada {landmark} saja, belum ada nama jalan maupun wilayah administrasi kelurahan dan kecamatan. Mohon bantuan tindak lanjutnya, terima kasih.",
    "Titik lokasi hanya berupa patokan {landmark} saja tanpa alamat jalan serta keterangan wilayah kelurahan atau kecamatan. Mohon bantuan tindak lanjutnya, terima kasih.",
    "Informasi posisi hanya berpatokan pada {landmark} saja, belum tercantum rincian jalan dan wilayah administrasi kelurahan maupun kecamatan. Mohon bantuan tindak lanjutnya, terima kasih.",
    "Keterangan lokasi hanya patokan {landmark} saja, tanpa nama jalan maupun wilayah administrasi kelurahan dan kecamatan. Mohon bantuan tindak lanjutnya, terima kasih.",
]

AMBIGUOUS_LOCATION_TEMPLATES = [
    "Titik patokan lokasi tidak pasti dan simpang siur antara {landmark_a} atau di seberang {landmark_b}, sehingga butuh klarifikasi petugas untuk memastikan posisi persisnya. Mohon bantuannya, terima kasih.",
    "Posisi patokan lokasi masih membingungkan dan belum pasti antara {landmark_a} atau samping {landmark_b}, perlu klarifikasi lebih lanjut untuk konfirmasi titik yang benar. Mohon bantuannya, terima kasih.",
    "Keterangan patokan lokasi simpang siur dan saling bertentangan antara area depan {landmark_a} atau seberang {landmark_b}, mohon klarifikasi dari pihak berwenang guna kepastian titik lokasi. Terima kasih.",
    "Informasi titik lokasi masih belum pasti antara berada di {landmark_a} atau justru dekat {landmark_b}, sangat memerlukan klarifikasi petugas untuk verifikasi lokasi sebenarnya. Terima kasih.",
]


def compute_file_sha256(path: Path | str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def audit_source_contamination(trajectories: list[ComplaintTrajectory], source_dir: Path) -> None:
    if not source_dir.is_dir():
        return
    source_files = [
        source_dir / "trajectories.jsonl", source_dir / "train.jsonl",
        source_dir / "dev.jsonl", source_dir / "test.jsonl",
    ]
    source_files = [f for f in source_files if f.is_file()] or list(source_dir.glob("*.jsonl"))
    if not source_files:
        return

    source_scenarios: set[str] = set()
    source_families: set[str] = set()
    source_normalized_texts: set[str] = set()
    source_ngrams: set[tuple[str, ...]] = set()

    for sf in source_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                sid = data.get("scenario_id")
                if sid:
                    source_scenarios.add(str(sid))
                fid = data.get("family_id")
                if fid:
                    source_families.add(str(fid))
                for turn in data.get("turns", []):
                    if isinstance(turn, dict):
                        for b in turn.get("bubbles", []):
                            if isinstance(b, dict) and "text" in b:
                                text_raw = str(b["text"])
                                norm = re.sub(r"\s+", " ", text_raw.strip().lower())
                                if len(norm) > 15:
                                    source_normalized_texts.add(norm)
                                tokens = re.findall(r"[a-zA-Z0-9_\-]+", text_raw.lower())
                                for size in (3, 4, 5):
                                    for ng in _extract_ngrams(tokens, size):
                                        if not _is_common_language_ngram(ng):
                                            source_ngrams.add(ng)

    holdout_scenarios = {t.scenario_id for t in trajectories}
    holdout_families = {t.family_id for t in trajectories}
    if holdout_scenarios & source_scenarios:
        raise ValueError(f"Scenario ID contamination: {holdout_scenarios & source_scenarios}")
    if holdout_families & source_families:
        raise ValueError(f"Family ID contamination: {holdout_families & source_families}")

    holdout_normalized_texts = {
        re.sub(r"\s+", " ", b.text.strip().lower())
        for t in trajectories for turn in t.turns for b in turn.bubbles
        if len(b.text.strip()) > 15
    }
    overlap = holdout_normalized_texts & source_normalized_texts
    if overlap:
        raise ValueError(f"Normalized text contamination: {list(overlap)[:3]}")

    for t in trajectories:
        for turn in t.turns:
            for b in turn.bubbles:
                b_lower = b.text.lower()
                for sid in list(source_scenarios)[:100]:
                    if sid.lower() in b_lower:
                        raise ValueError(f"Source scenario_id '{sid}' leaked into holdout bubble text")

    holdout_ngrams: set[tuple[str, ...]] = set()
    for t in trajectories:
        for turn in t.turns:
            for b in turn.bubbles:
                tokens = re.findall(r"[a-zA-Z0-9_\-]+", b.text.lower())
                for size in (3, 4, 5):
                    for ng in _extract_ngrams(tokens, size):
                        if not _is_common_language_ngram(ng):
                            holdout_ngrams.add(ng)

    ngram_overlap = holdout_ngrams & source_ngrams
    if ngram_overlap:
        raise ValueError(f"N-gram contamination: {list(ngram_overlap)[:3]}")

    pii_patterns = (
        re.compile(r"\b\d{16}\b"),
        re.compile(r"\b(?:\+62|62|08)\d{8,12}\b"),
        re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE),
    )
    for text in holdout_normalized_texts:
        if any(pattern.search(text) for pattern in pii_patterns):
            raise ValueError("PII detected in independent held-out text")


def build_trajectory(idx: int, rng: random.Random) -> ComplaintTrajectory:
    category = CATEGORIES[idx % len(CATEGORIES)]
    intent = INTENTS[idx % len(INTENTS)]
    risk = RISKS[idx % len(RISKS)]
    completeness = COMPLETENESS_LEVELS[idx % len(COMPLETENESS_LEVELS)]

    obj = INDEPENDENT_OBJECTS[category][(idx // len(CATEGORIES)) % len(INDEPENDENT_OBJECTS[category])]
    street = STREETS[idx % len(STREETS)]
    rtrw = RT_RW[idx % len(RT_RW)]
    kel = KELURAHANS[idx % len(KELURAHANS)]
    kec = KECAMATANS[idx % len(KECAMATANS)]
    city = CITIES[idx % len(CITIES)]
    landmark = LANDMARKS[idx % len(LANDMARKS)]
    time_str = TIMES[idx % len(TIMES)]
    greeting = GREETINGS[idx % len(GREETINGS)]
    action_phrase = INTENT_TEMPLATES[intent][idx % len(INTENT_TEMPLATES[intent])]
    risk_evidence_raw = INDEPENDENT_RISK_EVIDENCE[risk][(idx // len(RISKS)) % len(INDEPENDENT_RISK_EVIDENCE[risk])]
    risk_evidence = risk_evidence_raw[0].upper() + risk_evidence_raw[1:]

    scenario_id = f"indep_holdout_{idx + 1:04d}"
    family_id = f"fam_indep_{category.value.lower()}_{(idx // len(CATEGORIES)) + 1:03d}"

    if completeness == "SUFFICIENT":
        bubble1_text = (
            f"{greeting}, {action_phrase} {obj} di {street}, {rtrw}, {kel}, {kec}, {city}, "
            f"kejadiannya {time_str}. {risk_evidence}."
        )
        bubble2_text = (
            f"Patokan lokasi persis {landmark} {street}. Mohon bantuan tindak lanjutnya, terima kasih."
        )
        turn1_action = TurnExpectedAction(
            turn=1, allowed_actions=(DecisionMode.EXECUTE,), missing=(), strategy="DISPATCH",
        )
        turn1 = TrajectoryTurn(
            turn=1,
            bubbles=(TrajectoryBubble(source_message_id=f"msg_{scenario_id}_t1_b1", text=bubble1_text, offset_seconds=0),),
            observable_facts=(f"Jalan {street}", f"Kelurahan {kel}", f"Kecamatan {kec}"),
            hidden_facts=(),
            expected_action=turn1_action,
        )
        turn2_action = TurnExpectedAction(
            turn=2, allowed_actions=(DecisionMode.EXECUTE,), missing=(), strategy="CONFIRM",
        )
        turn2 = TrajectoryTurn(
            turn=2,
            bubbles=(TrajectoryBubble(source_message_id=f"msg_{scenario_id}_t2_b1", text=bubble2_text, offset_seconds=30),),
            observable_facts=(f"Jalan {street}",),
            hidden_facts=(),
            expected_action=turn2_action,
        )
        observable_facts = (obj, f"Jalan {street}", f"Kelurahan {kel}", f"Kecamatan {kec}", city)
        hidden_facts = ()

    elif completeness == "INCOMPLETE":
        bubble1_text = (
            f"{greeting}, {action_phrase} {obj}, "
            f"kejadiannya {time_str}. {risk_evidence}."
        )
        incomp_tmpl = INCOMPLETE_LOCATION_TEMPLATES[(idx // len(COMPLETENESS_LEVELS)) % len(INCOMPLETE_LOCATION_TEMPLATES)]
        bubble2_text = incomp_tmpl.format(landmark=landmark)

        turn1_action = TurnExpectedAction(
            turn=1, allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,), missing=("jurisdiction",), strategy="REQUEST_LOCATION",
        )
        turn1 = TrajectoryTurn(
            turn=1,
            bubbles=(TrajectoryBubble(source_message_id=f"msg_{scenario_id}_t1_b1", text=bubble1_text, offset_seconds=0),),
            observable_facts=(obj,),
            hidden_facts=(f"Jalan {street}", f"Kelurahan {kel}", f"Kecamatan {kec}", city),
            expected_action=turn1_action,
        )
        turn2_action = TurnExpectedAction(
            turn=2, allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,), missing=("jurisdiction",), strategy="REQUEST_LOCATION",
        )
        turn2 = TrajectoryTurn(
            turn=2,
            bubbles=(TrajectoryBubble(source_message_id=f"msg_{scenario_id}_t2_b1", text=bubble2_text, offset_seconds=30),),
            observable_facts=(landmark,),
            hidden_facts=(f"Jalan {street}", f"Kelurahan {kel}", f"Kecamatan {kec}", city),
            expected_action=turn2_action,
        )
        observable_facts = (obj, landmark)
        hidden_facts = (f"Jalan {street}", f"Kelurahan {kel}", f"Kecamatan {kec}", city)

    else:  # AMBIGUOUS
        landmark_a = landmark
        landmark_b = LANDMARKS[(idx + 3) % len(LANDMARKS)]
        bubble1_text = (
            f"{greeting}, {action_phrase} {obj} di sekitar {street}, "
            f"kejadiannya {time_str}. {risk_evidence}."
        )
        ambig_tmpl = AMBIGUOUS_LOCATION_TEMPLATES[(idx // len(COMPLETENESS_LEVELS)) % len(AMBIGUOUS_LOCATION_TEMPLATES)]
        bubble2_text = ambig_tmpl.format(landmark_a=landmark_a, landmark_b=landmark_b)

        turn1_action = TurnExpectedAction(
            turn=1, allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,), missing=("location_ambiguity",), strategy="REQUEST_CLARIFICATION",
        )
        turn1 = TrajectoryTurn(
            turn=1,
            bubbles=(TrajectoryBubble(source_message_id=f"msg_{scenario_id}_t1_b1", text=bubble1_text, offset_seconds=0),),
            observable_facts=(f"Jalan {street}",),
            hidden_facts=(),
            expected_action=turn1_action,
        )
        turn2_action = TurnExpectedAction(
            turn=2, allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,), missing=("location_ambiguity",), strategy="REQUEST_CLARIFICATION",
        )
        turn2 = TrajectoryTurn(
            turn=2,
            bubbles=(TrajectoryBubble(source_message_id=f"msg_{scenario_id}_t2_b1", text=bubble2_text, offset_seconds=30),),
            observable_facts=(f"antara {landmark_a} atau {landmark_b}",),
            hidden_facts=(),
            expected_action=turn2_action,
        )
        observable_facts = (obj, f"Jalan {street}", f"antara {landmark_a} atau {landmark_b}")
        hidden_facts = ()

    world_truth = {
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "human_written": False,
        "is_human_written": False,
        "dataset_origin": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "heldout_type": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "synthetic_heldout": True,
        "is_synthetic": True,
        "intent": intent,
        "risk": risk,
        "completeness": completeness,
        "category": category.value,
    }

    return ComplaintTrajectory(
        scenario_id=scenario_id,
        family_id=family_id,
        split=DatasetSplit.TEST,
        category=category,
        world_truth=world_truth,
        observable_facts=observable_facts,
        hidden_facts=hidden_facts,
        location_completeness=completeness,
        duration="UNKNOWN",
        claim_certainty="FIRST_HAND",
        persona="STANDARD",
        noise={},
        attachment_role="NONE",
        turns=(turn1, turn2),
        expected_action_by_turn=(turn1_action, turn2_action),
    )


def generate_independent_holdout(
    output_dir: Path,
    source_dir: Path,
    seed: int = 9001,
    count: int = 300,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    source_path = Path(source_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    trajectories = [build_trajectory(i, rng) for i in range(count)]
    audit_source_contamination(trajectories, source_path)

    holdout_file = output_path / "test_independent_holdout.jsonl"
    with open(holdout_file, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")

    metadata_entry = {
        "_type": "metadata",
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "human_written": False,
        "is_human_written": False,
        "dataset_type": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "generator": "scripts/generate_m3_independent_holdout.py",
        "total_records": count,
        "seed": seed,
        "split": "test",
        "created_at": "2026-09-12T00:00:00+00:00",
        "card": {
            "dataset_name": "m3_independent_holdout",
            "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            "synthetic_heldout": True,
            "categories": {cat.value: count // len(CATEGORIES) for cat in CATEGORIES},
            "intents": {intent: count // len(INTENTS) for intent in INTENTS},
            "risks": {risk: count // len(RISKS) for risk in RISKS},
            "completeness": {comp: count // len(COMPLETENESS_LEVELS) for comp in COMPLETENESS_LEVELS},
        },
    }
    meta_file = output_path / "metadata.jsonl"
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_entry) + "\n")

    holdout_sha = compute_file_sha256(holdout_file)
    holdout_size = holdout_file.stat().st_size
    meta_sha = compute_file_sha256(meta_file)
    meta_size = meta_file.stat().st_size

    manifest_dict = {
        "manifest_version": "1.0.0",
        "environment": "local-cpu",
        "artifacts": {
            "test_independent_holdout": {
                "name": "test_independent_holdout",
                "version": "1.0.0",
                "sha256": holdout_sha,
                "path": "test_independent_holdout.jsonl",
                "size_bytes": holdout_size,
                "task": "holdout",
                "precision": "fp32",
            },
            "metadata": {
                "name": "metadata",
                "version": "1.0.0",
                "sha256": meta_sha,
                "path": "metadata.jsonl",
                "size_bytes": meta_size,
                "task": "metadata",
                "precision": "fp32",
            },
        },
    }
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    manifest_sha = compute_file_sha256(manifest_file)

    sha_lines = [
        f"{holdout_sha}  test_independent_holdout.jsonl",
        f"{meta_sha}  metadata.jsonl",
        f"{manifest_sha}  manifest.json",
    ]
    sha_content = "\n".join(sha_lines) + "\n"

    sha_file = output_path / "sha256sums.txt"
    with open(sha_file, "w", encoding="utf-8") as f:
        f.write(sha_content)

    sha_file_std = output_path / "sha256sums"
    with open(sha_file_std, "w", encoding="utf-8") as f:
        f.write(sha_content)

    return {
        "total_records": count,
        "output_dir": str(output_path),
        "source_dir": str(source_path),
        "seed": seed,
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
        "categories": {cat.value: count // len(CATEGORIES) for cat in CATEGORIES},
        "artifacts": {
            "dataset": "test_independent_holdout.jsonl",
            "metadata": "metadata.jsonl",
            "manifest": "manifest.json",
            "sha256sums": "sha256sums.txt",
        },
    }


generate_m3_independent_holdout = generate_independent_holdout
generate_and_export_m3_independent_holdout = generate_independent_holdout
generate_and_export_independent_holdout = generate_independent_holdout
generate_holdout_dataset = generate_independent_holdout
generate_holdout = generate_independent_holdout
generate_dataset = generate_independent_holdout
run = generate_independent_holdout


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate independent held-out M3 evaluation dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m3_independent_holdout"), help="Directory to write holdout dataset and manifest")
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/m3_synthetic"), help="Source training artifacts directory for anti-contamination audit")
    parser.add_argument("--seed", type=int, default=9001, help="Deterministic pseudo-random seed")
    parser.add_argument("--count", type=int, default=300, help="Number of holdout trajectories to generate")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = generate_independent_holdout(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        seed=args.seed,
        count=args.count,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
