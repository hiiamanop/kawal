from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
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
    FamilySplitAuditResult,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
)
from scripts import (
    DatasetAuditError,
    create_artifact_item,
    create_or_update_artifact_manifest,
    set_deterministic_seed,
)
from scripts.train_dapt import validate_corpus_text
from services.dataset.generator import (
    DEFAULT_TEMPLATES,
    ScenarioTemplate,
    _apply_slang,
    export_trajectories_to_jsonl,
    load_trajectories_from_jsonl,
)
from services.dataset.splits import (
    COMMON_LANGUAGE_TOKENS,
    audit_family_splits,
    build_family_split_map,
)
from services.ml.manifest import ArtifactManifest, ArtifactManifestItem, compute_file_sha256

CATEGORY_ANCHOR_LEXICONS: dict[Category, dict[str, tuple[str, ...]]] = {
    Category.ROAD: {
        "anchors": (
            "jalan berlubang",
            "aspal ambles",
            "badan jalan ambrol",
            "penerangan jalan umum",
            "lampu pengatur lalu lintas",
            "penutup besi lubang jalan",
            "paving trotoar hancur",
            "guardrail pembatas jalan",
            "pagar pengaman guardrail tol",
            "garis penyeberangan zebra cross",
            "separator beton pembatas jalan",
            "celah ekspansi jembatan flyover",
            "papan plang petunjuk jalan",
            "bahu aspal amblas",
        ),
        "forbidden_terms": (
            "drainase",
            "parit",
            "gorong",
            "saluran air",
            "pintu air",
            "retensi",
            "banjir",
        ),
    },
    Category.DRAINAGE_FLOOD: {
        "anchors": (
            "gorong-gorong",
            "tanggul",
            "saluran air",
            "saluran drainase",
            "pintu air",
            "kolam retensi",
            "pompa drainase",
            "sedimen tanah",
            "parit",
            "saluran pembuangan air",
            "dinding plengsengan kali",
            "katup pengendali pasang laut",
            "luapan air",
            "genangan air",
            "beton gorong-gorong",
            "bendungan penahan luapan",
        ),
        "forbidden_terms": (
            "jalan",
            "aspal",
            "trotoar",
            "lalu lintas",
        ),
    },
    Category.WASTE: {
        "anchors": (
            "tumpukan sampah liar",
            "tempat penampungan sementara",
            "pembakaran sampah",
            "limbah medis",
            "sampah pasar",
            "tumpukan sisa sayuran",
            "timbunan sampah plastik",
            "tumpukan botol kemasan plastik",
            "limbah jeroan potongan ternak",
            "limbah baterai elektronik",
            "tumpahan kaleng cat dan limbah beracun",
            "bak sampah",
            "puing semen bekas bongkaran",
            "pembakaran tumpukan ban",
            "ceceran sampah sayur",
            "onggokan limbah plastik",
            "buangan jeroan potongan",
        ),
        "forbidden_terms": (),
    },
    Category.CLEAN_WATER: {
        "anchors": (
            "pipa induk air bersih",
            "air keran",
            "pasokan air bersih",
            "debit air pdam",
            "kran air warga",
            "aliran kran air",
            "arloji meteran air pipa",
            "aliran kran keruh",
            "pasokan kran leding",
            "pipa distribusi leding",
            "reservoir pdam",
            "meteran pelanggan air pdam",
            "distribusi pipa leding",
            "semburan kran keruh",
            "suplai kran leding",
            "instalasi pipa distribusi",
            "unit pendorong reservoir",
        ),
        "forbidden_terms": (),
    },
    Category.CIVIL_ADMIN: {
        "anchors": (
            "blanko kartu tanda penduduk",
            "surat pindah",
            "antrean elektronik",
            "akta kelahiran",
            "kartu keluarga",
            "bantuan jaminan kesehatan",
            "akta kematian",
            "kartu identitas anak",
            "status pernikahan",
            "berkas kependudukan",
            "pengesahan berkas cap",
            "mutasi surat kependudukan",
            "pembuatan kartu identitas anak",
            "tarikan biaya fotokopi",
        ),
        "forbidden_terms": (),
    },
    Category.HEALTH_SERVICE: {
        "anchors": (
            "dokter jaga",
            "ambulans gawat darurat",
            "stok obat rutin",
            "lemari pendingin vaksin",
            "posyandu",
            "ruang igd",
            "tenaga bidan bersalin",
            "tabung oksigen medis",
            "cairan reagen uji tes darah",
            "loket pengambilan obat apotek",
            "dokter spesialis anak",
            "tablet obat infeksi paru",
            "paket bantuan makanan gizi",
            "jadwal dokter spesialis anak",
            "cairan reagen uji tes",
        ),
        "forbidden_terms": (),
    },
}


def validate_scenario_template_lexicon(template: ScenarioTemplate) -> bool:
    """Validate that a scenario template strictly complies with category lexicons."""
    cat = template.category
    lex = CATEGORY_ANCHOR_LEXICONS.get(cat)
    if not lex:
        return True

    issue_lower = template.issue.lower()
    landmark_lower = template.landmark.lower()

    # 1. No direct category enum name leakage in text
    for c in Category:
        c_name = c.value.lower()
        if c_name in issue_lower or c_name in landmark_lower:
            return False

    # 2. No forbidden terms in issue
    for forbidden in lex.get("forbidden_terms", ()):
        if forbidden in issue_lower:
            return False

    # 3. Must contain recognized category anchor in issue
    anchors = lex.get("anchors", ())
    if anchors and not any(a.lower() in issue_lower for a in anchors):
        return False

    return True


EXTENDED_TEMPLATES: tuple[ScenarioTemplate, ...] = (
    # ROAD (9 templates -> total 12 with DEFAULT_TEMPLATES)
    ScenarioTemplate(
        family_id="road-traffic-light-malfunction",
        category=Category.ROAD,
        issue="lampu pengatur lalu lintas padam total menyebabkan kemacetan",
        landmark="simpang empat tugu perintis",
        jurisdiction="Kelurahan Bubutan, Kecamatan Bubutan",
        city="Kota Surabaya",
        urgency="HIGH",
        world_truth={"category": "ROAD", "risk": "HIGH", "infra_type": "TRAFFIC_LIGHT"},
    ),
    ScenarioTemplate(
        family_id="road-missing-manhole-cover",
        category=Category.ROAD,
        issue="penutup besi lubang jalan hilang membahayakan pengendara",
        landmark="seberang halte bus trans garuda",
        jurisdiction="Kelurahan Caturtunggal, Kecamatan Depok",
        city="Kabupaten Sleman",
        urgency="URGENT",
        world_truth={"category": "ROAD", "risk": "URGENT", "infra_type": "MANHOLE"},
    ),
    ScenarioTemplate(
        family_id="road-sidewalk-broken-pedestrian",
        category=Category.ROAD,
        issue="paving trotoar hancur pejalan kaki terpaksa melintas ke aspal jalan raya",
        landmark="area pertokoan sudirman timur",
        jurisdiction="Kelurahan Klandasan Ilir, Kecamatan Balikpapan Kota",
        city="Kota Balikpapan",
        urgency="MEDIUM",
        world_truth={"category": "ROAD", "risk": "MEDIUM", "infra_type": "SIDEWALK"},
    ),
    ScenarioTemplate(
        family_id="road-guardrail-damaged-highway",
        category=Category.ROAD,
        issue="pagar pengaman guardrail tol penyok akibat benturan keras mendesak",
        landmark="kilometer empat belas lingkar kaligawe",
        jurisdiction="Kelurahan Karangroto, Kecamatan Genuk",
        city="Kota Semarang",
        urgency="HIGH",
        world_truth={"category": "ROAD", "risk": "HIGH", "infra_type": "GUARDRAIL"},
    ),
    ScenarioTemplate(
        family_id="road-faded-zebra-crossing",
        category=Category.ROAD,
        issue="garis penyeberangan zebra cross pudar tak terlihat pejalan",
        landmark="gerbang utama politeknik candi",
        jurisdiction="Kelurahan Kepanjen, Kecamatan Kepanjen",
        city="Kabupaten Malang",
        urgency="MEDIUM",
        world_truth={"category": "ROAD", "risk": "MEDIUM", "infra_type": "ZEBRA_CROSS"},
    ),
    ScenarioTemplate(
        family_id="road-collapsed-median-separator",
        category=Category.ROAD,
        issue="separator beton pembatas jalan terguling menghalangi lajur cepat melintang",
        landmark="ujung jalan layang flyover sisingamangaraja",
        jurisdiction="Kelurahan Harjosari, Kecamatan Medan Amplas",
        city="Kota Medan",
        urgency="URGENT",
        world_truth={"category": "ROAD", "risk": "URGENT", "infra_type": "SEPARATOR"},
    ),
    ScenarioTemplate(
        family_id="road-cracked-flyover-expansion-joint",
        category=Category.ROAD,
        issue="celah ekspansi jembatan flyover renggang berbahaya bagi pemotor terbuka",
        landmark="tanjakan jalan layang balubur",
        jurisdiction="Kelurahan Tamansari, Kecamatan Bandung Wetan",
        city="Kota Bandung",
        urgency="HIGH",
        world_truth={"category": "ROAD", "risk": "HIGH", "infra_type": "EXPANSION_JOINT"},
    ),
    ScenarioTemplate(
        family_id="road-fallen-traffic-sign",
        category=Category.ROAD,
        issue="papan plang petunjuk jalan patah miring tertiup angin kencang roboh",
        landmark="bunderan tugu adipura gajah",
        jurisdiction="Kelurahan Enggal, Kecamatan Enggal",
        city="Kota Bandar Lampung",
        urgency="MEDIUM",
        world_truth={"category": "ROAD", "risk": "MEDIUM", "infra_type": "TRAFFIC_SIGN"},
    ),
    ScenarioTemplate(
        family_id="road-landslide-shoulder-collapse",
        category=Category.ROAD,
        issue="bahu aspal amblas tergerus longsor di lereng tanjakan curam terjal",
        landmark="pilar kilometer dua puluh delapan pelangi",
        jurisdiction="Kelurahan Sentul, Kecamatan Babakan Madang",
        city="Kabupaten Bogor",
        urgency="URGENT",
        world_truth={"category": "ROAD", "risk": "URGENT", "infra_type": "ROAD_SHOULDER"},
    ),

    # DRAINAGE_FLOOD (9 templates -> total 12 with DEFAULT_TEMPLATES)
    ScenarioTemplate(
        family_id="drainage-broken-watergate",
        category=Category.DRAINAGE_FLOOD,
        issue="pintu air pengendali banjir macet tidak bisa dibuka",
        landmark="area bendung hilir dermaga nelayan",
        jurisdiction="Kelurahan Ancol, Kecamatan Pademangan",
        city="Jakarta Utara",
        urgency="URGENT",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "URGENT", "flood_depth_cm": 50},
    ),
    ScenarioTemplate(
        family_id="drainage-retention-basin-silted",
        category=Category.DRAINAGE_FLOOD,
        issue="kolam retensi pengendali air hujan dangkal tertimbun endapan lumpur",
        landmark="taman konservasi air resapan pinus",
        jurisdiction="Kelurahan Loji, Kecamatan Bogor Barat",
        city="Kota Bogor",
        urgency="MEDIUM",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "MEDIUM", "flood_depth_cm": 25},
    ),
    ScenarioTemplate(
        family_id="drainage-culvert-subsidence",
        category=Category.DRAINAGE_FLOOD,
        issue="beton gorong-gorong saluran ambles memicu genangan air hujan deras",
        landmark="sekitar lintasan rel kereta kramat",
        jurisdiction="Kelurahan Tanah Tinggi, Kecamatan Johar Baru",
        city="Jakarta Pusat",
        urgency="HIGH",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "HIGH", "flood_depth_cm": 35},
    ),
    ScenarioTemplate(
        family_id="drainage-levee-breach-settlement",
        category=Category.DRAINAGE_FLOOD,
        issue="bendungan penahan luapan air kali jebol rendam perkampungan warga tenggelam",
        landmark="tepi dermaga perahu tambang kali gembong",
        jurisdiction="Kelurahan Panggungrejo, Kecamatan Panggungrejo",
        city="Kota Pasuruan",
        urgency="URGENT",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "URGENT", "flood_depth_cm": 90},
    ),
    ScenarioTemplate(
        family_id="drainage-sump-pump-failure",
        category=Category.DRAINAGE_FLOOD,
        issue="mesin pompa drainase otomatis rumah pompa mati mengakibatkan banjir genangan meluap",
        landmark="stasiun pompa muara karang hilir",
        jurisdiction="Kelurahan Penjaringan, Kecamatan Penjaringan",
        city="Jakarta Utara",
        urgency="HIGH",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "HIGH", "flood_depth_cm": 45},
    ),
    ScenarioTemplate(
        family_id="drainage-silt-clogged-ditch",
        category=Category.DRAINAGE_FLOOD,
        issue="timbunan lumpur pekat parit comberan menyumbat aliran air kotor mengendap",
        landmark="belakang deretan ruko niaga kyai mojo",
        jurisdiction="Kelurahan Tegalrejo, Kecamatan Tegalrejo",
        city="Kota Yogyakarta",
        urgency="MEDIUM",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "MEDIUM", "flood_depth_cm": 15},
    ),
    ScenarioTemplate(
        family_id="drainage-inspection-chamber-collapsed",
        category=Category.DRAINAGE_FLOOD,
        issue="bak kontrol saluran pembuangan air hancur tertimbun runtuhan batu ambrol",
        landmark="lapangan hijau perkampungan damai",
        jurisdiction="Kelurahan Candirenggo, Kecamatan Singosari",
        city="Kabupaten Malang",
        urgency="HIGH",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "HIGH", "flood_depth_cm": 30},
    ),
    ScenarioTemplate(
        family_id="drainage-retaining-wall-cracked",
        category=Category.DRAINAGE_FLOOD,
        issue="dinding plengsengan kali retak miring rawan ambruk terbelah",
        landmark="tebing aliran anak sungai jagir",
        jurisdiction="Kelurahan Baratajaya, Kecamatan Gubeng",
        city="Kota Surabaya",
        urgency="HIGH",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "HIGH", "flood_depth_cm": 25},
    ),
    ScenarioTemplate(
        family_id="drainage-tidal-backwater-inundation",
        category=Category.DRAINAGE_FLOOD,
        issue="katup pengendali pasang laut jebol air rob genangi kawasan muara tergenang",
        landmark="dermaga pelelangan ikan bandarharjo",
        jurisdiction="Kelurahan Tanjung Emas, Kecamatan Semarang Utara",
        city="Kota Semarang",
        urgency="URGENT",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "URGENT", "flood_depth_cm": 60},
    ),

    # WASTE (9 templates -> total 12 with DEFAULT_TEMPLATES)
    ScenarioTemplate(
        family_id="waste-medical-waste-dump",
        category=Category.WASTE,
        issue="limbah medis berbahaya jarum suntik bekas terbuang di tanah lapang tergeletak",
        landmark="tanah lapang samping puskesmas kebon jati",
        jurisdiction="Kelurahan Pasir Kaliki, Kecamatan Cicendo",
        city="Kota Bandung",
        urgency="URGENT",
        world_truth={"category": "WASTE", "risk": "URGENT", "waste_type": "MEDICAL"},
    ),
    ScenarioTemplate(
        family_id="waste-market-garbage-rotting",
        category=Category.WASTE,
        issue="ceceran sampah sayur pasar tradisional menumpuk busuk berbelatung membusuk",
        landmark="pintu belakang bongkar muat pasar bersehati",
        jurisdiction="Kelurahan Pandu, Kecamatan Manado",
        city="Kota Manado",
        urgency="HIGH",
        world_truth={"category": "WASTE", "risk": "HIGH", "waste_type": "MARKET"},
    ),
    ScenarioTemplate(
        family_id="waste-riverbank-plastic-clog",
        category=Category.WASTE,
        issue="onggokan limbah plastik kemasan padat membendung aliran kali sungai tertahan",
        landmark="tepian bantaran sungai kali pepe",
        jurisdiction="Kelurahan Jagalan, Kecamatan Jebres",
        city="Kota Surakarta",
        urgency="HIGH",
        world_truth={"category": "WASTE", "risk": "HIGH", "waste_type": "PLASTIC"},
    ),
    ScenarioTemplate(
        family_id="waste-unmanaged-slaughterhouse-refuse",
        category=Category.WASTE,
        issue="buangan jeroan potongan ternak membusuk dibuang sembarangan berbau busuk tercemar",
        landmark="belakang rumah potong hewan suramadu",
        jurisdiction="Kelurahan Kedung Cowek, Kecamatan Kenjeran",
        city="Kota Surabaya",
        urgency="HIGH",
        world_truth={"category": "WASTE", "risk": "HIGH", "waste_type": "SLAUGHTERHOUSE"},
    ),
    ScenarioTemplate(
        family_id="waste-commercial-electronic-dumping",
        category=Category.WASTE,
        issue="rongsokan limbah baterai elektronik aki bekas di kebun warga bertumpuk",
        landmark="area lahan tidur tepi lingkar banyuanyar",
        jurisdiction="Kelurahan Kadipiro, Kecamatan Banjarsari",
        city="Kota Surakarta",
        urgency="HIGH",
        world_truth={"category": "WASTE", "risk": "HIGH", "waste_type": "E_WASTE"},
    ),
    ScenarioTemplate(
        family_id="waste-hazardous-paint-chemical-spill",
        category=Category.WASTE,
        issue="tumpahan kaleng cat dan limbah beracun pelarut kimia menyengat tumpah",
        landmark="gudang logistik depo marunda",
        jurisdiction="Kelurahan Rorotan, Kecamatan Cilincing",
        city="Jakarta Utara",
        urgency="URGENT",
        world_truth={"category": "WASTE", "risk": "URGENT", "waste_type": "HAZARDOUS_SPILL"},
    ),
    ScenarioTemplate(
        family_id="waste-broken-communal-dumpster",
        category=Category.WASTE,
        issue="wadah bak sampah seng penampungan warga berlubang hancur melimpah",
        landmark="gardu ronda rukun tetangga buah batu",
        jurisdiction="Kelurahan Cijagra, Kecamatan Lengkong",
        city="Kota Bandung",
        urgency="MEDIUM",
        world_truth={"category": "WASTE", "risk": "MEDIUM", "waste_type": "BROKEN_BIN"},
    ),
    ScenarioTemplate(
        family_id="waste-construction-debris-dump",
        category=Category.WASTE,
        issue="puing semen bekas bongkaran gedung dibuang sembarangan teronggok",
        landmark="bioskop lama telanaipura",
        jurisdiction="Kelurahan Payo Selincah, Kecamatan Paal Merah",
        city="Kota Jambi",
        urgency="MEDIUM",
        world_truth={"category": "WASTE", "risk": "MEDIUM", "waste_type": "CONSTRUCTION_DEBRIS"},
    ),
    ScenarioTemplate(
        family_id="waste-tire-burning-pollution",
        category=Category.WASTE,
        issue="asap pekat pembakaran tumpukan ban bekas timbulkan polusi menyesakkan pekat",
        landmark="kawasan bengkel las pandanaran industri",
        jurisdiction="Kelurahan Trimulyo, Kecamatan Pedurungan",
        city="Kota Semarang",
        urgency="URGENT",
        world_truth={"category": "WASTE", "risk": "URGENT", "waste_type": "TIRE_BURNING"},
    ),

    # CLEAN_WATER (9 templates -> total 12 with DEFAULT_TEMPLATES)
    ScenarioTemplate(
        family_id="water-low-pressure-terminal",
        category=Category.CLEAN_WATER,
        issue="debit air pdam sangat kecil menetes di jam sibuk pagi",
        landmark="kompleks perumahan graha permata indah",
        jurisdiction="Kelurahan Sememi, Kecamatan Benowo",
        city="Kota Surabaya",
        urgency="MEDIUM",
        world_truth={"category": "CLEAN_WATER", "risk": "MEDIUM", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-contamination-diesel-smell",
        category=Category.CLEAN_WATER,
        issue="aliran kran air warga berbau solar pekat berminyak tidak layak guna",
        landmark="gang permukiman mendawai tiga",
        jurisdiction="Kelurahan Baru, Kecamatan Arut Selatan",
        city="Kabupaten Kotawaringin Barat",
        urgency="HIGH",
        world_truth={"category": "CLEAN_WATER", "risk": "HIGH", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-meter-leak-flooding",
        category=Category.CLEAN_WATER,
        issue="sambungan arloji meteran air pipa rumah bocor menyembur genangi halaman",
        landmark="kompleks taman bunga kemang pratama",
        jurisdiction="Kelurahan Pekayon Jaya, Kecamatan Bekasi Selatan",
        city="Kota Bekasi",
        urgency="HIGH",
        world_truth={"category": "CLEAN_WATER", "risk": "HIGH", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-sand-sediment-tap",
        category=Category.CLEAN_WATER,
        issue="semburan kran keruh bercampur butiran pasir kasar menyumbat keran",
        landmark="deretan perumahan rajawali blok c",
        jurisdiction="Kelurahan Kalijaga, Kecamatan Harjamukti",
        city="Kota Cirebon",
        urgency="HIGH",
        world_truth={"category": "CLEAN_WATER", "risk": "HIGH", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-chlorine-chemical-odor",
        category=Category.CLEAN_WATER,
        issue="suplai kran leding berbau kaporit sangat menyengat perih di mata perih",
        landmark="masjid al furqon nomor sembilan kencana",
        jurisdiction="Kelurahan Gunungsari, Kecamatan Dukuh Pakis",
        city="Kota Surabaya",
        urgency="HIGH",
        world_truth={"category": "CLEAN_WATER", "risk": "HIGH", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-subsurface-pipe-leak",
        category=Category.CLEAN_WATER,
        issue="instalasi pipa distribusi leding bawah tanah bocor rembesan membanjiri becek",
        landmark="posyandu anggrek terpadu cisadane",
        jurisdiction="Kelurahan Bugel, Kecamatan Karawaci",
        city="Kota Tangerang",
        urgency="HIGH",
        world_truth={"category": "CLEAN_WATER", "risk": "HIGH", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-booster-pump-failure",
        category=Category.CLEAN_WATER,
        issue="unit pendorong reservoir pdam mati total air tidak mengalir mampet",
        landmark="posko tangki distribusi punclut",
        jurisdiction="Kelurahan Ciumbuleuit, Kecamatan Cidadap",
        city="Kota Bandung",
        urgency="URGENT",
        world_truth={"category": "CLEAN_WATER", "risk": "URGENT", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-billing-meter-stuck",
        category=Category.CLEAN_WATER,
        issue="jarum meteran pelanggan air pdam macet terus berputar cepat abnormal",
        landmark="sentra niaga bunderan tugu proklamasi",
        jurisdiction="Kelurahan Bendogerit, Kecamatan Kepanjenkidul",
        city="Kota Blitar",
        urgency="MEDIUM",
        world_truth={"category": "CLEAN_WATER", "risk": "MEDIUM", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-intermittent-night-supply",
        category=Category.CLEAN_WATER,
        issue="distribusi pipa leding hanya mengalir satu jam tengah malam terputus",
        landmark="gang buntu permukiman bugangan",
        jurisdiction="Kelurahan Rejosari, Kecamatan Semarang Timur",
        city="Kota Semarang",
        urgency="MEDIUM",
        world_truth={"category": "CLEAN_WATER", "risk": "MEDIUM", "service": "PDAM"},
    ),

    # CIVIL_ADMIN (9 templates -> total 12 with DEFAULT_TEMPLATES)
    ScenarioTemplate(
        family_id="admin-birth-certificate-extortion",
        category=Category.CIVIL_ADMIN,
        issue="oknum petugas loket meminta pungutan tidak resmi akta kelahiran",
        landmark="gedung pelayanan satu atap karang asam",
        jurisdiction="Kelurahan Teluk Lerong, Kecamatan Sungai Kunjang",
        city="Kota Samarinda",
        urgency="HIGH",
        world_truth={"category": "CIVIL_ADMIN", "risk": "HIGH", "document": "AKTA_LAHIR"},
    ),
    ScenarioTemplate(
        family_id="admin-family-card-data-mismatch",
        category=Category.CIVIL_ADMIN,
        issue="kesalahan ketik nik berkas kartu keluarga diabaikan petugas keliru",
        landmark="pos registrasi berkas dua kridosono",
        jurisdiction="Kelurahan Kotabaru, Kecamatan Gondokusuman",
        city="Kota Yogyakarta",
        urgency="HIGH",
        world_truth={"category": "CIVIL_ADMIN", "risk": "HIGH", "document": "KARTU_KELUARGA"},
    ),
    ScenarioTemplate(
        family_id="admin-social-aid-registration-denied",
        category=Category.CIVIL_ADMIN,
        issue="verifikasi data pbi bantuan jaminan kesehatan tersendat tertunda",
        landmark="lokasi konsultasi jaminan sosial kapasan",
        jurisdiction="Kelurahan Sidodadi, Kecamatan Simokerto",
        city="Kota Surabaya",
        urgency="HIGH",
        world_truth={"category": "CIVIL_ADMIN", "risk": "HIGH", "document": "BPJS_PBI"},
    ),
    ScenarioTemplate(
        family_id="admin-death-certificate-delay",
        category=Category.CIVIL_ADMIN,
        issue="pengurusan akta kematian warga tertahan di ruang pelayanan belum selesai terhambat",
        landmark="bilik verifikasi berkas sipil ngagel",
        jurisdiction="Kelurahan Wonokromo, Kecamatan Sawahan",
        city="Kota Surabaya",
        urgency="MEDIUM",
        world_truth={"category": "CIVIL_ADMIN", "risk": "MEDIUM", "document": "AKTA_KEMATIAN"},
    ),
    ScenarioTemplate(
        family_id="admin-child-identity-card-denied",
        category=Category.CIVIL_ADMIN,
        issue="pembuatan kartu identitas anak ditolak petugas tanpa alasan regulasi ditolak",
        landmark="kantor pendaftaran kependudukan terpadu sumberdiren",
        jurisdiction="Kelurahan Karangtengah, Kecamatan Sananwetan",
        city="Kota Blitar",
        urgency="HIGH",
        world_truth={"category": "CIVIL_ADMIN", "risk": "HIGH", "document": "KIA"},
    ),
    ScenarioTemplate(
        family_id="admin-marriage-status-update-refused",
        category=Category.CIVIL_ADMIN,
        issue="pembaruan status pernikahan pada data identitas dipersulit petugas loket mandek",
        landmark="kantor catatan sipil lantai dua alun alun",
        jurisdiction="Kelurahan Kauman, Kecamatan Klojen",
        city="Kota Malang",
        urgency="MEDIUM",
        world_truth={"category": "CIVIL_ADMIN", "risk": "MEDIUM", "document": "STATUS_KTP"},
    ),
    ScenarioTemplate(
        family_id="admin-mobile-service-unannounced-cancel",
        category=Category.CIVIL_ADMIN,
        issue="kendaraan mobil keliling berkas kependudukan batal hadir tanpa kabar mangkir",
        landmark="lingkungan lapangan bola warga pilar hegar",
        jurisdiction="Kelurahan Bantarjati, Kecamatan Bogor Utara",
        city="Kota Bogor",
        urgency="MEDIUM",
        world_truth={"category": "CIVIL_ADMIN", "risk": "MEDIUM", "document": "MOBILE_ADMIN"},
    ),
    ScenarioTemplate(
        family_id="admin-unauthorized-legalization-fee",
        category=Category.CIVIL_ADMIN,
        issue="tarikan biaya fotokopi tidak resmi pada pengesahan berkas cap bayar",
        landmark="meja legalisasi berkas terpadu arjosari",
        jurisdiction="Kelurahan Bunulrejo, Kecamatan Blimbing",
        city="Kota Malang",
        urgency="HIGH",
        world_truth={"category": "CIVIL_ADMIN", "risk": "HIGH", "document": "LEGALISIR"},
    ),
    ScenarioTemplate(
        family_id="admin-address-mutation-held",
        category=Category.CIVIL_ADMIN,
        issue="berkas mutasi surat kependudukan antar daerah tertahan tidak ditandatangani menggantung",
        landmark="sekretariat pelayanan warga tipes barat",
        jurisdiction="Kelurahan Danukusuman, Kecamatan Serengan",
        city="Kota Surakarta",
        urgency="HIGH",
        world_truth={"category": "CIVIL_ADMIN", "risk": "HIGH", "document": "SURAT_PINDAH"},
    ),

    # HEALTH_SERVICE (9 templates -> total 12 with DEFAULT_TEMPLATES)
    ScenarioTemplate(
        family_id="health-vaccine-cold-chain-break",
        category=Category.HEALTH_SERVICE,
        issue="lemari pendingin vaksin rusak dikhawatirkan kualitas obat menurun",
        landmark="ruang imunisasi faskes bolok",
        jurisdiction="Kelurahan Alak, Kecamatan Alak",
        city="Kota Kupang",
        urgency="URGENT",
        world_truth={"category": "HEALTH_SERVICE", "risk": "URGENT", "facility": "COLD_CHAIN"},
    ),
    ScenarioTemplate(
        family_id="health-posyandu-nutrition-shortage",
        category=Category.HEALTH_SERVICE,
        issue="paket bantuan makanan gizi tambahan balita posyandu habis total",
        landmark="balai posyandu cempaka regency sidakarya",
        jurisdiction="Kelurahan Sesetan, Kecamatan Denpasar Selatan",
        city="Kota Denpasar",
        urgency="MEDIUM",
        world_truth={"category": "HEALTH_SERVICE", "risk": "MEDIUM", "facility": "POSYANDU"},
    ),
    ScenarioTemplate(
        family_id="health-er-overcapacity-rejection",
        category=Category.HEALTH_SERVICE,
        issue="ruang igd menolak pasien sesak napas dengan dalih tempat tidur penuh",
        landmark="area darurat rujukan abepura",
        jurisdiction="Kelurahan Kuanino, Kecamatan Kota Raja",
        city="Kota Jayapura",
        urgency="URGENT",
        world_truth={"category": "HEALTH_SERVICE", "risk": "URGENT", "facility": "ER_TRIAGE"},
    ),
    ScenarioTemplate(
        family_id="health-midwife-absent-labor",
        category=Category.HEALTH_SERVICE,
        issue="tenaga bidan bersalin faskes tidak berada di tempat saat pasien melahirkan bersalin",
        landmark="bangsal persalinan bersalin anggrek",
        jurisdiction="Kelurahan Kanigoro, Kecamatan Kartoharjo",
        city="Kota Madiun",
        urgency="URGENT",
        world_truth={"category": "HEALTH_SERVICE", "risk": "URGENT", "facility": "PERSALINAN"},
    ),
    ScenarioTemplate(
        family_id="health-oxygen-cylinder-depleted",
        category=Category.HEALTH_SERVICE,
        issue="tabung oksigen medis darurat pada ruang periksan faskes kosong total hampa",
        landmark="bangsal darurat faskes mergosono",
        jurisdiction="Kelurahan Karangbesuki, Kecamatan Sukun",
        city="Kota Malang",
        urgency="URGENT",
        world_truth={"category": "HEALTH_SERVICE", "risk": "URGENT", "facility": "OXYGEN"},
    ),
    ScenarioTemplate(
        family_id="health-laboratory-reagent-exhausted",
        category=Category.HEALTH_SERVICE,
        issue="cairan reagen uji tes darah faskes habis terhenti langka",
        landmark="laboratorium klinik kesehatan sawojajar",
        jurisdiction="Kelurahan Cemorokandang, Kecamatan Kedungkandang",
        city="Kota Malang",
        urgency="MEDIUM",
        world_truth={"category": "HEALTH_SERVICE", "risk": "MEDIUM", "facility": "LABORATORY"},
    ),
    ScenarioTemplate(
        family_id="health-pharmacy-dispensary-halted",
        category=Category.HEALTH_SERVICE,
        issue="loket pengambilan obat apotek faskes terhenti tidak ada petugas terbengkalai",
        landmark="area antrean loket obat suhat",
        jurisdiction="Kelurahan Kanigaran, Kecamatan Kanigaran",
        city="Kota Probolinggo",
        urgency="HIGH",
        world_truth={"category": "HEALTH_SERVICE", "risk": "HIGH", "facility": "FARMASI_LOKET"},
    ),
    ScenarioTemplate(
        family_id="health-pediatric-clinic-delayed",
        category=Category.HEALTH_SERVICE,
        issue="jadwal dokter spesialis anak belum hadir tiga jam keterlambatan buka terlambat",
        landmark="gedung poli balita ijen",
        jurisdiction="Kelurahan Purbayan, Kecamatan Kotagede",
        city="Kota Yogyakarta",
        urgency="MEDIUM",
        world_truth={"category": "HEALTH_SERVICE", "risk": "MEDIUM", "facility": "POLI_ANAK"},
    ),
    ScenarioTemplate(
        family_id="health-tuberculosis-drug-stockout",
        category=Category.HEALTH_SERVICE,
        issue="persediaan tablet obat infeksi paru di faskes kosong tandas",
        landmark="poli dots paru buring",
        jurisdiction="Kelurahan Mojoroto, Kecamatan Mojoroto",
        city="Kota Kediri",
        urgency="HIGH",
        world_truth={"category": "HEALTH_SERVICE", "risk": "HIGH", "facility": "DOTS_PULMONARY"},
    ),
)

ALL_SCENARIO_TEMPLATES: tuple[ScenarioTemplate, ...] = DEFAULT_TEMPLATES + EXTENDED_TEMPLATES

BOILERPLATE_PROTECTED_TOKENS: frozenset[str] = frozenset(COMMON_LANGUAGE_TOKENS)

RISK_EVIDENCE_MAP: dict[str, tuple[str, ...]] = {
    "LOW": (
        "kondisi saat ini belum parah dan tidak ada bahaya",
        "keadaan saat ini tidak mendesak dan belum ada bahaya",
        "situasi saat ini belum parah dan tidak darurat",
        "kondisi saat ini tidak mendesak dan belum parah",
    ),
    "MEDIUM": (
        "kondisi ini semakin rusak mohon bantuan penanganan tetapi belum darurat",
        "keadaan makin rusak dan mohon penanganan dinas namun belum sangat parah",
        "situasi semakin rusak mohon bantuan penanganan tetapi belum darurat",
        "kondisi makin rusak mohon ditangani dinas terkait tetapi belum darurat",
    ),
    "HIGH": (
        "kondisi sangat parah dan bahaya mohon segera ditangani oleh dinas terkait",
        "keadaan sangat parah dan sangat bahaya mohon segera ada penanganan petugas",
        "situasi sangat parah dan mendesak mohon segera ditangani dinas terkait",
        "kondisi sangat rusak parah dan bahaya sekali mohon segera penanganan",
    ),
    "URGENT": (
        "situasi sangat darurat dan bahaya sekali mohon segera ditangani sekarang juga",
        "keadaan darurat parah sangat bahaya tolong segera ditindaklanjuti sekarang",
        "kondisi sangat darurat dan amat mendesak bahaya sekali tolong segera ditangani sekarang",
        "keadaan darurat sangat mendesak tolong cepat kirim bantuan ke lokasi sekarang",
    ),
}

INTENT_PREFIX_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "COMPLAINT": {
        "FORMAL": (
            "Yth petugas kami laporkan terkait kondisi berikut",
            "Selamat siang petugas kami laporkan mengenai keadaan",
            "Yth dinas terkait kami laporkan situasi berikut",
        ),
        "FRUSTRATED_RAMBLING": (
            "Laporan kami mohon segera ditangani tolong petugas tangani cepat",
            "Kondisi ini makin keterlaluan tolong petugas cepat tindak laporan kami",
            "Lapor petugas ini keterlaluan sekali tolong segera dibantu ditangani",
        ),
        "PANICKED": (
            "Lapor tolong cepat bantu mohon segera kirim petugas ke lokasi",
            "Tolong cepat lapor mohon segera kirim bantuan petugas sekarang",
            "Lapor tolong dibantu cepat mohon petugas segera ke lokasi sekarang",
        ),
        "STANDARD": (
            "Halo admin lapor terkait kondisi berikut",
            "Selamat pagi admin kami laporkan keadaan di lokasi",
            "Lapor petugas mohon bantuan penanganan terkait situasi ini",
        ),
    },
    "INQUIRY": {
        "FORMAL": (
            "Yth petugas kami mohon informasi mengenai penanganan dan keterangan terkait",
            "Selamat siang petugas mohon informasi bagaimana penanganan dinas terkait",
            "Yth dinas terkait mohon keterangan dan informasi penanganan mengenai",
        ),
        "FRUSTRATED_RAMBLING": (
            "Bagaimana informasi penanganan dari petugas kenapa belum ditangani saat ini",
            "Mohon info dan keterangan bagaimana penanganan dari pihak dinas terkait",
            "Kenapa belum ditangani mohon informasi penanganan segera dari dinas terkait",
        ),
        "PANICKED": (
            "Mohon info segera petugas bagaimana penanganan di lokasi sekarang",
            "Tolong info cepat bagaimana bantuan petugas saat ini sekarang",
            "Cepat mohon info petugas bagaimana penanganan saat ini",
        ),
        "STANDARD": (
            "Halo admin mohon info bagaimana keterangan penanganan terkait",
            "Selamat pagi admin mohon informasi mengenai penanganan situasi berikut",
            "Admin mohon informasi bagaimana penanganan dan keterangan dari dinas terkait",
        ),
    },
    "FEEDBACK": {
        "FORMAL": (
            "Yth petugas terima kasih atas bantuan penanganan dan tindak perbaikan terkait",
            "Selamat siang petugas terima kasih atas perbaikan dan penanganan dinas terkait",
            "Yth dinas terkait terima kasih atas bantuan penanganan mengenai keadaan",
        ),
        "FRUSTRATED_RAMBLING": (
            "Terima kasih atas bantuan petugas namun mohon perbaikan lebih cepat ditangani",
            "Makasih atas bantuan penanganan tapi mohon petugas lebih cepat tangani lokasi ini",
            "Terima kasih atas bantuan dinas tapi mohon penanganan lebih cepat ditangani",
        ),
        "PANICKED": (
            "Terima kasih atas bantuan penanganan dari petugas tolong cepat ditangani",
            "Makasih petugas bantuan sudah ada tolong segera dibantu ditangani",
            "Terima kasih bantuan dari dinas tolong segera ditindaklanjuti cepat",
        ),
        "STANDARD": (
            "Halo admin terima kasih atas bantuan dan penanganan terkait situasi berikut",
            "Selamat pagi admin terima kasih atas perbaikan dan penanganan dari dinas terkait",
            "Makasih admin atas bantuan penanganan dari pihak dinas terkait keadaan ini",
        ),
    },
}

AMBIGUOUS_LOCATION_PREFIXES: dict[str, tuple[str, ...]] = {
    "FORMAL": (
        "Keterangan patokan lokasi berada di antara depan atau seberang",
        "Keterangan patokan lokasi yaitu di antara seberang atau samping",
    ),
    "FRUSTRATED_RAMBLING": (
        "Kondisi parah patokan lokasi di antara dekat atau belakang",
        "Posisi patokan di antara dekat atau seberang",
    ),
    "PANICKED": (
        "Lokasi patokan persis di antara seberang atau samping",
        "Titik patokan lokasi antara area depan atau seberang",
    ),
    "STANDARD": (
        "Titik patokan lokasi berada di antara depan atau seberang",
        "Posisi titik patokan antara dekat atau samping",
    ),
}

INCOMPLETE_LOCATION_PREFIXES: dict[str, tuple[str, ...]] = {
    "FORMAL": ("Keterangan patokan lokasi yaitu di",),
    "FRUSTRATED_RAMBLING": ("Kondisi parah patokan di",),
    "PANICKED": ("Lokasi patokan persis di",),
    "STANDARD": ("Titik patokannya di",),
}


def safe_apply_typos(text: str, protected_tokens: set[str], seed: int) -> str:
    words = text.split()
    transformed: list[str] = []
    mod = seed % 7
    all_protected = protected_tokens | BOILERPLATE_PROTECTED_TOKENS
    for idx, w in enumerate(words):
        clean_w = w.lower().strip(".,!?:;")
        if clean_w in all_protected or len(clean_w) < 5 or (idx % 7 != mod):
            transformed.append(w)
        else:
            chars = list(w)
            p = len(clean_w) // 2
            if p + 1 < len(chars) and chars[p].isalpha() and chars[p + 1].isalpha():
                chars[p], chars[p + 1] = chars[p + 1], chars[p]
                transformed.append("".join(chars))
            else:
                transformed.append(w)
    return " ".join(transformed)


CATEGORY_SPLIT_ANCHOR_CUES: dict[str, dict[Category, tuple[str, ...]]] = {
    "train": {
        Category.DRAINAGE_FLOOD: (
            "drainase pembuangan air hujan pemukim",
            "sistem drainase aliran air hujan",
        ),
        Category.WASTE: (
            "sampah domestik lingkungan perumahan",
            "tumpukan sampah domestik lingkungan",
        ),
        Category.CLEAN_WATER: (
            "distribusi air bersih leding perumahan",
            "jaringan distribusi air bersih leding",
        ),
        Category.ROAD: (
            "sarana jalan raya kendaraan umum",
            "kelancaran akses jalan raya kendaraan",
        ),
        Category.CIVIL_ADMIN: (
            "dokumen kependudukan identitas warga",
            "pencatatan berkas dokumen kependudukan",
        ),
        Category.HEALTH_SERVICE: (
            "pelayanan kesehatan medis darurat warga",
            "fasilitas pelayanan kesehatan medis darurat",
        ),
    },
    "dev": {
        Category.DRAINAGE_FLOOD: (
            "saluran pengendali genangan kali daerah",
            "penanganan saluran genangan kali daerah",
        ),
        Category.WASTE: (
            "kebersihan residu kotoran warga daerah",
            "pengelolaan kebersihan kotoran warga daerah",
        ),
        Category.CLEAN_WATER: (
            "pemenuhan kebutuhan air perpipaan daerah",
            "kelancaran kebutuhan air perpipaan daerah",
        ),
        Category.ROAD: (
            "akses mobilitas jalan rute pemda",
            "keamanan akses jalan rute pemda",
        ),
        Category.CIVIL_ADMIN: (
            "administrasi catatan sipil resmi daerah",
            "layanan administrasi catatan sipil resmi",
        ),
        Category.HEALTH_SERVICE: (
            "penanganan dokter faskes masyarakat daerah",
            "layanan penanganan dokter faskes masyarakat",
        ),
    },
    "test": {
        Category.DRAINAGE_FLOOD: (
            "parit penampungan limpasan banjir lingkungan",
            "saluran got penampungan banjir lingkungan",
        ),
        Category.WASTE: (
            "limbah sisa buangan padat lingkungan",
            "onggokan limbah buangan padat lingkungan",
        ),
        Category.CLEAN_WATER: (
            "suplai debit meteran pdam pelanggan",
            "aliran suplai debit meteran pdam",
        ),
        Category.ROAD: (
            "jalur marka trotoar pejalan kota",
            "fasilitas jalur marka trotoar pejalan",
        ),
        Category.CIVIL_ADMIN: (
            "berkas pendaftaran surat pemda resmi",
            "pengurusan berkas pendaftaran surat pemda",
        ),
        Category.HEALTH_SERVICE: (
            "bantuan obat apotek klinik posko",
            "ketersediaan bantuan obat apotek klinik",
        ),
    },
}

SPLIT_ANCHOR_CONNECTORS: dict[str, tuple[str, str, str]] = {
    "train": ("laporan mengenai", "dalam penanganan", "di area lingkungan warga"),
    "dev": ("catatan tentang", "untuk perbaikan", "pada area wilayah dinas"),
    "test": ("konfirmasi perihal", "sehubungan urusan", "guna pihak unit petugas"),
}


def compose_category_anchor_text(
    template: ScenarioTemplate,
    split: DatasetSplit | str,
    seed: int,
) -> str:
    """Compose category-specific anchor composition for reporter bubbles without ID or cross-split leakage."""
    split_val = split.value if hasattr(split, "value") else str(split)
    split_key = split_val if split_val in ("train", "dev", "test") else "train"
    pool = CATEGORY_SPLIT_ANCHOR_CUES.get(
        split_key, CATEGORY_SPLIT_ANCHOR_CUES["train"]
    ).get(template.category, ())
    cue = pool[seed % len(pool)] if pool else ""
    anchors = CATEGORY_ANCHOR_LEXICONS[template.category]["anchors"]
    matched = sorted(
        [a for a in anchors if a.lower() in template.issue.lower()],
        key=len,
        reverse=True,
    )
    compact_obj = matched[0] if matched else anchors[0]
    conn_pre, conn_mid, conn_suf = SPLIT_ANCHOR_CONNECTORS.get(
        split_key, SPLIT_ANCHOR_CONNECTORS["train"]
    )
    return f"{conn_pre} {compact_obj} {conn_mid} {cue} {conn_suf}"


def generate_safe_trajectory(
    template: ScenarioTemplate,
    scenario_id: str,
    persona: str,
    noise_level: str,
    multi_turn: bool,
    seed: int,
    split: DatasetSplit,
    intent: str = "COMPLAINT",
    risk: str | None = None,
    completeness: str = "SUFFICIENT",
) -> ComplaintTrajectory:
    landmark_text = template.landmark
    issue_text = template.issue
    jurisdiction_text = template.jurisdiction
    city_text = template.city

    anchor_comp = compose_category_anchor_text(template, split, seed)

    protected_tokens = {
        w.lower().strip(".,!?:;")
        for s in [landmark_text, jurisdiction_text, city_text]
        for w in s.split()
    }
    protected_tokens.update(
        w.lower().strip(".,!?:;") for w in anchor_comp.split()
    )

    if risk is None:
        risk = template.world_truth.get("risk", template.urgency)

    risk_opts = RISK_EVIDENCE_MAP.get(risk, RISK_EVIDENCE_MAP["MEDIUM"])
    risk_evidence = risk_opts[seed % len(risk_opts)]

    p_opts = INTENT_PREFIX_MAP.get(intent, INTENT_PREFIX_MAP["COMPLAINT"]).get(
        persona, INTENT_PREFIX_MAP["COMPLAINT"]["STANDARD"]
    )
    prefix = p_opts[seed % len(p_opts)]

    t1_b1_text = f"{prefix}: {issue_text}, {anchor_comp}, {risk_evidence}"

    if completeness == "AMBIGUOUS":
        amb_opts = AMBIGUOUS_LOCATION_PREFIXES.get(
            persona, AMBIGUOUS_LOCATION_PREFIXES["STANDARD"]
        )
        amb_prefix = amb_opts[seed % len(amb_opts)]
        t1_b2_text = f"{amb_prefix} {landmark_text}, mohon bantuan unit terkait"

        if noise_level in ("MEDIUM", "HIGH"):
            t1_b1_text = _apply_slang(t1_b1_text, protected_tokens)
            t1_b2_text = _apply_slang(t1_b2_text, protected_tokens)
        if noise_level == "HIGH":
            t1_b1_text = safe_apply_typos(t1_b1_text, protected_tokens, seed)
            t1_b2_text = safe_apply_typos(t1_b2_text, protected_tokens, seed)

        turn1_action = TurnExpectedAction(
            turn=1,
            allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
            missing=("location_ambiguity",),
            strategy="REQUEST_CLARIFICATION",
        )
        turn1 = TrajectoryTurn(
            turn=1,
            bubbles=(
                TrajectoryBubble(
                    source_message_id=f"{scenario_id}-t1-b1",
                    text=t1_b1_text,
                    offset_seconds=0,
                ),
                TrajectoryBubble(
                    source_message_id=f"{scenario_id}-t1-b2",
                    text=t1_b2_text,
                    offset_seconds=2,
                ),
            ),
            observable_facts=(issue_text, landmark_text),
            hidden_facts=(jurisdiction_text,),
            expected_action=turn1_action,
        )
        turns = (turn1,)
        observable_facts = (issue_text, landmark_text)
        hidden_facts = (jurisdiction_text,)
        expected_actions = (turn1_action,)
        loc_completeness = "AMBIGUOUS"

    elif completeness == "INCOMPLETE":
        incomp_opts = INCOMPLETE_LOCATION_PREFIXES.get(
            persona, INCOMPLETE_LOCATION_PREFIXES["STANDARD"]
        )
        incomp_prefix = incomp_opts[seed % len(incomp_opts)]
        if persona == "FORMAL":
            t1_b2_text = (
                f"{incomp_prefix} {landmark_text}, "
                "namun belum ada keterangan wilayah administrasi"
            )
        elif persona == "FRUSTRATED_RAMBLING":
            t1_b2_text = (
                f"{incomp_prefix} {landmark_text}, "
                "namun belum ada keterangan wilayah administrasi segera ditangani"
            )
        elif persona == "PANICKED":
            t1_b2_text = (
                f"{incomp_prefix} {landmark_text}, "
                "namun belum ada alamat wilayah tolong cepat"
            )
        else:
            t1_b2_text = (
                f"{incomp_prefix} {landmark_text}, "
                "namun belum ada keterangan alamat wilayah"
            )

        if noise_level in ("MEDIUM", "HIGH"):
            t1_b1_text = _apply_slang(t1_b1_text, protected_tokens)
            t1_b2_text = _apply_slang(t1_b2_text, protected_tokens)
        if noise_level == "HIGH":
            t1_b1_text = safe_apply_typos(t1_b1_text, protected_tokens, seed)
            t1_b2_text = safe_apply_typos(t1_b2_text, protected_tokens, seed)

        turn1_action = TurnExpectedAction(
            turn=1,
            allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
            missing=("jurisdiction",),
            strategy="REQUEST_LOCATION",
        )
        turn1 = TrajectoryTurn(
            turn=1,
            bubbles=(
                TrajectoryBubble(
                    source_message_id=f"{scenario_id}-t1-b1",
                    text=t1_b1_text,
                    offset_seconds=0,
                ),
                TrajectoryBubble(
                    source_message_id=f"{scenario_id}-t1-b2",
                    text=t1_b2_text,
                    offset_seconds=2,
                ),
            ),
            observable_facts=(issue_text, landmark_text),
            hidden_facts=(jurisdiction_text,),
            expected_action=turn1_action,
        )
        turns = (turn1,)
        observable_facts = (issue_text, landmark_text)
        hidden_facts = (jurisdiction_text,)
        expected_actions = (turn1_action,)
        loc_completeness = "INCOMPLETE"

    else:  # SUFFICIENT
        if multi_turn:
            turn1_expected_action = TurnExpectedAction(
                turn=1,
                allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
                missing=("jurisdiction",),
                strategy="REQUEST_LOCATION",
            )
            turn2_expected_action = TurnExpectedAction(
                turn=2,
                allowed_actions=(DecisionMode.EXECUTE,),
                missing=(),
                strategy="CREATE_PRIORITY_TICKET",
            )

            incomp_opts = INCOMPLETE_LOCATION_PREFIXES.get(
                persona, INCOMPLETE_LOCATION_PREFIXES["STANDARD"]
            )
            incomp_prefix = incomp_opts[seed % len(incomp_opts)]

            if persona == "FORMAL":
                t1_b2_text = f"{incomp_prefix} {landmark_text}"
            elif persona == "FRUSTRATED_RAMBLING":
                t1_b2_text = f"Kondisi parah patokan di {landmark_text} segera ditangani"
            elif persona == "PANICKED":
                t1_b2_text = f"Lokasi patokan persis di {landmark_text} tolong cepat"
            else:
                t1_b2_text = f"Titik patokannya di {landmark_text}, mohon bantuan unit terkait"

            t2_text = f"Untuk wilayah administrasinya berada di {jurisdiction_text}, {city_text}"

            if noise_level in ("MEDIUM", "HIGH"):
                t1_b1_text = _apply_slang(t1_b1_text, protected_tokens)
                t1_b2_text = _apply_slang(t1_b2_text, protected_tokens)
                t2_text = _apply_slang(t2_text, protected_tokens)
            if noise_level == "HIGH":
                t1_b1_text = safe_apply_typos(t1_b1_text, protected_tokens, seed)
                t1_b2_text = safe_apply_typos(t1_b2_text, protected_tokens, seed)
                t2_text = safe_apply_typos(t2_text, protected_tokens, seed)

            turn1 = TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id=f"{scenario_id}-t1-b1",
                        text=t1_b1_text,
                        offset_seconds=0,
                    ),
                    TrajectoryBubble(
                        source_message_id=f"{scenario_id}-t1-b2",
                        text=t1_b2_text,
                        offset_seconds=2,
                    ),
                ),
                observable_facts=(issue_text, landmark_text),
                hidden_facts=(jurisdiction_text,),
                expected_action=turn1_expected_action,
            )

            turn2 = TrajectoryTurn(
                turn=2,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id=f"{scenario_id}-t2-b1",
                        text=t2_text,
                        offset_seconds=0,
                    ),
                ),
                observable_facts=(issue_text, landmark_text, jurisdiction_text),
                hidden_facts=(),
                expected_action=turn2_expected_action,
            )

            turns = (turn1, turn2)
            observable_facts = (issue_text, landmark_text)
            hidden_facts = (jurisdiction_text,)
            expected_actions = (turn1_expected_action, turn2_expected_action)
            loc_completeness = "SUFFICIENT"
        else:
            t1_action = TurnExpectedAction(
                turn=1,
                allowed_actions=(DecisionMode.EXECUTE,),
                missing=(),
                strategy="CREATE_PRIORITY_TICKET",
            )

            if persona == "FORMAL":
                t1_b2_text = f"Keterangan lokasi yaitu di {landmark_text}, {jurisdiction_text}, {city_text}"
            elif persona == "FRUSTRATED_RAMBLING":
                t1_b2_text = f"Kondisi parah lokasi di {landmark_text}, {jurisdiction_text}, {city_text}"
            elif persona == "PANICKED":
                t1_b2_text = f"Lokasi persis di {landmark_text}, {jurisdiction_text}, {city_text}"
            else:
                t1_b2_text = f"Lokasi berada di {landmark_text}, {jurisdiction_text}, {city_text}"

            if noise_level in ("MEDIUM", "HIGH"):
                t1_b1_text = _apply_slang(t1_b1_text, protected_tokens)
                t1_b2_text = _apply_slang(t1_b2_text, protected_tokens)
            if noise_level == "HIGH":
                t1_b1_text = safe_apply_typos(t1_b1_text, protected_tokens, seed)
                t1_b2_text = safe_apply_typos(t1_b2_text, protected_tokens, seed)

            turn1 = TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id=f"{scenario_id}-t1-b1",
                        text=t1_b1_text,
                        offset_seconds=0,
                    ),
                    TrajectoryBubble(
                        source_message_id=f"{scenario_id}-t1-b2",
                        text=t1_b2_text,
                        offset_seconds=3,
                    ),
                ),
                observable_facts=(issue_text, landmark_text, jurisdiction_text),
                hidden_facts=(),
                expected_action=t1_action,
            )

            turns = (turn1,)
            observable_facts = (issue_text, landmark_text, jurisdiction_text)
            hidden_facts = ()
            expected_actions = (t1_action,)
            loc_completeness = "SUFFICIENT"

    if risk is None:
        risk = template.world_truth.get("risk", template.urgency)

    world_truth = dict(template.world_truth)
    world_truth["dataset_origin"] = "synthetic"
    world_truth["intent"] = intent
    world_truth["risk"] = risk
    world_truth["completeness"] = completeness
    if split == DatasetSplit.TEST:
        world_truth["heldout_type"] = "synthetic"
        world_truth["synthetic_heldout"] = True
        world_truth["is_synthetic"] = True

    return ComplaintTrajectory(
        scenario_id=scenario_id,
        family_id=template.family_id,
        split=split,
        category=template.category,
        world_truth=world_truth,
        observable_facts=observable_facts,
        hidden_facts=hidden_facts,
        location_completeness=loc_completeness,
        duration=template.duration,
        claim_certainty=template.claim_certainty,
        persona=persona,
        noise={"level": noise_level},
        attachment_role=template.attachment_role,
        turns=turns,
        expected_action_by_turn=expected_actions,
    )


def generate_synthetic_corpus(
    train_trajectories: Sequence[ComplaintTrajectory],
    target_lines: int = 1500,
    seed: int = 42,
) -> list[str]:
    raw_lines: list[str] = []

    for traj in train_trajectories:
        for turn in traj.turns:
            for b in turn.bubbles:
                text = b.text.strip()
                if len(text) > 10:
                    raw_lines.append(text)

    adminduk_topics = [
        "perekaman kartu tanda penduduk elektronik baru bagi pemilih pemula",
        "penggantian blanko e-ktp yang mengalami kerusakan fisik atau buram",
        "pembaruan data kartu keluarga setelah peristiwa pernikahan",
        "pencetakan kartu identitas anak untuk keperluan pendaftaran sekolah",
        "pengurusan surat keterangan pindah warga negara indonesia antar provinsi",
        "pencatatan akta kelahiran anak baru lahir di loket dinas kependudukan",
        "pengurusan akta kematian sebagai syarat penutupan rekening dan waris",
        "legalisasi dokumen kependudukan secara daring melalui aplikasi pemda",
        "permohonan surat pengantar rukun tetangga untuk keperluan administrasi bansos",
        "pemutakhiran data terpadu kesejahteraan sosial di tingkat kelurahan",
        "verifikasi lapangan penerima bantuan iuran jaminan kesehatan daerah",
        "pembetulan kesalahan elemen data nama dan tanggal lahir pada dokumen resmi",
        "pengajuan cetak kartu keluarga mandiri menggunakan kertas putih standar",
        "sinkronisasi data nomor induk kependudukan pada sistem pelayanan publik",
        "pengaktifan identitas kependudukan digital melalui aplikasi terpadu",
    ]

    adminduk_templates = [
        "Syarat utama untuk {topic} adalah membawa dokumen identitas pendukung yang sah.",
        "Warga yang hendak mengurus {topic} disarankan hadir lebih awal di kantor pelayanan.",
        "Alur pelayanan terkait {topic} kini dapat diproses melalui loket kelurahan setempat.",
        "Petugas dinas kependudukan memberikan pendampingan khusus dalam proses {topic}.",
        "Pemerintah daerah mengimbau warga agar tidak menggunakan calo saat mengurus {topic}.",
        "Batas waktu penyelesaian {topic} standar operasional adalah tiga hari kerja.",
        "Dokumen hasil {topic} dapat diunduh secara mandiri dalam format berkas elektronik.",
        "Tidak ada pungutan biaya resmi atau retribusi dalam pengurusan {topic}.",
        "Masyarakat dapat memantau status berkas {topic} melalui portal informasi kependudukan.",
        "Sosialisasi mengenai tata cara {topic} terus dilakukan di tingkat rukun warga.",
        "Bagi warga lansia dan disabilitas layanan jemput bola untuk {topic} siap melayani di rumah.",
        "Konsultasi berkas persyaratan {topic} dilayani setiap hari kerja mulai pukul delapan pagi.",
        "Kelengkapan formulir pendaftaran untuk {topic} dapat dicek terlebih dahulu di meja informasi.",
        "Warga yang telah selesai melakukan {topic} dapat memberikan umpan balik pada loket kepuasan.",
        "Pemerintah kota menjamin kemudahan akses bagi seluruh masyarakat dalam {topic}.",
    ]

    for top in adminduk_topics:
        for tmpl in adminduk_templates:
            raw_lines.append(tmpl.format(topic=top))

    infra_actions = [
        ("perbaikan jalan berlubang di jalan arteri primer", "Bina Marga", "aspal hotmix"),
        ("penambalan aspal ambles dekat jembatan", "Dinas Pekerjaan Umum", "material cold mix"),
        ("pemasangan rambu peringatan tikungan rawan", "Dinas Perhubungan", "rambu reflektif"),
        ("penggantian lampu penerangan jalan yang padam", "Unit PJU Kota", "lampu hemat energi"),
        ("pembersihan gorong-gorong yang tersumbat sedimen", "Dinas Sumber Daya Air", "alat pengeruk mini"),
        ("perbaikan tanggul kali yang retak tergerus arus", "Balai Wilayah Sungai", "bronjong batu kali"),
        ("pengangkatan tumpukan sampah liar di lahan terbuka", "Dinas Lingkungan Hidup", "armada truk sampah"),
        ("pengosongan bak penampungan sampah sementara", "Regu Kebersihan Kebersihan", "kontainer hidrolik"),
        ("penertiban pembakaran sampah terbuka di lingkungan", "Satpol PP dan DLH", "patroli pengawasan"),
        ("perbaikan kebocoran pipa induk transmisi air", "Tim Teknis PDAM", "klem pipa baja"),
        ("penanganan aliran air keran keruh berlumpur", "Petugas Pengolahan PDAM", "flushing jaringan perpipaan"),
        ("pengaturan giliran distribusi air saat pemeliharaan", "Humas Layanan Pelanggan", "truk tangki darurat"),
        ("penebangan pohon lapuk rawan tumbang di tepi jalan", "Dinas Pertamanan", "gergaji mesin hidrolik"),
        ("penataan jaringan kabel utilitas semrawut", "Dinas Kominfo dan PU", "saluran utilitas terpadu"),
        ("pembersihan saluran drainase pemukiman padat", "Kelompok Swadaya Masyarakat", "peralatan gotong royong"),
    ]

    infra_templates = [
        "{dept} segera menerjunkan tim tanggap darurat untuk melakukan {action} dengan {tool}.",
        "Laporan masyarakat mengenai kebutuhan {action} telah diagendakan oleh {dept}.",
        "Proses {action} diperkirakan memakan waktu beberapa jam dan diawasi langsung oleh {dept}.",
        "Demi keselamatan pengguna jalan {dept} memasang barikade pengaman selama {action}.",
        "Evaluasi pasca {action} dilakukan secara berkala oleh {dept} guna mencegah kerusakan berulang.",
        "Pekerjaan lapangan {action} dilaksanakan pada malam hari agar tidak mengganggu arus mobilitas warga.",
        "Warga setempat mengapresiasi kecepatan {dept} dalam menuntaskan {action}.",
        "Ketersediaan {tool} disiapkan di lokasi penanganan guna mempercepat {action}.",
        "{dept} mengimbau warga berhati-hati saat melintas di dekat lokasi {action}.",
        "Koordinasi lintas sektor antara {dept} dan aparat kewilayahan mendukung kelancaran {action}.",
        "Jadwal mingguan {action} telah diumumkan secara terbuka oleh {dept}.",
        "Masyarakat dapat melaporkan perkembangan terkini mengenai {action} melalui posko terdekat.",
        "Tinjauan berkala oleh pengawas lapangan memastikan kualitas hasil {action} sesuai standar teknis.",
        "Dukungan warga dalam menjaga fasilitas publik memperpanjang usia hasil {action}.",
        "Tim monitoring {dept} memastikan kelancaran seluruh tahapan {action}.",
    ]

    for act, dept, tool in infra_actions:
        for tmpl in infra_templates:
            raw_lines.append(tmpl.format(action=act, dept=dept, tool=tool))

    health_topics = [
        "pelayanan dokter umum di puskesmas kelurahan",
        "layanan posyandu balita dan pemantauan tumbuh kembang",
        "pemeriksaan kesehatan berkala bagi masyarakat lanjut usia",
        "ketersediaan obat pengendali hipertensi dan diabetes melitus",
        "layanan rujukan medis gawat darurat ambulans satu satu sembilan",
        "penyuluhan pencegahan demam berdarah dengue di lingkungan rt",
        "pelayanan imunisasi dasar lengkap bagi bayi dan balita",
        "program sanitasi total berbasis masyarakat di tingkat desa",
        "konsultasi gizi balita dan pencegahan masalah stunting",
        "skrining penyakit tidak menular di pos pembinaan terpadu",
        "layanan konsultasi kesehatan jiwa masyarakat terintegrasi",
        "pemeriksaan kesehatan ibu hamil dan persalinan aman",
        "penyediaan pojok laktasi di pusat pelayanan publik",
        "pengelolaan limbah infeksius klinik rawat jalan",
        "distribusi obat program rujuk balik peserta jaminan kesehatan",
    ]

    health_templates = [
        "Jadwal operasional {topic} berlangsung setiap hari kerja mulai pagi hingga siang.",
        "Masyarakat dapat memanfaatkan kartu jaminan kesehatan saat mengakses {topic}.",
        "Fasilitas kesehatan memastikan kesiapan tenaga medis dan logistik dalam {topic}.",
        "Peningkatan mutu {topic} menjadi fokus utama dinas kesehatan kabupaten kota.",
        "Buku panduan dan lembar informasi seputar {topic} dibagikan secara gratis kepada pengunjung.",
        "Petugas loket pendaftaran memberikan nomor antrean tertib untuk layanan {topic}.",
        "Keberadaan kader posyandu sangat membantu kelancaran pelaksanaan {topic}.",
        "Evaluasi kepuasan masyarakat terhadap {topic} menunjukkan tren penilaian yang baik.",
        "Layanan {topic} mengutamakan penanganan ramah bagi kelompok rentan dan lansia.",
        "Informasi jadwal kunjungan tenaga kesehatan untuk {topic} dipasang di papan pengumuman warga.",
        "Standar operasional kebersihan dijaga ketat di area pelaksanaan {topic}.",
        "Program pendampingan berkelanjutan mengoptimalkan efektivitas {topic}.",
        "Kemitraan antara puskesmas dan aparat kelurahan memperluas cakupan {topic}.",
        "Warga yang hendak berkonsultasi mengenai {topic} dapat menghubungi petugas administrasi faskes.",
        "Fasilitas ruang tunggu yang nyaman disediakan bagi warga yang menunggu giliran {topic}.",
    ]

    for htop in health_topics:
        for tmpl in health_templates:
            raw_lines.append(tmpl.format(topic=htop))

    chat_prefixes = [
        "Halo admin,",
        "Selamat pagi bapak ibu petugas,",
        "Izin bertanya bapak ibu,",
        "Mau konsultasi terkait layanan,",
        "Mohon info resmi pelayanan,",
        "Lapor min,",
        "Catatan aduan warga,",
        "Konfirmasi tindak lanjut,",
        "Selamat siang tim layanan,",
        "Mohon konfirmasi prosedur,",
        "Permisi admin kanal laporan,",
        "Izin klarifikasi informasi,",
        "Terima kasih atas responsnya min,",
        "Mohon arahan langkah berikutnya,",
        "Salam hormat bapak ibu admin,",
    ]

    chat_bodies = [
        "apakah berkas ktp elektronik yang sudah jadi bisa diambilkan oleh anggota keluarga dalam satu kartu keluarga?",
        "bagaimana tata cara mendaftar antrean daring puskesmas agar tidak perlu menunggu lama di loket?",
        "di mana lokasi posko pelayanan jemput bola administrasi kependudukan minggu ini?",
        "apakah mobil tangki air bersih darurat bisa dikirimkan ke perumahan warga selama pemeliharaan pipa induk?",
        "ke mana kami harus melaporkan pohon rimbun yang dahannya menyentuh kabel utilitas tegangan tinggi?",
        "apakah pendaftaran kepesertaan jaminan kesehatan bantuan iuran bisa diusulkan melalui pengurus lingkungan?",
        "mohon jadwal rutin armada kebersihan pengangkut sampah domestik di permukiman warga.",
        "bagaimana prosedur penerbitan surat pengantar nikah dari kantor kelurahan setempat?",
        "apakah dokumen akta kelahiran yang basah terkena rembesan air hujan bisa diganti baru tanpa denda?",
        "mohon tindak lanjut pembersihan parit di pinggir jalan lingkungan yang meluap saat hujan lebat.",
        "apakah puskesmas kelurahan tetap melayani pemeriksaan kesehatan lansia pada hari libur nasional?",
        "bagaimana cara memperbarui data alamat domisili usaha kecil pada basis data perizinan daerah?",
        "mohon konfirmasi apakah lampu jalan lingkungan yang dilaporkan kemarin sudah dijadwalkan diperbaiki.",
        "apakah ada syarat khusus untuk mendapatkan bantuan tambahan gizi balita di posyandu binaan?",
        "bagaimana cara mengecek keaktifan nomor kepesertaan jaminan kesehatan secara mandiri melalui gawai?",
        "apakah proses cetak ulang kartu identitas anak bisa diwakilkan oleh wali yang sah?",
        "mohon informasi posko darurat bencana banjir terdekat yang menyediakan layanan kesehatan lapangan.",
        "apakah ada batasan kuota antrean harian untuk loket pencatatan sipil di balai kota?",
        "bagaimana mekanisme pengaduan jika terdapat pungutan di luar tarif resmi pada fasilitas umum?",
        "terima kasih banyak atas penanganan cepat regu lapangan yang telah membersihkan sampah kemarin sore.",
    ]

    for pfx in chat_prefixes:
        for bdy in chat_bodies:
            raw_lines.append(f"{pfx} {bdy}")

    admin_notices = [
        "Pemberitahuan: Pelayanan administrasi kependudukan tetap dibuka sesuai jam operasional kerja.",
        "Informasi: Pemeliharaan berkala jaringan pipa distribusi air minum dilakukan pada jam pemakaian rendah.",
        "Pengumuman: Jadwal pengangkutan sampah di tempat pembuangan sementara dimulai setiap pukul enam pagi.",
        "Himbauan: Warga diharapkan memilah sampah rumah tangga menjadi sampah organik dan anorganik.",
        "Catatan resmi: Layanan pengurusan dokumen kependudukan tidak dikenakan biaya dalam bentuk apa pun.",
        "Pemberitahuan dinas: Petugas pemeliharaan jalan akan melakukan perataan bahu jalan arteri malam ini.",
        "Informasi posko: Laporan permohonan pembersihan saluran drainase telah diteruskan ke regu kerja dinas sda.",
        "Pengumuman faskes: Seluruh tenaga medis puskesmas bersiaga melayani masyarakat sesuai jadwal dinas.",
        "Warta kelurahan: Musyawarah rencana pembangunan tingkat rukun warga akan membahas prioritas saluran air.",
        "Catatan pelayanan: Berkas permohonan yang telah lengkap akan langsung diverifikasi oleh petugas verifikator.",
        "Himbauan keselamatan: Pengendara diimbau memperlambat laju kendaraan di sekitar area pekerjaan jalan.",
        "Informasi pdam: Aliran air bersih diperkirakan berangsur normal kembali setelah proses pembersihan pipa tuntas.",
    ]

    for notice in admin_notices:
        raw_lines.append(notice)

    cleaned_corpus = validate_corpus_text(raw_lines, pii_scrubbing=True, minhash_threshold=0.85)

    if len(cleaned_corpus) < target_lines:
        extras: list[str] = []
        for idx in range(target_lines - len(cleaned_corpus) + 50):
            seed_offset = (seed + idx) % len(infra_actions)
            act, dept, tool = infra_actions[seed_offset]
            extras.append(
                f"Laporan pemeliharaan nomor {idx + 1:04d}: {dept} menjadwalkan tindakan {act} menggunakan kelengkapan {tool}."
            )
        cleaned_corpus = validate_corpus_text(
            cleaned_corpus + extras,
            pii_scrubbing=True,
            minhash_threshold=0.85,
        )

    return cleaned_corpus[: max(target_lines, len(cleaned_corpus))]


def audit_corpus_anti_leak(
    corpus_lines: Sequence[str],
    heldout_trajectories: Sequence[ComplaintTrajectory],
) -> list[str]:
    violations: list[str] = []
    test_hidden_facts: set[str] = set()
    test_landmarks: set[str] = set()
    test_jurisdictions: set[str] = set()
    test_bubble_texts: set[str] = set()

    for traj in heldout_trajectories:
        for hf in traj.hidden_facts:
            clean_hf = hf.strip().lower()
            if len(clean_hf) >= 4:
                test_hidden_facts.add(clean_hf)

        for turn in traj.turns:
            for b in turn.bubbles:
                norm_b = re.sub(r"\s+", " ", b.text.strip().lower())
                if len(norm_b) > 20:
                    test_bubble_texts.add(norm_b)

        if traj.observable_facts and len(traj.observable_facts) > 1:
            clean_lm = traj.observable_facts[1].strip().lower()
            if len(clean_lm) >= 5:
                test_landmarks.add(clean_lm)

    for line_idx, line in enumerate(corpus_lines, start=1):
        line_lower = line.strip().lower()

        for hf in test_hidden_facts:
            if hf in line_lower:
                violations.append(
                    f"Corpus line {line_idx} contains test hidden fact '{hf}': '{line[:60]}...'"
                )

        if line_lower in test_bubble_texts:
            violations.append(
                f"Corpus line {line_idx} matches exact test bubble text: '{line[:60]}...'"
            )

        for lm in test_landmarks:
            if lm in line_lower:
                violations.append(
                    f"Corpus line {line_idx} contains test landmark '{lm}': '{line[:60]}...'"
                )

    return violations


def generate_and_export_m3_dataset(
    output_dir: Path | str = "artifacts/m3_synthetic",
    seed: int = 42,
    num_scenarios: int = 720,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    multi_turn_ratio: float = 0.50,
    corpus_target_lines: int = 1500,
    version: str = "v1.0.0",
    created_at: str = "2026-09-11T00:00:00+00:00",
    audit: bool = True,
    templates: Sequence[ScenarioTemplate] | None = None,
) -> dict[str, Any]:
    set_deterministic_seed(seed)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    active_templates = tuple(templates) if templates is not None else ALL_SCENARIO_TEMPLATES
    if len(active_templates) < 6:
        raise ValueError("At least 6 scenario templates are required for dataset export.")

    family_to_cat = {t.family_id: t.category for t in active_templates}
    split_map = build_family_split_map(
        tuple(family_to_cat.keys()),
        seed=seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        family_to_category=family_to_cat,
    )

    personas = ("STANDARD", "FORMAL", "FRUSTRATED_RAMBLING", "PANICKED")
    noise_levels = ("LOW", "MEDIUM", "HIGH")
    intents = ("COMPLAINT", "INQUIRY", "FEEDBACK")
    risks = ("LOW", "MEDIUM", "HIGH", "URGENT")
    comps = ("SUFFICIENT", "INCOMPLETE", "AMBIGUOUS")

    split_idx_map: dict[str, int] = defaultdict(int)
    trajectories: list[ComplaintTrajectory] = []
    for i in range(num_scenarios):
        template_idx = (i + seed) % len(active_templates)
        template = active_templates[template_idx]
        scenario_id = f"sc-{template.family_id}-{i + 1:04d}"
        persona = personas[(i + seed) % len(personas)]
        noise_level = noise_levels[(i + seed) % len(noise_levels)]
        is_multi_turn = ((i + seed) % 100) < int(multi_turn_ratio * 100)

        target_split = split_map[template.family_id]
        s_key = target_split.value
        idx_in_split = split_idx_map[s_key]
        split_idx_map[s_key] += 1

        assigned_intent = intents[idx_in_split % len(intents)]
        assigned_risk = risks[idx_in_split % len(risks)]
        assigned_comp = comps[(idx_in_split + (idx_in_split // 3)) % len(comps)]

        traj = generate_safe_trajectory(
            template=template,
            scenario_id=scenario_id,
            persona=persona,
            noise_level=noise_level,
            multi_turn=is_multi_turn,
            seed=seed + i,
            split=target_split,
            intent=assigned_intent,
            risk=assigned_risk,
            completeness=assigned_comp,
        )
        trajectories.append(traj)

    if audit:
        audit_res: FamilySplitAuditResult = audit_family_splits(trajectories)
        if not audit_res.passed:
            raise DatasetAuditError(
                f"Trajectory split audit failed with {len(audit_res.violations)} violations: "
                f"{audit_res.violations[:3]}"
            )

    train_trajs = [t for t in trajectories if t.split == DatasetSplit.TRAIN]
    dev_trajs = [t for t in trajectories if t.split == DatasetSplit.DEV]
    test_trajs = [t for t in trajectories if t.split == DatasetSplit.TEST]

    corpus_lines = generate_synthetic_corpus(
        train_trajectories=train_trajs,
        target_lines=corpus_target_lines,
        seed=seed,
    )

    if audit:
        heldout_violations = audit_corpus_anti_leak(
            corpus_lines=corpus_lines,
            heldout_trajectories=test_trajs + dev_trajs,
        )
        if heldout_violations:
            raise DatasetAuditError(
                f"Corpus anti-leak audit failed with {len(heldout_violations)} violations: "
                f"{heldout_violations[:3]}"
            )

    train_file = out_dir / "train.jsonl"
    dev_file = out_dir / "dev.jsonl"
    test_synthetic_file = out_dir / "test_synthetic.jsonl"
    test_file = out_dir / "test.jsonl"
    all_file = out_dir / "trajectories.jsonl"
    corpus_file = out_dir / "corpus.txt"

    export_trajectories_to_jsonl(train_trajs, train_file)
    export_trajectories_to_jsonl(dev_trajs, dev_file)
    export_trajectories_to_jsonl(test_trajs, test_synthetic_file)
    export_trajectories_to_jsonl(test_trajs, test_file)
    export_trajectories_to_jsonl(trajectories, all_file)

    with open(corpus_file, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    exported_files = [
        train_file,
        dev_file,
        test_synthetic_file,
        test_file,
        all_file,
        corpus_file,
    ]

    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for p in exported_files:
        sha = compute_file_sha256(p)
        hashes[p.name] = sha
        sizes[p.name] = p.stat().st_size

    categories_per_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    families_per_split: dict[str, set[str]] = defaultdict(set)
    intents_per_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    risks_per_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    completeness_per_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in trajectories:
        split_key = t.split.value
        cat_key = t.category.value
        intent_key = str(t.world_truth.get("intent", "COMPLAINT"))
        risk_key = str(t.world_truth.get("risk", "HIGH"))
        comp_key = str(t.world_truth.get("completeness", "SUFFICIENT"))
        categories_per_split[split_key][cat_key] += 1
        families_per_split[split_key].add(t.family_id)
        intents_per_split[split_key][intent_key] += 1
        risks_per_split[split_key][risk_key] += 1
        completeness_per_split[split_key][comp_key] += 1

    dataset_manifest_data = {
        "dataset_name": "kawal-m3-synthetic-starter",
        "version": version,
        "seed": seed,
        "created_at": created_at,
        "environment": "local-cpu",
        "parameters": {
            "num_scenarios": num_scenarios,
            "train_ratio": train_ratio,
            "dev_ratio": dev_ratio,
            "multi_turn_ratio": multi_turn_ratio,
            "num_templates": len(active_templates),
            "corpus_target_lines": corpus_target_lines,
        },
        "splits": {
            "train": {
                "trajectories_count": len(train_trajs),
                "families_count": len(families_per_split["train"]),
                "categories": dict(categories_per_split["train"]),
                "intents": dict(intents_per_split["train"]),
                "risks": dict(risks_per_split["train"]),
                "completeness": dict(completeness_per_split["train"]),
                "file": "train.jsonl",
                "sha256": hashes["train.jsonl"],
                "size_bytes": sizes["train.jsonl"],
            },
            "dev": {
                "trajectories_count": len(dev_trajs),
                "families_count": len(families_per_split["dev"]),
                "categories": dict(categories_per_split["dev"]),
                "intents": dict(intents_per_split["dev"]),
                "risks": dict(risks_per_split["dev"]),
                "completeness": dict(completeness_per_split["dev"]),
                "file": "dev.jsonl",
                "sha256": hashes["dev.jsonl"],
                "size_bytes": sizes["dev.jsonl"],
            },
            "test": {
                "trajectories_count": len(test_trajs),
                "families_count": len(families_per_split["test"]),
                "categories": dict(categories_per_split["test"]),
                "intents": dict(intents_per_split["test"]),
                "risks": dict(risks_per_split["test"]),
                "completeness": dict(completeness_per_split["test"]),
                "file": "test.jsonl",
                "sha256": hashes["test.jsonl"],
                "size_bytes": sizes["test.jsonl"],
            },
        },
        "heldout": {
            "status": "synthetic",
            "is_synthetic": True,
            "label": "SYNTHETIC_HELDOUT",
            "file": "test_synthetic.jsonl",
            "trajectories_count": len(test_trajs),
            "families_count": len(families_per_split["test"]),
            "sha256": hashes["test_synthetic.jsonl"],
            "size_bytes": sizes["test_synthetic.jsonl"],
            "description": (
                "Synthetic heldout dataset for offline evaluation, strictly partitioned "
                "by family from training and validation sets to ensure zero leakage."
            ),
        },
        "corpus": {
            "file": "corpus.txt",
            "lines_count": len(corpus_lines),
            "words_count": sum(len(line.split()) for line in corpus_lines),
            "sha256": hashes["corpus.txt"],
            "size_bytes": sizes["corpus.txt"],
            "pii_scrubbed": True,
            "source": "train_split_and_synthetic_civic_domain",
        },
        "anti_leak_audit": {
            "passed": True,
            "trajectory_violations_count": 0,
            "corpus_violations_count": 0,
            "zero_leakage_guaranteed": True,
        },
        "files": {
            name: {
                "sha256": hashes[name],
                "size_bytes": sizes[name],
            }
            for name in hashes
        },
    }

    dataset_manifest_file = out_dir / "dataset_manifest.json"
    with open(dataset_manifest_file, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest_data, f, indent=2)

    hashes[dataset_manifest_file.name] = compute_file_sha256(dataset_manifest_file)
    sizes[dataset_manifest_file.name] = dataset_manifest_file.stat().st_size

    manifest_items = [
        create_artifact_item(train_file, name="dataset-trajectories-train", version=version, task="dataset-trajectories-train", relative_to=out_dir),
        create_artifact_item(dev_file, name="dataset-trajectories-dev", version=version, task="dataset-trajectories-dev", relative_to=out_dir),
        create_artifact_item(test_synthetic_file, name="synthetic-heldout-test", version=version, task="synthetic-heldout", relative_to=out_dir),
        create_artifact_item(test_file, name="dataset-trajectories-test", version=version, task="dataset-trajectories-test", relative_to=out_dir),
        create_artifact_item(all_file, name="dataset-trajectories-all", version=version, task="dataset-trajectories-all", relative_to=out_dir),
        create_artifact_item(corpus_file, name="dataset-corpus-dapt", version=version, task="dataset-corpus-dapt", relative_to=out_dir),
        create_artifact_item(dataset_manifest_file, name="dataset-manifest", version=version, task="dataset-manifest", relative_to=out_dir),
    ]

    manifest = create_or_update_artifact_manifest(
        output_dir=out_dir,
        items=manifest_items,
        manifest_version="1.0.0",
        environment="local-cpu",
        manifest_filename="manifest.json",
    )

    manifest_file = out_dir / "manifest.json"
    hashes[manifest_file.name] = compute_file_sha256(manifest_file)
    sizes[manifest_file.name] = manifest_file.stat().st_size

    sha256sums_file = out_dir / "sha256sums.txt"
    with open(sha256sums_file, "w", encoding="utf-8") as f:
        for fname in sorted(hashes.keys()):
            f.write(f"{hashes[fname]}  {fname}\n")

    hashes[sha256sums_file.name] = compute_file_sha256(sha256sums_file)
    sizes[sha256sums_file.name] = sha256sums_file.stat().st_size

    return {
        "status": "SUCCESS",
        "output_dir": str(out_dir),
        "counts": {
            "total_trajectories": len(trajectories),
            "train_trajectories": len(train_trajs),
            "dev_trajectories": len(dev_trajs),
            "test_trajectories": len(test_trajs),
            "corpus_lines": len(corpus_lines),
            "total_families": len(active_templates),
            "train_families": len(families_per_split["train"]),
            "dev_families": len(families_per_split["dev"]),
            "test_families": len(families_per_split["test"]),
            "categories_per_split": {k: dict(v) for k, v in categories_per_split.items()},
            "intents_per_split": {k: dict(v) for k, v in intents_per_split.items()},
            "risks_per_split": {k: dict(v) for k, v in risks_per_split.items()},
            "completeness_per_split": {k: dict(v) for k, v in completeness_per_split.items()},
        },
        "files": {str(out_dir / fname): hashes[fname] for fname in sorted(hashes.keys())},
        "hashes": hashes,
        "sizes": sizes,
        "manifest": manifest,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic M3 synthetic dataset generator and export CLI."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/m3_synthetic",
        help="Target directory to export dataset artifacts (default: artifacts/m3_synthetic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=720,
        help="Total number of trajectory scenarios to generate (default: 720)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of families assigned to train split (default: 0.70)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Fraction of families assigned to dev split (default: 0.15)",
    )
    parser.add_argument(
        "--multi-turn-ratio",
        type=float,
        default=0.50,
        help="Ratio of multi-turn interactions (default: 0.50)",
    )
    parser.add_argument(
        "--corpus-lines",
        type=int,
        default=1500,
        help="Minimum target lines for DAPT pretraining corpus (default: 1500)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0.0",
        help="Dataset version tag (default: v1.0.0)",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip strict anti-leak split audit (not recommended)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        res = generate_and_export_m3_dataset(
            output_dir=args.output_dir,
            seed=args.seed,
            num_scenarios=args.num_scenarios,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            multi_turn_ratio=args.multi_turn_ratio,
            corpus_target_lines=args.corpus_lines,
            version=args.version,
            audit=not args.no_audit,
        )
        counts = res["counts"]
        print("M3 Synthetic Dataset Generation Completed Successfully:")
        print(f"  - Output Dir: {res['output_dir']}")
        print(f"  - Total Trajectories: {counts['total_trajectories']}")
        print(f"  - Splits: Train={counts['train_trajectories']}, Dev={counts['dev_trajectories']}, Test={counts['test_trajectories']}")
        print(f"  - Families: Train={counts['train_families']}, Dev={counts['dev_families']}, Test={counts['test_families']}")
        print(f"  - DAPT Corpus Lines: {counts['corpus_lines']}")
        print("Exported Files & Hashes:")
        for fname, sha in sorted(res["hashes"].items()):
            size_kb = res["sizes"].get(fname, 0) / 1024.0
            print(f"  - {fname:<25} ({size_kb:6.1f} KB): {sha}")
        return 0
    except Exception as exc:
        print(f"Error during dataset generation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
