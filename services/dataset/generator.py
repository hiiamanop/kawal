from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from contracts.models import (
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
    MessageInput,
    ReplayFixture,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
)
from services.dataset.splits import assign_split_by_family, build_family_split_map


@dataclass(frozen=True)
class ScenarioTemplate:
    family_id: str
    category: Category
    issue: str
    landmark: str
    jurisdiction: str
    city: str
    urgency: str = "HIGH"
    location_completeness: str = "LANDMARK_ONLY"
    duration: str = "ONE_WEEK"
    claim_certainty: str = "FIRST_HAND"
    attachment_role: str = "NONE"
    world_truth: dict[str, Any] = field(default_factory=dict)


DEFAULT_TEMPLATES: tuple[ScenarioTemplate, ...] = (
    ScenarioTemplate(
        family_id="road-pothole-arterial",
        category=Category.ROAD,
        issue="jalan berlubang parah dan aspal ambles di jalur arteri utama",
        landmark="dekat perempatan lampu merah sudiman",
        jurisdiction="Kelurahan Cibadak, Kecamatan Sukajadi",
        city="Kota Bandung",
        urgency="HIGH",
        world_truth={"category": "ROAD", "risk": "HIGH", "infra_type": "ROAD"},
    ),
    ScenarioTemplate(
        family_id="road-collapsed-culvert-bridge",
        category=Category.ROAD,
        issue="badan jalan ambrol di akses jembatan penghubung",
        landmark="sekitar seratus meter dari jembatan baru ciliwung",
        jurisdiction="Kelurahan Margasari, Kecamatan Ciomas",
        city="Kabupaten Bogor",
        urgency="URGENT",
        world_truth={"category": "ROAD", "risk": "URGENT", "infra_type": "BRIDGE_ACCESS"},
    ),
    ScenarioTemplate(
        family_id="road-broken-street-light",
        category=Category.ROAD,
        issue="penerangan jalan umum mati total dan jalan bergelombang",
        landmark="depan SPBU veteran dua puluh satu",
        jurisdiction="Kelurahan Menteng, Kecamatan Menteng",
        city="Jakarta Pusat",
        urgency="MEDIUM",
        world_truth={"category": "ROAD", "risk": "MEDIUM", "infra_type": "STREET_LIGHT"},
    ),
    ScenarioTemplate(
        family_id="drainage-blocked-culvert",
        category=Category.DRAINAGE_FLOOD,
        issue="gorong-gorong tersumbat sampah dan sedimen tanah",
        landmark="depan gedung sekolah dasar negeri empat pagi",
        jurisdiction="Kelurahan Rawasari, Kecamatan Cempaka Putih",
        city="Jakarta Pusat",
        urgency="HIGH",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "HIGH", "flood_depth_cm": 40},
    ),
    ScenarioTemplate(
        family_id="drainage-river-overflow",
        category=Category.DRAINAGE_FLOOD,
        issue="tanggul saluran air retak dan luapan air masuk permukiman",
        landmark="belakang perumahan griya indah asri",
        jurisdiction="Kelurahan Mekarwangi, Kecamatan Bojongloa",
        city="Kota Bandung",
        urgency="URGENT",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "URGENT", "flood_depth_cm": 80},
    ),
    ScenarioTemplate(
        family_id="drainage-stagnant-ditch",
        category=Category.DRAINAGE_FLOOD,
        issue="saluran drainase mampet air limbah meluber bau menyengat",
        landmark="samping ruko pasar lama utara",
        jurisdiction="Kelurahan Sukamulya, Kecamatan Semampir",
        city="Kota Surabaya",
        urgency="MEDIUM",
        world_truth={"category": "DRAINAGE_FLOOD", "risk": "MEDIUM", "flood_depth_cm": 20},
    ),
    ScenarioTemplate(
        family_id="waste-illegal-dumping",
        category=Category.WASTE,
        issue="tumpukan sampah liar menumpuk di pinggir jalan raya berbau busuk",
        landmark="lahan kosong samping gerbang tol lingkar luar",
        jurisdiction="Kelurahan Pondok Ranji, Kecamatan Ciputat Timur",
        city="Tangerang Selatan",
        urgency="HIGH",
        world_truth={"category": "WASTE", "risk": "HIGH", "waste_type": "ILLEGAL_DUMP"},
    ),
    ScenarioTemplate(
        family_id="waste-uncollected-tps",
        category=Category.WASTE,
        issue="sampah di tempat penampungan sementara lingkungan dua minggu tidak diangkut",
        landmark="dekat pos ronda rukun warga lima",
        jurisdiction="Kelurahan Harapan Baru, Kecamatan Bekasi Utara",
        city="Kota Bekasi",
        urgency="HIGH",
        world_truth={"category": "WASTE", "risk": "HIGH", "waste_type": "TPS_OVERFLOW"},
    ),
    ScenarioTemplate(
        family_id="waste-open-burning",
        category=Category.WASTE,
        issue="pembakaran sampah kabel dan plastik menimbulkan asap pekat berbahaya",
        landmark="belakang kawasan pergudangan siera",
        jurisdiction="Kelurahan Kalirungkut, Kecamatan Rungkut",
        city="Kota Surabaya",
        urgency="URGENT",
        world_truth={"category": "WASTE", "risk": "URGENT", "waste_type": "OPEN_BURNING"},
    ),
    ScenarioTemplate(
        family_id="water-pipe-burst-main",
        category=Category.CLEAN_WATER,
        issue="pipa induk air bersih pecah semburan air membanjiri aspal",
        landmark="depan gedung balai pertemuan warga utama",
        jurisdiction="Kelurahan Baktijaya, Kecamatan Sukmajaya",
        city="Kota Depok",
        urgency="URGENT",
        world_truth={"category": "CLEAN_WATER", "risk": "URGENT", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-turbid-brown-tap",
        category=Category.CLEAN_WATER,
        issue="air keran mengalir keruh cokelat berlumpur dan berbau karat",
        landmark="kompleks perumahan blok b selatan",
        jurisdiction="Kelurahan Sukapura, Kecamatan Kiaracondong",
        city="Kota Bandung",
        urgency="HIGH",
        world_truth={"category": "CLEAN_WATER", "risk": "HIGH", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="water-outage-unannounced",
        category=Category.CLEAN_WATER,
        issue="pasokan air bersih terhenti total tiga hari tanpa pemberitahuan",
        landmark="sekitar masjid agung darussalam",
        jurisdiction="Kelurahan Panji, Kecamatan Tenggarong",
        city="Kabupaten Kutai Kartanegara",
        urgency="HIGH",
        world_truth={"category": "CLEAN_WATER", "risk": "HIGH", "service": "PDAM"},
    ),
    ScenarioTemplate(
        family_id="admin-ktp-blanko-delay",
        category=Category.CIVIL_ADMIN,
        issue="pencetakan blanko kartu tanda penduduk tertunda tiga bulan",
        landmark="gedung loket pelayanan terpadu warga",
        jurisdiction="Kelurahan Baloi Permai, Kecamatan Batam Kota",
        city="Kota Batam",
        urgency="MEDIUM",
        world_truth={"category": "CIVIL_ADMIN", "risk": "MEDIUM", "document": "KTP"},
    ),
    ScenarioTemplate(
        family_id="admin-illegal-levy-officer",
        category=Category.CIVIL_ADMIN,
        issue="pungutan liar tidak resmi dalam pengurusan surat pindah domisili",
        landmark="loket informasi publik lantai dasar",
        jurisdiction="Kelurahan Sidorejo, Kecamatan Tuban",
        city="Kabupaten Tuban",
        urgency="HIGH",
        world_truth={"category": "CIVIL_ADMIN", "risk": "HIGH", "document": "SURAT_PINDAH"},
    ),
    ScenarioTemplate(
        family_id="admin-online-queue-crash",
        category=Category.CIVIL_ADMIN,
        issue="sistem antrean elektronik pelayanan kependudukan eror antrean membludak",
        landmark="gedung sentra mal pelayanan publik",
        jurisdiction="Kelurahan Kejuron, Kecamatan Taman",
        city="Kota Madiun",
        urgency="HIGH",
        world_truth={"category": "CIVIL_ADMIN", "risk": "HIGH", "document": "ANTREAN_ONLINE"},
    ),
    ScenarioTemplate(
        family_id="health-doctor-absent-puskesmas",
        category=Category.HEALTH_SERVICE,
        issue="dokter jaga tidak berada di puskesmas saat jam operasional",
        landmark="klinik rawat jalan melati putih",
        jurisdiction="Kelurahan Karanganyar, Kecamatan Kawalu",
        city="Kota Tasikmalaya",
        urgency="HIGH",
        world_truth={"category": "HEALTH_SERVICE", "risk": "HIGH", "facility": "PUSKESMAS"},
    ),
    ScenarioTemplate(
        family_id="health-ambulance-unresponsive",
        category=Category.HEALTH_SERVICE,
        issue="telepon ambulans gawat darurat puskesmas tidak diangkat saat pasien kritis",
        landmark="instalasi rawat inap terpadu merbabu",
        jurisdiction="Kelurahan Tlogomas, Kecamatan Lowokwaru",
        city="Kota Malang",
        urgency="URGENT",
        world_truth={"category": "HEALTH_SERVICE", "risk": "URGENT", "facility": "AMBULANCE"},
    ),
    ScenarioTemplate(
        family_id="health-essential-medicine-empty",
        category=Category.HEALTH_SERVICE,
        issue="stok obat rutin diabetes dan darah tinggi di fasilitas kesehatan habis",
        landmark="depo farmasi layanan masyarakat cemara",
        jurisdiction="Kelurahan Labukkang, Kecamatan Ujung",
        city="Kota Parepare",
        urgency="HIGH",
        world_truth={"category": "HEALTH_SERVICE", "risk": "HIGH", "facility": "FARMASI"},
    ),
)


SLANG_DICTIONARY: dict[str, str] = {
    "yang": "yg",
    "sudah": "udh",
    "tidak": "gak",
    "dengan": "dg",
    "banget": "bgt",
    "tolong": "tlg",
    "mohon": "mhn",
    "untuk": "utk",
    "bisa": "bs",
    "karena": "krn",
    "rusak": "rsk",
    "jalan": "jln",
    "kemarin": "kmrn",
    "sekarang": "skrg",
    "bagaimana": "gmn",
}


def _apply_slang(text: str, protected_tokens: set[str]) -> str:
    words = text.split()
    transformed: list[str] = []
    for w in words:
        clean_w = w.lower().strip(".,!?:;")
        if clean_w in protected_tokens:
            transformed.append(w)
        elif clean_w in SLANG_DICTIONARY:
            replacement = SLANG_DICTIONARY[clean_w]
            if w.endswith((".", ",", "!", "?")):
                transformed.append(f"{replacement}{w[-1]}")
            else:
                transformed.append(replacement)
        else:
            transformed.append(w)
    return " ".join(transformed)


def _apply_typos(text: str, protected_tokens: set[str], seed: int) -> str:
    words = text.split()
    transformed: list[str] = []
    mod = seed % 7
    for idx, w in enumerate(words):
        clean_w = w.lower().strip(".,!?:;")
        if clean_w in protected_tokens or len(clean_w) < 5 or (idx % 7 != mod):
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


class TrajectoryGenerator:
    def __init__(
        self,
        templates: Sequence[ScenarioTemplate] | None = None,
    ) -> None:
        self._templates = tuple(templates) if templates is not None else DEFAULT_TEMPLATES
        self._family_map: dict[str, ScenarioTemplate] = {
            t.family_id: t for t in self._templates
        }

    def generate_trajectory(
        self,
        scenario_id: str,
        family_id: str | None = None,
        category: Category | None = None,
        persona: str = "STANDARD",
        noise_level: str = "LOW",
        multi_turn: bool = True,
        seed: int = 42,
        split_seed: int = 42,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        split_map: dict[str, DatasetSplit] | None = None,
    ) -> ComplaintTrajectory:
        selected_template: ScenarioTemplate
        if family_id is not None and family_id in self._family_map:
            selected_template = self._family_map[family_id]
        elif category is not None:
            matching = [t for t in self._templates if t.category == category]
            if not matching:
                raise ValueError(f"No templates found for category {category}")
            selected_template = matching[seed % len(matching)]
        else:
            selected_template = self._templates[seed % len(self._templates)]

        assigned_family_id = selected_template.family_id
        assigned_category = selected_template.category

        if split_map is not None and assigned_family_id in split_map:
            assigned_split = split_map[assigned_family_id]
        else:
            family_to_cat = {t.family_id: t.category for t in self._templates}
            stratified_map = build_family_split_map(
                tuple(family_to_cat.keys()),
                seed=split_seed,
                train_ratio=train_ratio,
                dev_ratio=dev_ratio,
                family_to_category=family_to_cat,
            )
            assigned_split = stratified_map.get(
                assigned_family_id,
                assign_split_by_family(
                    assigned_family_id,
                    seed=split_seed,
                    train_ratio=train_ratio,
                    dev_ratio=dev_ratio,
                ),
            )

        landmark_text = selected_template.landmark
        issue_text = selected_template.issue
        jurisdiction_text = selected_template.jurisdiction

        protected_tokens = {
            w.lower().strip(".,!?:;")
            for s in [landmark_text, jurisdiction_text, selected_template.city]
            for w in s.split()
        }

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

            t1_b1_text = f"Lapor: {issue_text}"
            t1_b2_text = f"Titik patokannya di {landmark_text}, mohon bantuan unit terkait"

            if persona == "FORMAL":
                t1_b1_text = f"Yth petugas kami laporkan: {issue_text}"
                t1_b2_text = f"Keterangan patokan lokasi yaitu di {landmark_text}"
            elif persona == "FRUSTRATED_RAMBLING":
                t1_b1_text = f"Laporan sangat mendesak: {issue_text}"
                t1_b2_text = f"Kondisi parah patokan di {landmark_text} segera ditangani"
            elif persona == "PANICKED":
                t1_b1_text = f"Darurat berbahaya: {issue_text}"
                t1_b2_text = f"Lokasi patokan persis di {landmark_text} tolong cepat"

            if noise_level in ("MEDIUM", "HIGH"):
                t1_b1_text = _apply_slang(t1_b1_text, protected_tokens)
                t1_b2_text = _apply_slang(t1_b2_text, protected_tokens)
            if noise_level == "HIGH":
                t1_b1_text = _apply_typos(t1_b1_text, protected_tokens, seed)
                t1_b2_text = _apply_typos(t1_b2_text, protected_tokens, seed + 1)

            t2_text = f"Untuk wilayah administrasinya berada di {jurisdiction_text}, {selected_template.city}"

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
            loc_completeness = "LANDMARK_ONLY"
        else:
            t1_action = TurnExpectedAction(
                turn=1,
                allowed_actions=(DecisionMode.EXECUTE,),
                missing=(),
                strategy="CREATE_TICKET",
            )
            t1_b1_text = f"Laporan: {issue_text}"
            t1_b2_text = (
                f"Lokasi berada di {landmark_text}, wilayah {jurisdiction_text}, {selected_template.city}"
            )

            if persona == "FORMAL":
                t1_b1_text = f"Yth petugas kami laporkan: {issue_text}"
                t1_b2_text = f"Keterangan patokan lokasi yaitu di {landmark_text}, wilayah {jurisdiction_text}, {selected_template.city}"
            elif persona == "FRUSTRATED_RAMBLING":
                t1_b1_text = f"Laporan sangat mendesak: {issue_text}"
                t1_b2_text = f"Kondisi parah patokan di {landmark_text}, wilayah {jurisdiction_text}, {selected_template.city} segera ditangani"
            elif persona == "PANICKED":
                t1_b1_text = f"Darurat berbahaya: {issue_text}"
                t1_b2_text = f"Lokasi patokan persis di {landmark_text}, wilayah {jurisdiction_text}, {selected_template.city} tolong cepat"

            if noise_level in ("MEDIUM", "HIGH"):
                t1_b1_text = _apply_slang(t1_b1_text, protected_tokens)
                t1_b2_text = _apply_slang(t1_b2_text, protected_tokens)
            if noise_level == "HIGH":
                t1_b1_text = _apply_typos(t1_b1_text, protected_tokens, seed)
                t1_b2_text = _apply_typos(t1_b2_text, protected_tokens, seed + 1)

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
            loc_completeness = "COMPLETE"

        return ComplaintTrajectory(
            scenario_id=scenario_id,
            family_id=assigned_family_id,
            split=assigned_split,
            category=assigned_category,
            world_truth=dict(selected_template.world_truth),
            observable_facts=observable_facts,
            hidden_facts=hidden_facts,
            location_completeness=loc_completeness,
            duration=selected_template.duration,
            claim_certainty=selected_template.claim_certainty,
            persona=persona,
            noise={"level": noise_level},
            attachment_role=selected_template.attachment_role,
            turns=turns,
            expected_action_by_turn=expected_actions,
        )

    def generate_dataset(
        self,
        num_scenarios: int = 36,
        seed: int = 42,
        categories: Sequence[Category] | None = None,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        multi_turn_ratio: float = 0.5,
    ) -> list[ComplaintTrajectory]:
        active_templates = self._templates
        if categories is not None:
            active_templates = tuple(t for t in self._templates if t.category in categories)
            if not active_templates:
                raise ValueError("No templates match the requested categories")

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

        trajectories: list[ComplaintTrajectory] = []
        for i in range(num_scenarios):
            template_idx = (i + seed) % len(active_templates)
            template = active_templates[template_idx]
            scenario_id = f"sc-{template.family_id}-{i + 1:04d}"
            persona = personas[(i + seed) % len(personas)]
            noise_level = noise_levels[(i + seed) % len(noise_levels)]
            is_multi_turn = ((i + seed) % 100) < int(multi_turn_ratio * 100)

            traj = self.generate_trajectory(
                scenario_id=scenario_id,
                family_id=template.family_id,
                persona=persona,
                noise_level=noise_level,
                multi_turn=is_multi_turn,
                seed=seed + i,
                split_seed=seed,
                train_ratio=train_ratio,
                dev_ratio=dev_ratio,
                split_map=split_map,
            )
            trajectories.append(traj)

        return trajectories


_DEFAULT_GENERATOR = TrajectoryGenerator()


def generate_trajectory(
    scenario_id: str,
    family_id: str | None = None,
    category: Category | None = None,
    persona: str = "STANDARD",
    noise_level: str = "LOW",
    multi_turn: bool = True,
    seed: int = 42,
    split_seed: int = 42,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    split_map: dict[str, DatasetSplit] | None = None,
) -> ComplaintTrajectory:
    return _DEFAULT_GENERATOR.generate_trajectory(
        scenario_id=scenario_id,
        family_id=family_id,
        category=category,
        persona=persona,
        noise_level=noise_level,
        multi_turn=multi_turn,
        seed=seed,
        split_seed=split_seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        split_map=split_map,
    )


def generate_dataset(
    num_scenarios: int = 36,
    seed: int = 42,
    categories: Sequence[Category] | None = None,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    multi_turn_ratio: float = 0.5,
) -> list[ComplaintTrajectory]:
    return _DEFAULT_GENERATOR.generate_dataset(
        num_scenarios=num_scenarios,
        seed=seed,
        categories=categories,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        multi_turn_ratio=multi_turn_ratio,
    )


def trajectory_to_replay_fixture(
    trajectory: ComplaintTrajectory,
    turn_index: int = 0,
    tenant_id: str = "research",
    conversation_id: str | None = None,
) -> ReplayFixture:
    if not (0 <= turn_index < len(trajectory.turns)):
        raise IndexError(f"turn_index {turn_index} out of range (total turns: {len(trajectory.turns)})")

    turn = trajectory.turns[turn_index]
    conv_id = conversation_id or f"conv-{trajectory.scenario_id}-t{turn.turn}"

    bubbles: list[MessageInput] = []
    for b in turn.bubbles[:3]:
        bubbles.append(
            MessageInput(
                source_message_id=b.source_message_id,
                text=b.text,
                offset_seconds=b.offset_seconds,
            )
        )

    return ReplayFixture(
        scenario_id=trajectory.scenario_id,
        tenant_id=tenant_id,
        conversation_id=conv_id,
        bubbles=tuple(bubbles),
    )


def export_trajectories_to_jsonl(
    trajectories: Sequence[ComplaintTrajectory | dict[str, Any]],
    path: Path | str,
) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        for item in trajectories:
            if isinstance(item, ComplaintTrajectory):
                f.write(item.model_dump_json() + "\n")
            else:
                f.write(json.dumps(item) + "\n")


def load_trajectories_from_jsonl(path: Path | str) -> list[ComplaintTrajectory]:
    source_path = Path(path)
    trajectories: list[ComplaintTrajectory] = []
    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                trajectories.append(ComplaintTrajectory.model_validate_json(line_str))
    return trajectories
