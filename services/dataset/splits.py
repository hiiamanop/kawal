from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from contracts.models import (
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
    FamilySplitAuditResult,
)


def assign_split_by_family(
    family_id: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
) -> DatasetSplit:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0.0 <= dev_ratio < 1.0):
        raise ValueError("dev_ratio must be between 0 and 1")
    if train_ratio + dev_ratio >= 1.0:
        raise ValueError("train_ratio + dev_ratio must be less than 1.0")

    token = f"{seed}:{family_id}".encode("utf-8")
    hash_int = int(hashlib.sha256(token).hexdigest(), 16)
    uniform_val = hash_int / float(1 << 256)

    if uniform_val < train_ratio:
        return DatasetSplit.TRAIN
    if uniform_val < train_ratio + dev_ratio:
        return DatasetSplit.DEV
    return DatasetSplit.TEST


def build_stratified_family_split_map(
    family_to_category: dict[str, Category | str] | Sequence[tuple[str, Category | str]],
    seed: int = 42,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
) -> dict[str, DatasetSplit]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0.0 <= dev_ratio < 1.0):
        raise ValueError("dev_ratio must be between 0 and 1")
    if train_ratio + dev_ratio >= 1.0:
        raise ValueError("train_ratio + dev_ratio must be less than 1.0")

    if isinstance(family_to_category, Sequence) and not isinstance(family_to_category, dict):
        mapping = dict(family_to_category)
    else:
        mapping = dict(family_to_category)

    cat_to_families: dict[str, list[str]] = defaultdict(list)
    for fam_id in sorted(mapping.keys()):
        raw_cat = mapping[fam_id]
        cat_key = raw_cat.value if hasattr(raw_cat, "value") else str(raw_cat)
        cat_to_families[cat_key].append(fam_id)

    test_ratio = 1.0 - train_ratio - dev_ratio
    result_map: dict[str, DatasetSplit] = {}

    for cat_key, families in sorted(cat_to_families.items()):
        def _sort_key(f: str) -> tuple[str, str]:
            token = f"{seed}:{cat_key}:{f}".encode("utf-8")
            return (hashlib.sha256(token).hexdigest(), f)

        ordered = sorted(families, key=_sort_key)
        k = len(ordered)
        if k == 0:
            continue
        elif k == 1:
            result_map[ordered[0]] = DatasetSplit.TRAIN
        elif k == 2:
            result_map[ordered[0]] = DatasetSplit.TRAIN
            result_map[ordered[1]] = DatasetSplit.TEST
        else:
            quotas = {"train": 1, "dev": 1, "test": 1}
            rem = k - 3
            if rem > 0:
                t_train = k * train_ratio
                t_dev = k * dev_ratio
                t_test = k * test_ratio
                for _ in range(rem):
                    best = max(
                        [
                            ("train", t_train - quotas["train"]),
                            ("dev", t_dev - quotas["dev"]),
                            ("test", t_test - quotas["test"]),
                        ],
                        key=lambda x: x[1],
                    )[0]
                    quotas[best] += 1

            n_train = quotas["train"]
            n_dev = quotas["dev"]

            for f in ordered[:n_train]:
                result_map[f] = DatasetSplit.TRAIN
            for f in ordered[n_train : n_train + n_dev]:
                result_map[f] = DatasetSplit.DEV
            for f in ordered[n_train + n_dev :]:
                result_map[f] = DatasetSplit.TEST

    return result_map


def partition_stratified_family_splits(
    family_to_category: dict[str, Category | str] | Sequence[tuple[str, Category | str]],
    seed: int = 42,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
) -> dict[DatasetSplit, tuple[str, ...]]:
    split_map = build_stratified_family_split_map(
        family_to_category=family_to_category,
        seed=seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
    )
    grouped: dict[DatasetSplit, list[str]] = defaultdict(list)
    for fam_id in sorted(split_map.keys()):
        grouped[split_map[fam_id]].append(fam_id)

    return {
        DatasetSplit.TRAIN: tuple(grouped[DatasetSplit.TRAIN]),
        DatasetSplit.DEV: tuple(grouped[DatasetSplit.DEV]),
        DatasetSplit.TEST: tuple(grouped[DatasetSplit.TEST]),
    }


def partition_families_by_split(
    family_ids: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    family_to_category: dict[str, Category | str] | None = None,
) -> dict[DatasetSplit, tuple[str, ...]]:
    if family_to_category is not None:
        sub_map = {f: family_to_category[f] for f in sorted(set(family_ids)) if f in family_to_category}
        if len(sub_map) == len(set(family_ids)):
            return partition_stratified_family_splits(
                sub_map,
                seed=seed,
                train_ratio=train_ratio,
                dev_ratio=dev_ratio,
            )

    unique_families = sorted(set(family_ids))
    train_families: list[str] = []
    dev_families: list[str] = []
    test_families: list[str] = []

    for family_id in unique_families:
        split = assign_split_by_family(
            family_id,
            seed=seed,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
        )
        if split == DatasetSplit.TRAIN:
            train_families.append(family_id)
        elif split == DatasetSplit.DEV:
            dev_families.append(family_id)
        else:
            test_families.append(family_id)

    return {
        DatasetSplit.TRAIN: tuple(train_families),
        DatasetSplit.DEV: tuple(dev_families),
        DatasetSplit.TEST: tuple(test_families),
    }


def build_family_split_map(
    family_ids: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    family_to_category: dict[str, Category | str] | None = None,
) -> dict[str, DatasetSplit]:
    if family_to_category is not None:
        sub_map = {f: family_to_category[f] for f in sorted(set(family_ids)) if f in family_to_category}
        if len(sub_map) == len(set(family_ids)):
            return build_stratified_family_split_map(
                sub_map,
                seed=seed,
                train_ratio=train_ratio,
                dev_ratio=dev_ratio,
            )

    return {
        family_id: assign_split_by_family(
            family_id,
            seed=seed,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
        )
        for family_id in sorted(set(family_ids))
    }


COMMON_LANGUAGE_TOKENS: frozenset[str] = frozenset(
    {
        # Grammatical particles / stop words
        "dan", "di", "ke", "dari", "yang", "yg", "ini", "itu", "ada", "bisa", "bs",
        "sudah", "udh", "belum", "blm", "tidak", "tak", "gak", "saat", "pada", "untuk",
        "utk", "dengan", "dg", "oleh", "akan", "kami", "saya", "sy", "kita", "kamu",
        "anda", "dia", "mereka", "karena", "krn", "agar", "supaya", "tetapi", "tapi",
        "namun", "atau", "seperti", "sebagai", "dalam", "luar", "atas", "bawah",
        "juga", "hanya", "saja", "lagi", "pun", "tentang", "mengenai", "terkait",
        "sehubungan", "hal", "adapun", "berikut", "yaitu", "bahwa", "bhwa",
        # Politeness & greetings
        "mohon", "mhn", "tolong", "tlg", "yth", "terima", "trm", "kasih", "ksh",
        "makasih", "mks", "halo", "selamat", "slmt", "pak", "bu", "bapak", "bpk",
        "ibu", "ib", "min", "admin", "mas", "mbak",
        # Location & administrative connectors
        "titik", "ttk", "patokan", "patokannya", "lokasi", "lks", "tempat", "tmpt",
        "posisi", "wilayah", "wil", "area", "daerah", "alamat", "berada", "brd",
        "depan", "dpn", "belakang", "blkg", "dekat", "dkt", "samping", "spg",
        "antara", "sekitar", "sktr", "persis", "seberang", "gedung", "rumah",
        "kantor", "posko", "kompleks", "perumahan", "lingkungan", "warga", "rt", "rw",
        "kelurahan", "kecamatan", "kota", "kabupaten", "provinsi", "desa", "gang",
        # Reporting / procedural
        "lapor", "lpr", "laporkan", "laporan", "bantuan", "petugas", "unit", "dinas",
        "pemda", "pemkot", "pihak", "keterangan", "ket", "informasi", "info",
         "administrasi", "adm", "administrasinya", "masyarakat", "publik",
         "jalan", "aspal", "lubang", "bolong", "got", "parit", "selokan", "air",
         "pipa", "bocor", "mati", "kotor", "keruh", "sampah", "tumpukan", "bau",
         "busuk", "lalat", "ktp", "kk", "berkas", "antre", "antrean", "calo",
         "puskesmas", "rsud", "sakit", "dokter", "poli", "obat", "pasien", "bpjs",
         "rujukan", "nakes", "macet", "lampu", "trotoar", "marka", "genangan", "banjir",
         "bener", "beneran", "bener2", "udah", "ga", "gk", "tdk", "bngt", "bgtu", "bgtan",
         "dpan", "sebrang", "sbrng", "pas", "pasca", "pasang", "waktu", "wkt", "jam",
         "jm", "kemaren", "td", "tadi", "malem", "deket", "bantuin", "prmsi", "tlong",
         "infokan", "infonya", "aduan", "keluhan", "komplain", "saran", "masukan", "tanggapan",
         "respon", "responnya", "tindaklanjuti", "solusinya", "solusiny", "tolonglah",
         "dll",
         # Urgency & descriptive connectors

        "kondisi", "knds", "keadaan", "situasi", "sangat", "bgt", "banget", "amat",
        "lebih", "paling", "sekali", "parah", "prh", "darurat", "drrt", "bahaya",
        "bhya", "mendesak", "segera", "sgr", "cepat", "cpt", "rusak", "rsk",
        "perbaikan", "penanganan", "tangani", "ditangani", "tindak", "ditindaklanjuti",
        "bantu", "dibantu", "kirim", "krm", "keterlaluan", "makin", "semakin",
        # Time & interrogative
        "hari", "hr", "kemarin", "kmrn", "sekarang", "skrg", "besok", "bsk",
        "pagi", "pg", "siang", "sg", "sore", "sr", "malam", "mlm",
        "bagaimana", "gmn", "kenapa", "knp", "mengapa",
        # Major administrative cities
        "bandung", "jakarta", "surabaya", "bogor", "depok", "bekasi", "tangerang",
        "madiun", "malang", "batam", "tuban", "tasikmalaya", "parepare", "tenggarong",
    }
)


def _extract_text_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
    stop_words = {
        "dan",
        "di",
        "ke",
        "dari",
        "yang",
        "ini",
        "itu",
        "ada",
        "bisa",
        "sudah",
        "belum",
        "tidak",
        "saat",
        "pada",
        "untuk",
        "dengan",
        "oleh",
        "akan",
        "kami",
        "saya",
        "min",
        "mohon",
        "tolong",
    }
    return {w for w in words if len(w) >= 3 and w not in stop_words}


def _extract_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _is_common_language_ngram(ngram: tuple[str, ...]) -> bool:
    return all(token in COMMON_LANGUAGE_TOKENS for token in ngram)


_CATEGORY_LABEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(category|kategori)\s*[:=]\s*[a-zA-Z_]+\b", re.IGNORECASE),
    re.compile(r"\b(drainage_flood|clean_water|civil_admin|health_service)\b", re.IGNORECASE),
    re.compile(r"\b(drainage flood|clean water|civil admin|health service)\b", re.IGNORECASE),
    re.compile(r"\b(ROAD|WASTE)\b"),
    re.compile(r"\bCategory\.[A-Z_]+\b"),
)


def _check_bubble_identifier_leakages(
    text: str,
    scenario_id: str,
    family_id: str,
    all_families: set[str],
    world_truth: dict[str, Any],
) -> list[str]:
    leaks: list[str] = []
    text_lower = text.lower()

    if len(scenario_id) >= 3:
        pattern = rf"(?<![a-zA-Z0-9_\-]){re.escape(scenario_id.lower())}(?![a-zA-Z0-9_\-])"
        if re.search(pattern, text_lower):
            leaks.append(f"scenario_id '{scenario_id}'")

    if len(family_id) >= 3:
        pattern = rf"(?<![a-zA-Z0-9_\-]){re.escape(family_id.lower())}(?![a-zA-Z0-9_\-])"
        if re.search(pattern, text_lower):
            leaks.append(f"family_id '{family_id}'")

    for known_fam in all_families:
        if known_fam != family_id and len(known_fam) >= 5:
            pattern = rf"(?<![a-zA-Z0-9_\-]){re.escape(known_fam.lower())}(?![a-zA-Z0-9_\-])"
            if re.search(pattern, text_lower):
                leaks.append(f"cross family_id '{known_fam}'")

    for pat in _CATEGORY_LABEL_PATTERNS:
        match = pat.search(text)
        if match:
            leaks.append(f"category label '{match.group(0)}'")

    for k, v in world_truth.items():
        k_str = str(k).lower()
        v_str = str(v).lower()

        if "_" in k_str or k_str in ("infra_type", "waste_type", "flood_depth_cm", "claim_certainty"):
            pattern = rf"(?<![a-zA-Z0-9_\-]){re.escape(k_str)}(?![a-zA-Z0-9_\-])"
            if re.search(pattern, text_lower):
                leaks.append(f"world truth key '{k}'")

        if isinstance(v, str) and "_" in v:
            pattern = rf"(?<![a-zA-Z0-9_\-]){re.escape(v_str)}(?![a-zA-Z0-9_\-])"
            if re.search(pattern, text_lower):
                leaks.append(f"world truth code identifier '{v}'")

        kv_pattern = rf"(?<![a-zA-Z0-9_\-]){re.escape(k_str)}\s*[:=]\s*{re.escape(v_str)}(?![a-zA-Z0-9_\-])"
        if re.search(kv_pattern, text_lower):
            leaks.append(f"world truth pair '{k}: {v}'")

    if "world_truth" in text_lower:
        leaks.append("world_truth identifier")

    return leaks


def audit_family_splits(
    trajectories: Sequence[ComplaintTrajectory | dict[str, Any]],
) -> FamilySplitAuditResult:
    split_trajectories: dict[str, int] = defaultdict(int)
    split_families: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)

    bubble_texts_by_split: dict[str, set[str]] = defaultdict(set)
    seen_ngrams_by_split: dict[int, dict[tuple[str, ...], tuple[str, str]]] = {
        6: {},
        7: {},
    }

    cross_split_duplicates: list[str] = []
    oracle_leakages: list[str] = []
    violations: list[str] = []

    all_families: set[str] = set()
    for traj in trajectories:
        if isinstance(traj, ComplaintTrajectory):
            all_families.add(traj.family_id)
        else:
            all_families.add(str(traj.get("family_id", "unknown")))

    for traj in trajectories:
        if isinstance(traj, ComplaintTrajectory):
            scenario_id = traj.scenario_id
            family_id = traj.family_id
            split_str = str(traj.split.value if hasattr(traj.split, "value") else traj.split)
            world_truth = traj.world_truth
            turns_data = traj.turns
        else:
            scenario_id = str(traj.get("scenario_id", "unknown"))
            family_id = str(traj.get("family_id", "unknown"))
            split_val = traj.get("split", "unknown")
            split_str = str(split_val.value if hasattr(split_val, "value") else split_val)
            world_truth = traj.get("world_truth", {})
            turns_data = traj.get("turns", [])

        split_trajectories[split_str] += 1
        split_families[split_str].add(family_id)
        family_splits[family_id].add(split_str)

        for turn_idx, turn in enumerate(turns_data):
            if hasattr(turn, "bubbles"):
                bubbles = turn.bubbles
                hidden_facts = turn.hidden_facts
                expected_action = turn.expected_action
            else:
                bubbles = turn.get("bubbles", [])
                hidden_facts = turn.get("hidden_facts", [])
                expected_action = turn.get("expected_action", {})

            if hasattr(expected_action, "allowed_actions"):
                actions = expected_action.allowed_actions
                missing = expected_action.missing
            else:
                actions = expected_action.get("allowed_actions", ())
                missing = expected_action.get("missing", ())

            action_names = [
                a.value if hasattr(a, "value") else str(a) for a in actions
            ]
            if DecisionMode.EXECUTE.value in action_names and len(missing) > 0:
                v = (
                    f"Scenario {scenario_id} turn {turn_idx + 1}: "
                    f"EXECUTE specified but missing fields {missing} is non-empty"
                )
                violations.append(v)
            if DecisionMode.REQUEST_CLARIFICATION.value in action_names and len(missing) == 0:
                v = (
                    f"Scenario {scenario_id} turn {turn_idx + 1}: "
                    "REQUEST_CLARIFICATION specified but missing fields is empty"
                )
                violations.append(v)

            turn_texts: list[str] = []
            for b in bubbles:
                b_text = b.text if hasattr(b, "text") else b.get("text", "")
                turn_texts.append(b_text)

                norm_b_text = re.sub(r"\s+", " ", b_text.strip().lower())
                if len(norm_b_text) > 20:
                    for other_split, texts in bubble_texts_by_split.items():
                        if other_split != split_str and norm_b_text in texts:
                            dup_msg = (
                                f"Duplicate text between split '{split_str}' and '{other_split}' "
                                f"in scenario {scenario_id}: '{norm_b_text[:60]}...'"
                            )
                            cross_split_duplicates.append(dup_msg)
                            violations.append(dup_msg)
                    bubble_texts_by_split[split_str].add(norm_b_text)

                ident_leaks = _check_bubble_identifier_leakages(
                    text=b_text,
                    scenario_id=scenario_id,
                    family_id=family_id,
                    all_families=all_families,
                    world_truth=world_truth,
                )
                for leak in ident_leaks:
                    msg = (
                        f"Identifier leakage in scenario {scenario_id} turn {turn_idx + 1}: "
                        f"{leak} leaked into reporter bubble text"
                    )
                    oracle_leakages.append(msg)
                    violations.append(msg)

                clean_b_tokens = re.findall(r"[a-zA-Z0-9_\-]+", b_text.lower())
                for n in (6, 7):
                    for ng in _extract_ngrams(clean_b_tokens, n):
                        if _is_common_language_ngram(ng):
                            continue
                        if ng in seen_ngrams_by_split[n]:
                            other_split, other_scen = seen_ngrams_by_split[n][ng]
                            if other_split != split_str:
                                ng_str = " ".join(ng)
                                leak_msg = (
                                    f"Cross-split {n}-gram leakage between split '{split_str}' and '{other_split}' "
                                    f"in scenario {scenario_id} vs {other_scen}: '{ng_str}'"
                                )
                                cross_split_duplicates.append(leak_msg)
                                violations.append(leak_msg)
                        else:
                            seen_ngrams_by_split[n][ng] = (split_str, scenario_id)

            combined_turn_text = " ".join(turn_texts).lower()
            turn_tokens = _extract_text_tokens(combined_turn_text)

            for hf in hidden_facts:
                hf_str = str(hf).strip()
                if not hf_str:
                    continue
                hf_lower = hf_str.lower()
                if hf_lower in combined_turn_text:
                    leak_msg = (
                        f"Oracle leakage in scenario {scenario_id} turn {turn_idx + 1}: "
                        f"hidden fact '{hf_str}' leaked as exact substring into bubble text"
                    )
                    oracle_leakages.append(leak_msg)
                    violations.append(leak_msg)
                    continue

                hf_tokens = _extract_text_tokens(hf_lower)
                identifier_tokens = {t for t in hf_tokens if any(c.isdigit() for c in t) or len(t) >= 6}
                leaked_identifiers = identifier_tokens & turn_tokens
                if leaked_identifiers:
                    leak_msg = (
                        f"Oracle leakage in scenario {scenario_id} turn {turn_idx + 1}: "
                        f"hidden fact identifier token(s) {leaked_identifiers} leaked into bubble text"
                    )
                    oracle_leakages.append(leak_msg)
                    violations.append(leak_msg)

    overlap_pairs: dict[str, tuple[str, ...]] = {}
    splits_list = sorted(split_families.keys())
    for i in range(len(splits_list)):
        for j in range(i + 1, len(splits_list)):
            s1 = splits_list[i]
            s2 = splits_list[j]
            overlap = split_families[s1] & split_families[s2]
            if overlap:
                pair_key = f"{s1}_vs_{s2}"
                overlap_pairs[pair_key] = tuple(sorted(overlap))
                violations.append(
                    f"Family overlap detected between splits {s1} and {s2}: {sorted(overlap)}"
                )

    passed = len(violations) == 0

    return FamilySplitAuditResult(
        passed=passed,
        total_trajectories=len(trajectories),
        total_families=len(all_families),
        split_trajectories=dict(split_trajectories),
        split_families={k: len(v) for k, v in split_families.items()},
        family_overlap=overlap_pairs,
        oracle_leakages=tuple(oracle_leakages),
        cross_split_text_duplicates=tuple(cross_split_duplicates),
        violations=tuple(violations),
    )
