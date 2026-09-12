from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import pytest

from contracts.models import (
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
)
from scripts.train_dapt import EMAIL_REGEX, NIK_REGEX, PHONE_REGEX
from scripts.train_ner import (
    ADMIN_LOC_PATTERNS,
    COMPACT_OBJECT_PATTERNS,
    DAMAGE_PATTERNS,
    TIME_PATTERNS,
)
from services.dataset.generator import load_trajectories_from_jsonl
from services.dataset.splits import (
    _extract_ngrams,
    _is_common_language_ngram,
)
from services.ml.manifest import ArtifactManifest, compute_file_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_m3_independent_holdout.py"
SOURCE_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "m3_synthetic"


def _load_generator_module() -> Any:
    if not GENERATOR_SCRIPT_PATH.is_file():
        pytest.skip(f"Blocker: Generator file '{GENERATOR_SCRIPT_PATH}' is absent.")

    spec = importlib.util.spec_from_file_location(
        "generate_m3_independent_holdout",
        GENERATOR_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {GENERATOR_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_m3_independent_holdout"] = module
    spec.loader.exec_module(module)
    return module


def _invoke_generator(
    module: Any,
    output_dir: Path,
    source_dir: Path = SOURCE_ARTIFACTS_DIR,
    seed: int = 42,
    count: int = 300,
) -> Any:
    candidate_func_names = (
        "generate_independent_holdout",
        "generate_m3_independent_holdout",
        "generate_and_export_m3_independent_holdout",
        "generate_and_export_independent_holdout",
        "generate_holdout_dataset",
        "generate_holdout",
        "generate_dataset",
        "run",
    )
    gen_func = None
    for name in candidate_func_names:
        fn = getattr(module, name, None)
        if callable(fn):
            gen_func = fn
            break

    if gen_func is not None:
        kwarg_attempts = (
            {"output_dir": output_dir, "source_dir": source_dir, "seed": seed, "count": count},
            {"output_dir": output_dir, "source_dir": source_dir, "seed": seed},
            {"output_dir": output_dir, "source_artifacts_dir": source_dir, "seed": seed, "count": count},
            {"output_dir": output_dir, "source_artifacts_dir": source_dir, "seed": seed},
            {"output_dir": output_dir, "source_dir": source_dir},
            {"output_dir": output_dir, "seed": seed, "count": count},
            {"output_dir": output_dir, "seed": seed},
            {"output_dir": output_dir},
        )
        for kwargs in kwarg_attempts:
            try:
                return gen_func(**kwargs)
            except TypeError:
                continue
        try:
            return gen_func(output_dir, source_dir)
        except TypeError:
            try:
                return gen_func(output_dir)
            except TypeError:
                pass

    if hasattr(module, "main") and callable(module.main):
        try:
            return module.main([
                "--output-dir", str(output_dir),
                "--source-dir", str(source_dir),
                "--seed", str(seed),
                "--count", str(count),
            ])
        except (TypeError, SystemExit):
            try:
                return module.main()
            except Exception:
                pass

    raise RuntimeError(f"Unable to invoke holdout generator in module {module}.")


def _load_holdout_trajectories(output_dir: Path) -> list[ComplaintTrajectory]:
    preferred_names = (
        "test_independent_holdout.jsonl",
        "independent_holdout.jsonl",
        "holdout.jsonl",
        "test_synthetic.jsonl",
        "test.jsonl",
        "trajectories.jsonl",
    )
    for name in preferred_names:
        p = output_dir / name
        if p.is_file():
            return load_trajectories_from_jsonl(p)

    jsonl_files = sorted(output_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL trajectory files found in {output_dir}")
    return load_trajectories_from_jsonl(jsonl_files[0])


def _assert_trajectory_count(trajectories: Sequence[ComplaintTrajectory]) -> None:
    count = len(trajectories)
    assert count >= 300, f"Expected >= 300 holdout trajectories, got {count}"


def _assert_metadata_provenance(
    trajectories: Sequence[ComplaintTrajectory],
    output_dir: Path,
) -> None:
    assert len(trajectories) > 0, "Trajectories collection is empty"

    for idx, t in enumerate(trajectories):
        assert t.scenario_id and isinstance(t.scenario_id, str), (
            f"Trajectory {idx} missing scenario_id"
        )
        assert t.family_id and isinstance(t.family_id, str), (
            f"Trajectory {idx} missing family_id"
        )
        assert isinstance(t.category, Category), (
            f"Trajectory {idx} has invalid category {t.category}"
        )
        assert t.split in (DatasetSplit.TEST, "test", DatasetSplit.DEV, "dev", DatasetSplit.TRAIN, "train"), (
            f"Trajectory {idx} has invalid split {t.split}"
        )
        assert isinstance(t.world_truth, dict), (
            f"Trajectory {idx} world_truth is not a dict"
        )
        has_provenance = any(
            key in t.world_truth
            for key in (
                "provenance",
                "dataset_origin",
                "heldout_type",
                "synthetic_heldout",
                "is_synthetic",
                "intent",
                "risk",
                "completeness",
            )
        )
        assert has_provenance, (
            f"Trajectory {t.scenario_id} missing provenance fields in world_truth: {t.world_truth}"
        )
        assert t.world_truth.get("human_written") is not True, (
            f"Trajectory {t.scenario_id} claims human_written=True"
        )
        assert t.world_truth.get("is_human_written") is not True, (
            f"Trajectory {t.scenario_id} claims is_human_written=True"
        )
        assert len(t.turns) >= 1, f"Trajectory {t.scenario_id} has no turns"
        for turn_idx, turn in enumerate(t.turns):
            assert isinstance(turn, TrajectoryTurn), (
                f"Trajectory {t.scenario_id} turn {turn_idx + 1} is not a TrajectoryTurn"
            )
            assert len(turn.bubbles) >= 1, (
                f"Trajectory {t.scenario_id} turn {turn_idx + 1} has no bubbles"
            )
            for bubble_idx, b in enumerate(turn.bubbles):
                assert isinstance(b, TrajectoryBubble), (
                    f"Trajectory {t.scenario_id} turn {turn_idx + 1} bubble {bubble_idx + 1} is not a TrajectoryBubble"
                )
                assert isinstance(b.text, str) and len(b.text.strip()) > 0, (
                    f"Trajectory {t.scenario_id} turn {turn_idx + 1} bubble {bubble_idx + 1} has empty text"
                )
            assert turn.expected_action is not None, (
                f"Trajectory {t.scenario_id} turn {turn_idx + 1} missing expected_action"
            )

    manifest_file = output_dir / "manifest.json"
    dataset_manifest_file = output_dir / "dataset_manifest.json"
    assert manifest_file.is_file() or dataset_manifest_file.is_file(), (
        f"Missing manifest file in {output_dir} (expected manifest.json or dataset_manifest.json)"
    )

    metadata_file = output_dir / "metadata.jsonl"
    if metadata_file.is_file():
        meta_lines = [
            json.loads(line)
            for line in metadata_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(meta_lines) >= 1, f"metadata.jsonl in {output_dir} is empty"
        for meta in meta_lines:
            assert meta.get("human_written") is not True, "metadata claims human_written"
            assert meta.get("is_human_written") is not True, "metadata claims is_human_written"
            assert meta.get("provenance") in ("synthetic_independent", "synthetic", None)


def _assert_six_category_and_head_coverage(
    trajectories: Sequence[ComplaintTrajectory],
) -> None:
    categories_present = {t.category for t in trajectories}
    expected_categories = set(Category)
    assert categories_present == expected_categories, (
        f"Missing categories in holdout: {expected_categories - categories_present}"
    )

    intents = {
        str(t.world_truth.get("intent"))
        for t in trajectories
        if t.world_truth.get("intent") is not None
    }
    if intents:
        expected_intents = {"COMPLAINT", "INQUIRY", "FEEDBACK"}
        assert expected_intents.issubset(intents), (
            f"Holdout missing expected intents: {expected_intents - intents}"
        )

    risks = {
        str(t.world_truth.get("risk"))
        for t in trajectories
        if t.world_truth.get("risk") is not None
    }
    if risks:
        expected_risks = {"LOW", "MEDIUM", "HIGH", "URGENT"}
        assert expected_risks.issubset(risks), (
            f"Holdout missing expected risks: {expected_risks - risks}"
        )

    completeness = {
        str(t.world_truth.get("completeness"))
        for t in trajectories
        if t.world_truth.get("completeness") is not None
    }
    if completeness:
        expected_completeness = {"SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"}
        assert expected_completeness.issubset(completeness), (
            f"Holdout missing expected completeness: {expected_completeness - completeness}"
        )


def _assert_risk_semantic_grounding(
    trajectories: Sequence[ComplaintTrajectory],
) -> None:
    low_keywords = (
        "belum parah",
        "tidak ada bahaya",
        "tidak darurat",
        "tidak mendesak",
        "masih ringan",
        "aman terkendali",
    )
    med_growth_keywords = ("makin rusak", "semakin rusak")
    med_non_emergency_keywords = ("belum darurat", "belum sangat parah")
    high_urgency_keywords = ("bahaya", "mendesak", "segera ditangani")
    urgent_emergency_keywords = ("sangat darurat", "darurat parah", "darurat sangat mendesak")

    for t in trajectories:
        risk = t.world_truth.get("risk")
        assert risk in ("LOW", "MEDIUM", "HIGH", "URGENT"), (
            f"Scenario {t.scenario_id} has invalid risk '{risk}'"
        )
        all_text = " ".join(b.text for turn in t.turns for b in turn.bubbles)
        all_text_lower = all_text.lower()

        # Bug regression check: static generic risk clause MUST NOT exist
        assert "situasi di lokasi membutuhkan koordinasi petugas" not in all_text_lower, (
            f"Static generic risk clause found in scenario {t.scenario_id}"
        )

        if risk == "LOW":
            assert any(k in all_text_lower for k in low_keywords), (
                f"LOW risk ungrounded in scenario {t.scenario_id}: {all_text}"
            )
            assert not any(
                k in all_text_lower
                for k in ("sangat parah", "darurat sangat mendesak", "bahaya sekali", "makin rusak", "semakin rusak")
            ), f"Contradictory risk evidence in LOW scenario {t.scenario_id}: {all_text}"

        elif risk == "MEDIUM":
            assert any(k in all_text_lower for k in med_growth_keywords), (
                f"MEDIUM risk damage progression ungrounded in scenario {t.scenario_id}: {all_text}"
            )
            assert any(k in all_text_lower for k in med_non_emergency_keywords), (
                f"MEDIUM non-emergency evidence missing in scenario {t.scenario_id}: {all_text}"
            )
            assert not any(
                k in all_text_lower
                for k in ("tidak ada bahaya", "aman terkendali", "darurat sangat mendesak", "bahaya sekali")
            ), f"Contradictory risk evidence in MEDIUM scenario {t.scenario_id}: {all_text}"

        elif risk == "HIGH":
            assert "sangat parah" in all_text_lower, (
                f"HIGH risk severity ungrounded in scenario {t.scenario_id}: {all_text}"
            )
            assert any(k in all_text_lower for k in high_urgency_keywords), (
                f"HIGH risk urgency ungrounded in scenario {t.scenario_id}: {all_text}"
            )
            assert not any(
                k in all_text_lower
                for k in ("belum parah", "tidak ada bahaya", "tidak darurat", "darurat sangat mendesak", "bahaya sekali")
            ), f"Contradictory risk evidence in HIGH scenario {t.scenario_id}: {all_text}"

        elif risk == "URGENT":
            assert any(k in all_text_lower for k in urgent_emergency_keywords), (
                f"URGENT emergency wording ungrounded in scenario {t.scenario_id}: {all_text}"
            )
            assert "bahaya sekali" in all_text_lower, (
                f"URGENT severe hazard evidence ungrounded in scenario {t.scenario_id}: {all_text}"
            )
            assert "sekarang" in all_text_lower, (
                f"URGENT immediate action timing ungrounded in scenario {t.scenario_id}: {all_text}"
            )
            assert not any(
                k in all_text_lower for k in ("belum parah", "tidak mendesak", "tidak darurat")
            ), f"Contradictory risk evidence in URGENT scenario {t.scenario_id}: {all_text}"


def _assert_completeness_semantic_grounding(
    trajectories: Sequence[ComplaintTrajectory],
) -> None:
    for t in trajectories:
        comp = t.world_truth.get("completeness")
        assert comp in ("SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"), (
            f"Scenario {t.scenario_id} has invalid completeness '{comp}'"
        )
        all_text = " ".join(b.text for turn in t.turns for b in turn.bubbles)
        all_text_lower = all_text.lower()
        t1 = t.turns[0]
        t2 = t.turns[1] if len(t.turns) > 1 else None

        if comp == "SUFFICIENT":
            assert t.location_completeness == "SUFFICIENT", (
                f"SUFFICIENT scenario {t.scenario_id} has location_completeness '{t.location_completeness}'"
            )
            # Precise jurisdiction: street + RT/RW + kelurahan + kecamatan + city
            assert "jalan " in all_text_lower, f"SUFFICIENT scenario {t.scenario_id} missing street"
            assert "kelurahan " in all_text_lower, f"SUFFICIENT scenario {t.scenario_id} missing Kelurahan"
            assert "kecamatan " in all_text_lower, f"SUFFICIENT scenario {t.scenario_id} missing Kecamatan"
            assert any(k in all_text_lower for k in ("kota ", "kabupaten ")), (
                f"SUFFICIENT scenario {t.scenario_id} missing City"
            )
            assert "rt " in all_text_lower and "rw " in all_text_lower, (
                f"SUFFICIENT scenario {t.scenario_id} missing RT/RW"
            )
            assert t1.expected_action.allowed_actions == (DecisionMode.EXECUTE,)
            assert t1.expected_action.missing == ()
            if t2 is not None:
                assert t2.expected_action.allowed_actions == (DecisionMode.EXECUTE,)
                assert t2.expected_action.missing == ()

        elif comp == "INCOMPLETE":
            assert t.location_completeness == "INCOMPLETE", (
                f"INCOMPLETE scenario {t.scenario_id} has location_completeness '{t.location_completeness}'"
            )
            # Omit jurisdiction: must not match any admin jurisdiction patterns
            admin_loc_match = ADMIN_LOC_PATTERNS.search(all_text)
            assert admin_loc_match is None, (
                f"INCOMPLETE scenario {t.scenario_id} leaked administrative jurisdiction: '{admin_loc_match.group(0)}'"
            )
            # State landmark-only
            assert any(
                k in all_text_lower
                for k in (
                    "hanya berpatokan pada",
                    "hanya berupa patokan",
                    "hanya patokan",
                    "patokan lokasi hanya",
                )
            ), f"INCOMPLETE scenario {t.scenario_id} missing landmark-only statement: {all_text}"
            assert any(
                k in all_text_lower
                for k in (
                    "tanpa nama jalan",
                    "tanpa alamat jalan",
                    "belum ada nama jalan",
                    "belum tercantum rincian jalan",
                )
            ), f"INCOMPLETE scenario {t.scenario_id} missing jurisdiction omission statement: {all_text}"
            assert t1.expected_action.allowed_actions == (DecisionMode.REQUEST_CLARIFICATION,)
            assert t1.expected_action.missing == ("jurisdiction",)
            if t2 is not None:
                assert t2.expected_action.allowed_actions == (DecisionMode.REQUEST_CLARIFICATION,)
                assert t2.expected_action.missing == ("jurisdiction",)

        elif comp == "AMBIGUOUS":
            assert t.location_completeness == "AMBIGUOUS", (
                f"AMBIGUOUS scenario {t.scenario_id} has location_completeness '{t.location_completeness}'"
            )
            # Conflicting/uncertain location
            assert any(
                k in all_text_lower
                for k in (
                    "simpang siur",
                    "tidak pasti",
                    "belum pasti",
                    "membingungkan",
                    "saling bertentangan",
                )
            ), f"AMBIGUOUS scenario {t.scenario_id} missing uncertainty wording: {all_text}"
            assert "antara" in all_text_lower and "atau" in all_text_lower, (
                f"AMBIGUOUS scenario {t.scenario_id} missing conflicting landmark indicators (antara...atau): {all_text}"
            )
            # Explicit clarification-compatible wording
            assert any(
                k in all_text_lower
                for k in (
                    "klarifikasi",
                    "butuh klarifikasi",
                    "perlu klarifikasi",
                    "mohon klarifikasi",
                    "memerlukan klarifikasi",
                )
            ), f"AMBIGUOUS scenario {t.scenario_id} missing explicit clarification wording: {all_text}"
            assert t1.expected_action.allowed_actions == (DecisionMode.REQUEST_CLARIFICATION,)
            assert t1.expected_action.missing == ("location_ambiguity",)
            if t2 is not None:
                assert t2.expected_action.allowed_actions == (DecisionMode.REQUEST_CLARIFICATION,)
                assert t2.expected_action.missing == ("location_ambiguity",)


def _assert_canonical_obj_loc_time_strings(
    trajectories: Sequence[ComplaintTrajectory],
) -> None:
    obj_matches = 0
    loc_matches = 0
    time_matches = 0

    common_obj_keywords = (
        "jalan",
        "aspal",
        "lubang",
        "rusak",
        "trotoar",
        "lampu",
        "penerangan",
        "drainase",
        "banjir",
        "gorong",
        "saluran",
        "tanggul",
        "sampah",
        "limbah",
        "pipa",
        "air",
        "keran",
        "ktp",
        "kartu keluarga",
        "akta",
        "bansos",
        "puskesmas",
        "obat",
        "dokter",
    )
    common_loc_keywords = (
        "kelurahan",
        "kecamatan",
        "kota",
        "kabupaten",
        "jalan",
        "gang",
        "rt",
        "rw",
        "dekat",
        "seberang",
        "depan",
        "samping",
        "blok",
    )
    common_time_keywords = (
        "saat ini",
        "sekarang",
        "hari ini",
        "kemarin",
        "tadi",
        "pagi ini",
        "siang ini",
        "sore ini",
        "malam ini",
        "hari",
        "minggu",
        "bulan",
        "jam",
    )

    for t in trajectories:
        for turn in t.turns:
            for b in turn.bubbles:
                text = b.text
                text_lower = text.lower()

                if COMPACT_OBJECT_PATTERNS.search(text) or DAMAGE_PATTERNS.search(text) or any(k in text_lower for k in common_obj_keywords):
                    obj_matches += 1
                if ADMIN_LOC_PATTERNS.search(text) or any(k in text_lower for k in common_loc_keywords):
                    loc_matches += 1
                if TIME_PATTERNS.search(text) or any(k in text_lower for k in common_time_keywords):
                    time_matches += 1

    assert obj_matches > 0, "No canonical OBJ strings detected in reporter bubbles"
    assert loc_matches > 0, "No canonical LOC strings detected in reporter bubbles"
    assert time_matches > 0, "No canonical TIME strings detected in reporter bubbles"


def _assert_no_contamination(
    trajectories: Sequence[ComplaintTrajectory],
    source_dir: Path = SOURCE_ARTIFACTS_DIR,
) -> None:
    assert source_dir.is_dir(), f"Source artifacts directory '{source_dir}' must exist"

    source_trajectories_file = source_dir / "trajectories.jsonl"
    if not source_trajectories_file.is_file():
        source_trajectories_file = source_dir / "train.jsonl"
    assert source_trajectories_file.is_file(), f"Source trajectories file missing in {source_dir}"

    source_trajectories = load_trajectories_from_jsonl(source_trajectories_file)

    source_scenario_ids = {t.scenario_id for t in source_trajectories}
    source_family_ids = {t.family_id for t in source_trajectories}
    holdout_scenario_ids = {t.scenario_id for t in trajectories}
    holdout_family_ids = {t.family_id for t in trajectories}

    id_overlap = source_scenario_ids & holdout_scenario_ids
    assert len(id_overlap) == 0, f"Scenario ID contamination detected: {list(id_overlap)[:5]}"

    fam_overlap = source_family_ids & holdout_family_ids
    assert len(fam_overlap) == 0, f"Family ID contamination detected: {list(fam_overlap)[:5]}"

    for t in trajectories:
        for turn in t.turns:
            for b in turn.bubbles:
                b_lower = b.text.lower()
                for sid in list(source_scenario_ids)[:100]:
                    assert sid.lower() not in b_lower, (
                        f"Source scenario_id '{sid}' leaked into holdout bubble text"
                    )

    source_bubble_texts = {
        re.sub(r"\s+", " ", b.text.strip().lower())
        for t in source_trajectories
        for turn in t.turns
        for b in turn.bubbles
        if len(b.text.strip()) > 15
    }
    holdout_bubble_texts = {
        re.sub(r"\s+", " ", b.text.strip().lower())
        for t in trajectories
        for turn in t.turns
        for b in turn.bubbles
        if len(b.text.strip()) > 15
    }
    text_overlap = source_bubble_texts & holdout_bubble_texts
    assert len(text_overlap) == 0, (
        f"Text duplicate contamination between source and holdout: {list(text_overlap)[:3]}"
    )

    def _get_non_common_ngrams_with_sources(
        trajs: Sequence[ComplaintTrajectory],
        n: int,
    ) -> dict[tuple[str, ...], set[str]]:
        ngrams_map: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for t in trajs:
            for turn in t.turns:
                for b in turn.bubbles:
                    tokens = re.findall(r"[a-zA-Z0-9_\-]+", b.text.lower())
                    for ng in _extract_ngrams(tokens, n):
                        if not _is_common_language_ngram(ng):
                            ngrams_map[ng].add(t.scenario_id)
        return ngrams_map

    contam_violations: list[str] = []
    for n in (3, 4, 5):
        source_ngrams = _get_non_common_ngrams_with_sources(source_trajectories, n)
        holdout_ngrams = _get_non_common_ngrams_with_sources(trajectories, n)
        shared_ngrams = set(source_ngrams.keys()) & set(holdout_ngrams.keys())
        for ng in sorted(shared_ngrams):
            ng_text = " ".join(ng)
            src_samples = sorted(source_ngrams[ng])[:3]
            holdout_samples = sorted(holdout_ngrams[ng])[:3]
            contam_violations.append(
                f"Non-common {n}-gram '{ng_text}' appears in source scenarios {src_samples} "
                f"and holdout scenarios {holdout_samples}"
            )

    assert len(contam_violations) == 0, (
        f"Contamination detected between source dataset and holdout dataset ({len(contam_violations)} occurrences):\n"
        + "\n".join(f"  - {v}" for v in contam_violations)
    )


def _assert_no_pii(trajectories: Sequence[ComplaintTrajectory]) -> None:
    for t in trajectories:
        for turn_idx, turn in enumerate(t.turns):
            for b_idx, b in enumerate(turn.bubbles):
                text = b.text
                nik_match = NIK_REGEX.search(text)
                assert not nik_match, (
                    f"PII violation (NIK detected: '{nik_match.group(0)}') in scenario {t.scenario_id} "
                    f"turn {turn_idx + 1} bubble {b_idx + 1}"
                )
                phone_match = PHONE_REGEX.search(text)
                assert not phone_match, (
                    f"PII violation (Phone detected: '{phone_match.group(0)}') in scenario {t.scenario_id} "
                    f"turn {turn_idx + 1} bubble {b_idx + 1}"
                )
                email_match = EMAIL_REGEX.search(text)
                assert not email_match, (
                    f"PII violation (Email detected: '{email_match.group(0)}') in scenario {t.scenario_id} "
                    f"turn {turn_idx + 1} bubble {b_idx + 1}"
                )


def _assert_sha_manifest_validity_and_determinism(
    module: Any,
    output_dir: Path,
    source_dir: Path = SOURCE_ARTIFACTS_DIR,
    seed: int = 42,
) -> None:
    manifest_file = output_dir / "manifest.json"
    sha256sums_file = output_dir / "sha256sums.txt"

    if manifest_file.is_file():
        manifest = ArtifactManifest.from_file(manifest_file)
        for key, item in manifest.artifacts.items():
            artifact_file = output_dir / item.path
            assert artifact_file.is_file(), f"Artifact file '{artifact_file}' listed in manifest missing"
            computed_sha = compute_file_sha256(artifact_file)
            assert computed_sha == item.sha256, (
                f"SHA256 mismatch for {item.path}: expected {item.sha256}, got {computed_sha}"
            )

    if sha256sums_file.is_file():
        content = sha256sums_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected_sha, fname = parts
                fpath = output_dir / fname.strip()
                assert fpath.is_file(), f"File {fpath} listed in sha256sums.txt does not exist"
                computed = compute_file_sha256(fpath)
                assert computed == expected_sha, (
                    f"SHA256 mismatch for {fname}: expected {expected_sha}, got {computed}"
                )

    output_dir_determinism = output_dir.parent / "determinism_check"
    output_dir_determinism.mkdir(parents=True, exist_ok=True)

    _invoke_generator(module, output_dir=output_dir_determinism, source_dir=source_dir, seed=seed)

    for item in output_dir.iterdir():
        if item.is_file() and not item.name.startswith("."):
            det_file = output_dir_determinism / item.name
            assert det_file.is_file(), f"Deterministic re-run missing file '{item.name}'"
            sha1 = compute_file_sha256(item)
            sha2 = compute_file_sha256(det_file)
            assert sha1 == sha2, f"Determinism failure for file '{item.name}': {sha1} != {sha2}"


@pytest.fixture(scope="module")
def generated_holdout(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    holdout_dir = tmp_path_factory.mktemp("m3_independent_holdout_fixture")
    module = _load_generator_module()
    result = _invoke_generator(
        module=module,
        output_dir=holdout_dir,
        source_dir=SOURCE_ARTIFACTS_DIR,
        seed=42,
        count=300,
    )
    trajectories = _load_holdout_trajectories(holdout_dir)
    return {
        "module": module,
        "output_dir": holdout_dir,
        "result": result,
        "trajectories": trajectories,
    }


def test_holdout_trajectory_count(generated_holdout: dict[str, Any]) -> None:
    _assert_trajectory_count(generated_holdout["trajectories"])


def test_holdout_schema_and_metadata_provenance(generated_holdout: dict[str, Any]) -> None:
    _assert_metadata_provenance(
        generated_holdout["trajectories"],
        generated_holdout["output_dir"],
    )


def test_holdout_coverage_all_labels(generated_holdout: dict[str, Any]) -> None:
    _assert_six_category_and_head_coverage(generated_holdout["trajectories"])


def test_holdout_canonical_entities(generated_holdout: dict[str, Any]) -> None:
    _assert_canonical_obj_loc_time_strings(generated_holdout["trajectories"])


def test_holdout_no_pii(generated_holdout: dict[str, Any]) -> None:
    _assert_no_pii(generated_holdout["trajectories"])


def test_holdout_sha_manifest_and_hashes(generated_holdout: dict[str, Any]) -> None:
    output_dir = generated_holdout["output_dir"]
    manifest_file = output_dir / "manifest.json"
    sha256sums_file = output_dir / "sha256sums.txt"

    assert manifest_file.is_file(), f"Missing manifest.json in {output_dir}"
    manifest = ArtifactManifest.from_file(manifest_file)
    for key, item in manifest.artifacts.items():
        artifact_file = output_dir / item.path
        assert artifact_file.is_file(), f"Artifact file '{artifact_file}' listed in manifest missing"
        computed_sha = compute_file_sha256(artifact_file)
        assert computed_sha == item.sha256, (
            f"SHA256 mismatch for {item.path}: expected {item.sha256}, got {computed_sha}"
        )

    assert sha256sums_file.is_file(), f"Missing sha256sums.txt in {output_dir}"
    content = sha256sums_file.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            expected_sha, fname = parts
            fpath = output_dir / fname.strip()
            assert fpath.is_file(), f"File {fpath} listed in sha256sums.txt does not exist"
            computed = compute_file_sha256(fpath)
            assert computed == expected_sha, (
                f"SHA256 mismatch for {fname}: expected {expected_sha}, got {computed}"
            )


def test_holdout_determinism(generated_holdout: dict[str, Any]) -> None:
    module = generated_holdout["module"]
    output_dir = generated_holdout["output_dir"]
    output_dir_determinism = output_dir.parent / "determinism_modular_check"
    output_dir_determinism.mkdir(parents=True, exist_ok=True)

    _invoke_generator(module, output_dir=output_dir_determinism, source_dir=SOURCE_ARTIFACTS_DIR, seed=42)

    for item in output_dir.iterdir():
        if item.is_file() and not item.name.startswith("."):
            det_file = output_dir_determinism / item.name
            assert det_file.is_file(), f"Deterministic re-run missing file '{item.name}'"
            sha1 = compute_file_sha256(item)
            sha2 = compute_file_sha256(det_file)
            assert sha1 == sha2, f"Determinism failure for file '{item.name}': {sha1} != {sha2}"


def test_holdout_no_source_contamination(generated_holdout: dict[str, Any]) -> None:
    _assert_no_contamination(generated_holdout["trajectories"], SOURCE_ARTIFACTS_DIR)


def test_holdout_risk_semantic_grounding(generated_holdout: dict[str, Any]) -> None:
    _assert_risk_semantic_grounding(generated_holdout["trajectories"])


def test_holdout_completeness_semantic_grounding(generated_holdout: dict[str, Any]) -> None:
    _assert_completeness_semantic_grounding(generated_holdout["trajectories"])


def test_committed_artifact_label_text_alignment() -> None:
    holdout_dir = REPO_ROOT / "artifacts" / "m3_independent_holdout"
    trajectories = _load_holdout_trajectories(holdout_dir)
    assert len(trajectories) == 300, f"Expected 300 holdout trajectories, got {len(trajectories)}"
    _assert_risk_semantic_grounding(trajectories)
    _assert_completeness_semantic_grounding(trajectories)


def test_m3_independent_holdout_pipeline(tmp_path: Path) -> None:
    generator_module = _load_generator_module()

    holdout_out_dir = tmp_path / "holdout_run"
    holdout_out_dir.mkdir(parents=True, exist_ok=True)

    _invoke_generator(
        module=generator_module,
        output_dir=holdout_out_dir,
        source_dir=SOURCE_ARTIFACTS_DIR,
        seed=42,
    )

    trajectories = _load_holdout_trajectories(holdout_out_dir)

    _assert_trajectory_count(trajectories)
    _assert_metadata_provenance(trajectories, holdout_out_dir)
    _assert_six_category_and_head_coverage(trajectories)
    _assert_canonical_obj_loc_time_strings(trajectories)
    _assert_risk_semantic_grounding(trajectories)
    _assert_completeness_semantic_grounding(trajectories)
    _assert_no_contamination(trajectories, SOURCE_ARTIFACTS_DIR)
    _assert_no_pii(trajectories)
    _assert_sha_manifest_validity_and_determinism(
        module=generator_module,
        output_dir=holdout_out_dir,
        source_dir=SOURCE_ARTIFACTS_DIR,
        seed=42,
    )
