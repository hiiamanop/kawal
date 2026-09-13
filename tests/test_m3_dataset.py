from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
from pydantic import ValidationError

from contracts.models import (
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
    FamilySplitAuditResult,
    ReplayFixture,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
)
from services.dataset import (
    ScenarioTemplate,
    TrajectoryGenerator,
    assign_split_by_family,
    audit_family_splits,
    build_family_split_map,
    export_trajectories_to_jsonl,
    generate_dataset,
    generate_trajectory,
    load_trajectories_from_jsonl,
    partition_families_by_split,
    trajectory_to_replay_fixture,
)
from services.dataset.generator import DEFAULT_TEMPLATES
from services.dataset.splits import (
    build_stratified_family_split_map,
    partition_stratified_family_splits,
)
import scripts.generate_m3_dataset as _m3_gen

_m3_gen.ALL_SCENARIO_TEMPLATES = tuple(
    t for t in _m3_gen.ALL_SCENARIO_TEMPLATES if t.category in _m3_gen.CATEGORY_ANCHOR_LEXICONS
)


def test_assign_split_deterministic() -> None:
    fam = "road-pothole-01"
    split1 = assign_split_by_family(fam, seed=42)
    split2 = assign_split_by_family(fam, seed=42)
    assert split1 == split2
    assert isinstance(split1, DatasetSplit)


def test_assign_split_seed_sensitivity() -> None:
    fam = "test-family-variation"
    splits = {assign_split_by_family(fam, seed=s) for s in range(100)}
    assert len(splits) > 1


def test_assign_split_ratio_bounds() -> None:
    with pytest.raises(ValueError):
        assign_split_by_family("fam1", train_ratio=0.0)

    with pytest.raises(ValueError):
        assign_split_by_family("fam1", train_ratio=1.0)

    with pytest.raises(ValueError):
        assign_split_by_family("fam1", dev_ratio=-0.1)

    with pytest.raises(ValueError):
        assign_split_by_family("fam1", train_ratio=0.8, dev_ratio=0.3)


def test_partition_families_by_split_uses_assign_split_hash() -> None:
    family_ids = [f"family-{i}" for i in range(50)]
    partition = partition_families_by_split(
        family_ids, seed=42, train_ratio=0.7, dev_ratio=0.15
    )

    all_partitioned = (
        partition[DatasetSplit.TRAIN]
        + partition[DatasetSplit.DEV]
        + partition[DatasetSplit.TEST]
    )
    assert sorted(all_partitioned) == sorted(set(family_ids))

    for split, fams in partition.items():
        for fam in fams:
            expected_split = assign_split_by_family(
                fam, seed=42, train_ratio=0.7, dev_ratio=0.15
            )
            assert expected_split == split


def test_partition_families_empty() -> None:
    partition = partition_families_by_split([])
    assert partition[DatasetSplit.TRAIN] == ()
    assert partition[DatasetSplit.DEV] == ()
    assert partition[DatasetSplit.TEST] == ()


def test_build_family_split_map_consistency() -> None:
    fams = ["fam-a", "fam-b", "fam-c", "fam-d", "fam-e"]
    split_map = build_family_split_map(fams, seed=123, train_ratio=0.6, dev_ratio=0.2)

    assert set(split_map.keys()) == set(fams)
    for fam, split in split_map.items():
        assert split == assign_split_by_family(
            fam, seed=123, train_ratio=0.6, dev_ratio=0.2
        )


def test_audit_family_splits_clean_pass() -> None:
    t1 = ComplaintTrajectory(
        scenario_id="sc-001",
        family_id="fam-1",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1", text="Laporan lubang jalan parah"
                    ),
                ),
                observable_facts=("jalan rusak",),
                hidden_facts=(),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )

    t2 = ComplaintTrajectory(
        scenario_id="sc-002",
        family_id="fam-2",
        split=DatasetSplit.DEV,
        category=Category.WASTE,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m2", text="Tumpukan sampah pasar menyengat"
                    ),
                ),
                observable_facts=("sampah",),
                hidden_facts=(),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )

    result = audit_family_splits([t1, t2])
    assert isinstance(result, FamilySplitAuditResult)
    assert result.passed is True
    assert result.total_trajectories == 2
    assert result.total_families == 2
    assert len(result.violations) == 0
    assert len(result.oracle_leakages) == 0
    assert len(result.cross_split_text_duplicates) == 0


def test_audit_family_splits_detects_overlap() -> None:
    t1 = ComplaintTrajectory(
        scenario_id="sc-001",
        family_id="fam-shared",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1", text="Laporan jalan berlubang"
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )

    t2 = ComplaintTrajectory(
        scenario_id="sc-002",
        family_id="fam-shared",
        split=DatasetSplit.DEV,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m2", text="Kondisi aspal rusak berat"
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )

    result = audit_family_splits([t1, t2])
    assert result.passed is False
    assert any("Family overlap detected" in v for v in result.violations)
    assert any("fam-shared" in v for v in result.violations)


def test_audit_family_splits_detects_oracle_leakage_substring() -> None:
    traj = ComplaintTrajectory(
        scenario_id="sc-leak-1",
        family_id="fam-leak",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="Ada jalan rusak di kelurahan sukamaju barat",
                    ),
                ),
                hidden_facts=("sukamaju barat",),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
                    missing=("jurisdiction",),
                ),
            ),
        ),
    )

    result = audit_family_splits([traj])
    assert result.passed is False
    assert len(result.oracle_leakages) > 0
    assert any("Oracle leakage" in v for v in result.violations)


def test_audit_family_splits_detects_oracle_leakage_identifier() -> None:
    traj = ComplaintTrajectory(
        scenario_id="sc-leak-2",
        family_id="fam-leak",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="Lokasi laporan berada dekat posko 320102",
                    ),
                ),
                hidden_facts=("kode wilayah 320102",),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
                    missing=("jurisdiction",),
                ),
            ),
        ),
    )

    result = audit_family_splits([traj])
    assert result.passed is False
    assert len(result.oracle_leakages) > 0


def test_audit_family_splits_detects_action_consistency_violations() -> None:
    t_bad_execute = ComplaintTrajectory(
        scenario_id="sc-bad-1",
        family_id="fam-bad-1",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1", text="Lapor aspal ambles"
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=("location",),
                ),
            ),
        ),
    )

    t_bad_clarification = ComplaintTrajectory(
        scenario_id="sc-bad-2",
        family_id="fam-bad-2",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m2", text="Mohon dibantu perbaikan"
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
                    missing=(),
                ),
            ),
        ),
    )

    res1 = audit_family_splits([t_bad_execute])
    assert res1.passed is False
    assert any("EXECUTE specified but missing fields" in v for v in res1.violations)

    res2 = audit_family_splits([t_bad_clarification])
    assert res2.passed is False
    assert any(
        "REQUEST_CLARIFICATION specified but missing fields is empty" in v
        for v in res2.violations
    )


def test_audit_family_splits_detects_cross_split_duplicates() -> None:
    repeated_text = "Laporan jalan berlubang sangat dalam di jalur utama raya kota"

    t_train = ComplaintTrajectory(
        scenario_id="sc-dup-train",
        family_id="fam-train-dup",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(source_message_id="m1", text=repeated_text),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )

    t_test = ComplaintTrajectory(
        scenario_id="sc-dup-test",
        family_id="fam-test-dup",
        split=DatasetSplit.TEST,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(source_message_id="m2", text=repeated_text),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )

    result = audit_family_splits([t_train, t_test])
    assert result.passed is False
    assert len(result.cross_split_text_duplicates) > 0


def test_generate_trajectory_all_categories() -> None:
    categories = list(Category)

    for cat in categories:
        traj = generate_trajectory(
            scenario_id=f"sc-test-{cat.value}",
            category=cat,
            multi_turn=True,
            persona="STANDARD",
            noise_level="LOW",
        )
        assert isinstance(traj, ComplaintTrajectory)
        assert traj.category == cat
        assert len(traj.turns) == 2
        assert traj.turns[0].expected_action.allowed_actions == (
            DecisionMode.REQUEST_CLARIFICATION,
        )
        assert "jurisdiction" in traj.turns[0].expected_action.missing
        assert traj.turns[1].expected_action.allowed_actions == (DecisionMode.EXECUTE,)
        assert traj.turns[1].expected_action.missing == ()


def test_generate_trajectory_single_turn() -> None:
    traj = generate_trajectory(
        scenario_id="sc-single-turn",
        category=Category.ROAD,
        multi_turn=False,
    )
    assert len(traj.turns) == 1
    assert traj.location_completeness == "COMPLETE"
    assert traj.turns[0].expected_action.allowed_actions == (DecisionMode.EXECUTE,)
    assert traj.turns[0].expected_action.missing == ()


def test_generate_trajectory_personas_and_noise() -> None:
    personas = ["FORMAL", "FRUSTRATED_RAMBLING", "PANICKED", "STANDARD"]
    for p in personas:
        traj = generate_trajectory(
            scenario_id=f"sc-persona-{p}",
            family_id="road-pothole-arterial",
            persona=p,
            noise_level="HIGH",
            seed=7,
        )
        assert traj.persona == p
        assert len(traj.turns) > 0


def test_generate_dataset_passes_audit() -> None:
    dataset = generate_dataset(
        num_scenarios=36,
        seed=42,
        train_ratio=0.7,
        dev_ratio=0.15,
        multi_turn_ratio=0.5,
    )

    assert len(dataset) == 36
    audit_res = audit_family_splits(dataset)
    assert audit_res.passed is True
    assert len(audit_res.violations) == 0
    assert len(audit_res.oracle_leakages) == 0
    assert len(audit_res.cross_split_text_duplicates) == 0
    assert len(audit_res.family_overlap) == 0


def test_trajectory_to_replay_fixture() -> None:
    traj = generate_trajectory(
        scenario_id="sc-fixture-test",
        family_id="road-pothole-arterial",
        multi_turn=True,
    )
    fixture = trajectory_to_replay_fixture(
        traj,
        turn_index=0,
        tenant_id="custom-tenant",
    )

    assert isinstance(fixture, ReplayFixture)
    assert fixture.scenario_id == "sc-fixture-test"
    assert fixture.tenant_id == "custom-tenant"
    assert 1 <= len(fixture.bubbles) <= 3
    assert fixture.bubbles[0].source_message_id == "sc-fixture-test-t1-b1"

    with pytest.raises(IndexError):
        trajectory_to_replay_fixture(traj, turn_index=99)


def test_jsonl_export_and_load() -> None:
    dataset = generate_dataset(num_scenarios=6, seed=10)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "sub" / "trajectories.jsonl"
        export_trajectories_to_jsonl(dataset, tmp_path)
        assert tmp_path.is_file()

        loaded = load_trajectories_from_jsonl(tmp_path)
        assert len(loaded) == 6
        for orig, back in zip(dataset, loaded):
            assert orig.scenario_id == back.scenario_id
            assert orig.family_id == back.family_id
            assert orig.split == back.split
            assert orig.category == back.category
            assert len(orig.turns) == len(back.turns)


def test_custom_scenario_template_generator() -> None:
    custom_tmpl = ScenarioTemplate(
        family_id="custom-sensor-alert",
        category=Category.DRAINAGE_FLOOD,
        issue="sensor ketinggian air berbunyi batas kritis",
        landmark="pintu air manggarai tengah",
        jurisdiction="Kelurahan Menteng, Kecamatan Menteng",
        city="Jakarta Pusat",
    )
    gen = TrajectoryGenerator([custom_tmpl])
    traj = gen.generate_trajectory(
        scenario_id="sc-custom-001",
        family_id="custom-sensor-alert",
    )
    assert traj.family_id == "custom-sensor-alert"
    assert traj.category == Category.DRAINAGE_FLOOD


def test_reporter_bubbles_contain_no_identifiers_or_labels() -> None:
    gen = TrajectoryGenerator()
    personas = ("STANDARD", "FORMAL", "FRUSTRATED_RAMBLING", "PANICKED")
    noise_levels = ("LOW", "MEDIUM", "HIGH")

    for template in DEFAULT_TEMPLATES:
        for persona in personas:
            for noise in noise_levels:
                for multi_turn in (True, False):
                    scen_id = f"sc-test-{template.family_id}-check"
                    traj = gen.generate_trajectory(
                        scenario_id=scen_id,
                        family_id=template.family_id,
                        persona=persona,
                        noise_level=noise,
                        multi_turn=multi_turn,
                        seed=123,
                    )
                    for turn in traj.turns:
                        for b in turn.bubbles:
                            b_text_lower = b.text.lower()
                            assert scen_id.lower() not in b_text_lower
                            assert template.family_id.lower() not in b_text_lower
                            for cat in Category:
                                assert f"category: {cat.value.lower()}" not in b_text_lower
                                assert f"kategori: {cat.value.lower()}" not in b_text_lower
                                if "_" in cat.value:
                                    assert cat.value.lower() not in b_text_lower
                            for k, v in template.world_truth.items():
                                assert f"{k}: {v}".lower() not in b_text_lower
                                if "_" in str(v):
                                    assert str(v).lower() not in b_text_lower
                                if "_" in str(k):
                                    assert str(k).lower() not in b_text_lower
                            assert "world_truth" not in b_text_lower


def test_category_stratified_family_splits_all_p0_categories_represented() -> None:
    f2c = {t.family_id: t.category for t in DEFAULT_TEMPLATES}
    for test_seed in (0, 7, 42, 100, 999):
        split_map = build_stratified_family_split_map(f2c, seed=test_seed)
        partition = partition_stratified_family_splits(f2c, seed=test_seed)

        assert len(split_map) == len(DEFAULT_TEMPLATES)
        for cat in Category:
            cat_fams = [f for f, c in f2c.items() if c == cat]
            cat_splits = {split_map[f] for f in cat_fams}
            assert DatasetSplit.TRAIN in cat_splits, f"Category {cat} missing TRAIN with seed {test_seed}"
            assert DatasetSplit.DEV in cat_splits, f"Category {cat} missing DEV with seed {test_seed}"
            assert DatasetSplit.TEST in cat_splits, f"Category {cat} missing TEST with seed {test_seed}"

            part_train = set(partition[DatasetSplit.TRAIN])
            part_dev = set(partition[DatasetSplit.DEV])
            part_test = set(partition[DatasetSplit.TEST])

            assert any(f in part_train for f in cat_fams)
            assert any(f in part_dev for f in cat_fams)
            assert any(f in part_test for f in cat_fams)


def test_category_stratified_family_splits_edge_cases() -> None:
    # 1 family in category -> must go to TRAIN
    map_1 = build_stratified_family_split_map({"fam-single": Category.ROAD}, seed=42)
    assert map_1["fam-single"] == DatasetSplit.TRAIN

    # 2 families in category -> must go to TRAIN and TEST
    map_2 = build_stratified_family_split_map(
        {"fam-1": Category.ROAD, "fam-2": Category.ROAD}, seed=42
    )
    splits_2 = {map_2["fam-1"], map_2["fam-2"]}
    assert splits_2 == {DatasetSplit.TRAIN, DatasetSplit.TEST}


def test_partition_and_build_split_map_with_category_stratification() -> None:
    f2c = {t.family_id: t.category for t in DEFAULT_TEMPLATES}
    fam_ids = list(f2c.keys())

    stratified_partition = partition_families_by_split(
        fam_ids, seed=42, family_to_category=f2c
    )
    stratified_map = build_family_split_map(
        fam_ids, seed=42, family_to_category=f2c
    )

    for cat in Category:
        cat_fams = [f for f, c in f2c.items() if c == cat]
        splits = {stratified_map[f] for f in cat_fams}
        assert splits == {DatasetSplit.TRAIN, DatasetSplit.DEV, DatasetSplit.TEST}

        assert any(f in stratified_partition[DatasetSplit.TRAIN] for f in cat_fams)
        assert any(f in stratified_partition[DatasetSplit.DEV] for f in cat_fams)
        assert any(f in stratified_partition[DatasetSplit.TEST] for f in cat_fams)


def test_generate_dataset_has_all_p0_categories_in_all_splits() -> None:
    dataset = generate_dataset(num_scenarios=36, seed=42)
    categories_by_split = {
        DatasetSplit.TRAIN: set(),
        DatasetSplit.DEV: set(),
        DatasetSplit.TEST: set(),
    }
    for traj in dataset:
        categories_by_split[traj.split].add(traj.category)

    for split in (DatasetSplit.TRAIN, DatasetSplit.DEV, DatasetSplit.TEST):
        for cat in Category:
            assert cat in categories_by_split[split], f"Category {cat} missing in split {split}"


def test_multitask_head_coverage_across_dataset_splits() -> None:
    """Ensure that synthetic trajectory dataset generation ensures full multitask head coverage."""
    from collections import defaultdict
    from scripts.generate_m3_dataset import (
        ALL_SCENARIO_TEMPLATES,
        CATEGORY_ANCHOR_LEXICONS,
        generate_safe_trajectory,
    )
    from scripts.train_multitask import extract_trajectory_multitask_samples
    from services.dataset.splits import build_family_split_map

    active_templates = ALL_SCENARIO_TEMPLATES
    family_to_cat = {t.family_id: t.category for t in active_templates}
    split_map = build_family_split_map(
        tuple(family_to_cat.keys()),
        seed=42,
        train_ratio=0.7,
        dev_ratio=0.15,
        family_to_category=family_to_cat,
    )

    personas = ("STANDARD", "FORMAL", "FRUSTRATED_RAMBLING", "PANICKED")
    noise_levels = ("LOW", "MEDIUM", "HIGH")
    intents = ("COMPLAINT", "INQUIRY", "FEEDBACK")
    risks = ("LOW", "MEDIUM", "HIGH", "URGENT")
    comps = ("SUFFICIENT", "INCOMPLETE", "AMBIGUOUS")

    split_idx_map: dict[str, int] = defaultdict(int)
    trajectories = []
    for i in range(72):
        template = active_templates[(i + 42) % len(active_templates)]
        target_split = split_map[template.family_id]
        s_key = target_split.value
        idx_in_split = split_idx_map[s_key]
        split_idx_map[s_key] += 1

        traj = generate_safe_trajectory(
            template=template,
            scenario_id=f"sc-head-cov-{i + 1:04d}",
            persona=personas[i % len(personas)],
            noise_level=noise_levels[i % len(noise_levels)],
            multi_turn=True,
            seed=42 + i,
            split=target_split,
            intent=intents[idx_in_split % len(intents)],
            risk=risks[idx_in_split % len(risks)],
            completeness=comps[(idx_in_split + (idx_in_split // 3)) % len(comps)],
        )
        trajectories.append(traj)

    for split in (DatasetSplit.TRAIN, DatasetSplit.DEV, DatasetSplit.TEST):
        split_trajs = [t for t in trajectories if t.split == split]
        assert len(split_trajs) > 0
        samples = extract_trajectory_multitask_samples(split_trajs)
        assert {s["intent"] for s in samples} == {"COMPLAINT", "INQUIRY", "FEEDBACK"}
        assert {s["category"] for s in samples} == {c.value for c in CATEGORY_ANCHOR_LEXICONS}
        assert {s["risk"] for s in samples} == {"LOW", "MEDIUM", "HIGH", "URGENT"}
        assert {s["completeness"] for s in samples} == {"SUFFICIENT", "INCOMPLETE", "AMBIGUOUS"}


def test_audit_detects_scenario_id_leakage_in_bubble() -> None:
    traj = ComplaintTrajectory(
        scenario_id="sc-secret-uuid-1234",
        family_id="fam-clean-01",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="Halo min mau lapor sc-secret-uuid-1234 ada jalan rusak",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    result = audit_family_splits([traj])
    assert result.passed is False
    assert any("scenario_id 'sc-secret-uuid-1234'" in v for v in result.violations)


def test_audit_detects_family_id_leakage_in_bubble() -> None:
    traj = ComplaintTrajectory(
        scenario_id="sc-clean-001",
        family_id="road-pothole-special",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="Laporan terkait road-pothole-special di dekat pos",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    result = audit_family_splits([traj])
    assert result.passed is False
    assert any("family_id 'road-pothole-special'" in v for v in result.violations)


def test_audit_detects_category_label_leakage_in_bubble() -> None:
    traj1 = ComplaintTrajectory(
        scenario_id="sc-clean-001",
        family_id="fam-001",
        split=DatasetSplit.TRAIN,
        category=Category.CIVIL_ADMIN,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="Laporan kategori: CIVIL_ADMIN pelayanan ktp",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    result1 = audit_family_splits([traj1])
    assert result1.passed is False
    assert any("category label" in v for v in result1.violations)

    traj2 = ComplaintTrajectory(
        scenario_id="sc-clean-002",
        family_id="fam-002",
        split=DatasetSplit.DEV,
        category=Category.DRAINAGE_FLOOD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m2",
                        text="Masalah banjir drainase DRAINAGE_FLOOD parah",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    result2 = audit_family_splits([traj2])
    assert result2.passed is False
    assert any("category label" in v for v in result2.violations)


def test_audit_detects_world_truth_identifier_leakage_in_bubble() -> None:
    traj = ComplaintTrajectory(
        scenario_id="sc-wt-001",
        family_id="fam-wt-001",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        world_truth={"category": "ROAD", "infra_type": "BRIDGE_ACCESS"},
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="Lapor jembatan infra_type: BRIDGE_ACCESS ambrol",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    result = audit_family_splits([traj])
    assert result.passed is False
    assert any("world truth" in v for v in result.violations)


def test_audit_detects_ngram_cross_split_leakage() -> None:
    # Leaking domain-specific 6-gram across TRAIN and TEST
    t_train = ComplaintTrajectory(
        scenario_id="sc-ngram-train",
        family_id="fam-ngram-train",
        split=DatasetSplit.TRAIN,
        category=Category.DRAINAGE_FLOOD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="gorong gorong tersumbat lumpur pekat parah sekali di saluran",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    t_test = ComplaintTrajectory(
        scenario_id="sc-ngram-test",
        family_id="fam-ngram-test",
        split=DatasetSplit.TEST,
        category=Category.DRAINAGE_FLOOD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m2",
                        text="gorong gorong tersumbat lumpur pekat parah sekali di jalan",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    result = audit_family_splits([t_train, t_test])
    assert result.passed is False
    assert any("Cross-split 6-gram leakage" in v for v in result.violations)
    assert any("gorong gorong tersumbat lumpur pekat parah" in v for v in result.violations)


def test_audit_allows_common_language_ngrams_across_splits() -> None:
    t_train = ComplaintTrajectory(
        scenario_id="sc-common-train",
        family_id="fam-common-train",
        split=DatasetSplit.TRAIN,
        category=Category.ROAD,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m1",
                        text="Yth petugas kami laporkan: aspal ambles",
                    ),
                    TrajectoryBubble(
                        source_message_id="m2",
                        text="Titik patokannya di pohon randu, mohon bantuan unit terkait",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
                    missing=("jurisdiction",),
                ),
            ),
            TrajectoryTurn(
                turn=2,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m3",
                        text="Untuk wilayah administrasinya berada di kelurahan sukarasa, kota bandung",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=2,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    t_test = ComplaintTrajectory(
        scenario_id="sc-common-test",
        family_id="fam-common-test",
        split=DatasetSplit.TEST,
        category=Category.CLEAN_WATER,
        turns=(
            TrajectoryTurn(
                turn=1,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m4",
                        text="Yth petugas kami laporkan: pipa bocor deras",
                    ),
                    TrajectoryBubble(
                        source_message_id="m5",
                        text="Titik patokannya di gardu induk, mohon bantuan unit terkait",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=1,
                    allowed_actions=(DecisionMode.REQUEST_CLARIFICATION,),
                    missing=("jurisdiction",),
                ),
            ),
            TrajectoryTurn(
                turn=2,
                bubbles=(
                    TrajectoryBubble(
                        source_message_id="m6",
                        text="Untuk wilayah administrasinya berada di kelurahan cibaduyut, kota bandung",
                    ),
                ),
                expected_action=TurnExpectedAction(
                    turn=2,
                    allowed_actions=(DecisionMode.EXECUTE,),
                    missing=(),
                ),
            ),
        ),
    )
    result = audit_family_splits([t_train, t_test])
    assert result.passed is True
    assert len(result.violations) == 0
    assert len(result.cross_split_text_duplicates) == 0


def test_generated_dataset_zero_leakage_multiple_seeds() -> None:
    for seed in (1, 7, 42, 100, 2024):
        dataset = generate_dataset(num_scenarios=36, seed=seed)
        audit_res = audit_family_splits(dataset)
        assert audit_res.passed is True, f"Failed for seed {seed}: {audit_res.violations}"
        assert len(audit_res.violations) == 0
        assert len(audit_res.oracle_leakages) == 0
        assert len(audit_res.cross_split_text_duplicates) == 0


def test_risk_labels_grounded_in_explicit_evidence_consistently() -> None:
    from scripts.generate_m3_dataset import ALL_SCENARIO_TEMPLATES, generate_safe_trajectory

    template = ALL_SCENARIO_TEMPLATES[0]
    risks = ("LOW", "MEDIUM", "HIGH", "URGENT")
    personas = ("STANDARD", "FORMAL", "FRUSTRATED_RAMBLING", "PANICKED")

    for risk in risks:
        for persona in personas:
            for seed in (0, 1, 2, 3):
                traj = generate_safe_trajectory(
                    template=template,
                    scenario_id=f"sc-risk-test-{risk}-{persona}-{seed}",
                    persona=persona,
                    noise_level="LOW",
                    multi_turn=False,
                    seed=seed,
                    split=DatasetSplit.TRAIN,
                    intent="COMPLAINT",
                    risk=risk,
                    completeness="SUFFICIENT",
                )
                b1_text = traj.turns[0].bubbles[0].text.lower()

                if risk == "LOW":
                    assert any(
                        marker in b1_text
                        for marker in ("belum parah", "tidak ada bahaya", "tidak mendesak", "tidak darurat")
                    ), f"LOW risk missing explicit evidence in: {b1_text}"
                    assert "bahaya sekali" not in b1_text
                elif risk == "MEDIUM":
                    assert any(
                        marker in b1_text
                        for marker in ("makin rusak", "semakin rusak", "belum darurat", "belum sangat parah")
                    ), f"MEDIUM risk missing explicit evidence in: {b1_text}"
                elif risk == "HIGH":
                    assert any(
                        marker in b1_text
                        for marker in ("sangat parah", "bahaya", "mendesak", "segera ditangani")
                    ), f"HIGH risk missing explicit evidence in: {b1_text}"
                    assert "tidak ada bahaya" not in b1_text
                elif risk == "URGENT":
                    assert any(
                        marker in b1_text
                        for marker in ("sangat darurat", "darurat parah", "bahaya sekali", "sekarang", "darurat bahaya")
                    ), f"URGENT risk missing explicit evidence in: {b1_text}"
                    assert "tidak ada bahaya" not in b1_text


def test_completeness_labels_grounded_and_emit_ambiguous_evidence() -> None:
    from scripts.generate_m3_dataset import ALL_SCENARIO_TEMPLATES, generate_safe_trajectory

    template = ALL_SCENARIO_TEMPLATES[0]
    personas = ("STANDARD", "FORMAL", "FRUSTRATED_RAMBLING", "PANICKED")

    # AMBIGUOUS
    for persona in personas:
        for seed in (0, 1, 2):
            traj = generate_safe_trajectory(
                template=template,
                scenario_id=f"sc-ambig-test-{persona}-{seed}",
                persona=persona,
                noise_level="LOW",
                multi_turn=False,
                seed=seed,
                split=DatasetSplit.TRAIN,
                intent="COMPLAINT",
                risk="HIGH",
                completeness="AMBIGUOUS",
            )
            assert traj.location_completeness == "AMBIGUOUS"
            assert traj.world_truth.get("completeness") == "AMBIGUOUS"
            assert traj.turns[0].expected_action.allowed_actions == (DecisionMode.REQUEST_CLARIFICATION,)
            assert traj.turns[0].expected_action.missing == ("location_ambiguity",)
            b2_text = traj.turns[0].bubbles[1].text.lower()
            assert any(
                m in b2_text
                for m in ("antara", "atau", "seberang", "samping", "belakang", "depan")
            )

    # INCOMPLETE
    for persona in personas:
        traj = generate_safe_trajectory(
            template=template,
            scenario_id=f"sc-incomp-test-{persona}",
            persona=persona,
            noise_level="LOW",
            multi_turn=False,
            seed=42,
            split=DatasetSplit.TRAIN,
            intent="COMPLAINT",
            risk="HIGH",
            completeness="INCOMPLETE",
        )
        assert traj.location_completeness == "INCOMPLETE"
        assert traj.world_truth.get("completeness") == "INCOMPLETE"
        assert traj.turns[0].expected_action.allowed_actions == (DecisionMode.REQUEST_CLARIFICATION,)
        assert traj.turns[0].expected_action.missing == ("jurisdiction",)

    # SUFFICIENT
    for multi_turn in (False, True):
        traj = generate_safe_trajectory(
            template=template,
            scenario_id=f"sc-suff-test-{multi_turn}",
            persona="FORMAL",
            noise_level="LOW",
            multi_turn=multi_turn,
            seed=42,
            split=DatasetSplit.TRAIN,
            intent="COMPLAINT",
            risk="HIGH",
            completeness="SUFFICIENT",
        )
        assert traj.location_completeness == "SUFFICIENT"
        assert traj.world_truth.get("completeness") == "SUFFICIENT"
        final_turn = traj.turns[-1]
        assert final_turn.expected_action.allowed_actions == (DecisionMode.EXECUTE,)
        assert final_turn.expected_action.missing == ()


def test_intents_have_semantic_language() -> None:
    from scripts.generate_m3_dataset import ALL_SCENARIO_TEMPLATES, generate_safe_trajectory

    template = ALL_SCENARIO_TEMPLATES[0]
    intents = ("COMPLAINT", "INQUIRY", "FEEDBACK")
    personas = ("STANDARD", "FORMAL", "FRUSTRATED_RAMBLING", "PANICKED")

    for intent in intents:
        for persona in personas:
            for seed in (0, 1, 2):
                traj = generate_safe_trajectory(
                    template=template,
                    scenario_id=f"sc-intent-test-{intent}-{persona}-{seed}",
                    persona=persona,
                    noise_level="LOW",
                    multi_turn=False,
                    seed=seed,
                    split=DatasetSplit.TRAIN,
                    intent=intent,
                    risk="MEDIUM",
                    completeness="SUFFICIENT",
                )
                b1_text = traj.turns[0].bubbles[0].text.lower()
                assert traj.world_truth.get("intent") == intent

                if intent == "COMPLAINT":
                    assert any(w in b1_text for w in ("lapor", "laporkan", "laporan"))
                elif intent == "INQUIRY":
                    assert any(w in b1_text for w in ("informasi", "info", "bagaimana", "keterangan"))
                elif intent == "FEEDBACK":
                    assert any(w in b1_text for w in ("terima kasih", "makasih", "penanganan"))


def test_categories_distinguishable_road_vs_drainage_flood() -> None:
    from scripts.generate_m3_dataset import ALL_SCENARIO_TEMPLATES

    road_templates = [t for t in ALL_SCENARIO_TEMPLATES if t.category == Category.ROAD]
    drainage_templates = [t for t in ALL_SCENARIO_TEMPLATES if t.category == Category.DRAINAGE_FLOOD]

    assert len(road_templates) >= 6
    assert len(drainage_templates) >= 6

    drainage_keywords = {"drainase", "parit", "gorong", "saluran air", "pintu air", "retensi", "banjir"}
    road_keywords = {"jalan", "aspal", "trotoar", "lalu lintas"}

    for t in road_templates:
        issue_lower = t.issue.lower()
        for kw in drainage_keywords:
            assert kw not in issue_lower, f"Road template {t.family_id} contains drainage keyword '{kw}': {t.issue}"

    for t in drainage_templates:
        issue_lower = t.issue.lower()
        for kw in road_keywords:
            assert kw not in issue_lower, f"Drainage template {t.family_id} contains road keyword '{kw}': {t.issue}"


def test_family_diversity_per_category_at_least_12() -> None:
    """Verify that ALL_SCENARIO_TEMPLATES provides at least 12 independent families per category."""
    from scripts.generate_m3_dataset import ALL_SCENARIO_TEMPLATES, CATEGORY_ANCHOR_LEXICONS

    assert len(ALL_SCENARIO_TEMPLATES) >= 72
    family_ids = [t.family_id for t in ALL_SCENARIO_TEMPLATES]
    assert len(family_ids) == len(set(family_ids)), "Family IDs must be strictly unique"

    for cat in CATEGORY_ANCHOR_LEXICONS:
        cat_templates = [t for t in ALL_SCENARIO_TEMPLATES if t.category == cat]
        assert (
            len(cat_templates) >= 12
        ), f"Category {cat} has only {len(cat_templates)} families, expected >= 12"


def test_semantic_category_anchor_lexicons_no_enum_leaks() -> None:
    """Verify that category anchor lexicons are respected without direct enum or ID leaks."""
    from scripts.generate_m3_dataset import (
        ALL_SCENARIO_TEMPLATES,
        CATEGORY_ANCHOR_LEXICONS,
        validate_scenario_template_lexicon,
    )

    for cat in CATEGORY_ANCHOR_LEXICONS:
        assert cat in CATEGORY_ANCHOR_LEXICONS, f"Missing lexicon entry for {cat}"
        assert len(CATEGORY_ANCHOR_LEXICONS[cat]["anchors"]) > 0

    for template in ALL_SCENARIO_TEMPLATES:
        assert (
            validate_scenario_template_lexicon(template) is True
        ), f"Template {template.family_id} failed lexicon validation: {template.issue}"

        issue_lower = template.issue.lower()
        landmark_lower = template.landmark.lower()
        for c in Category:
            assert (
                c.value.lower() not in issue_lower
            ), f"Category enum {c.value} leaked into issue of {template.family_id}"
            assert (
                c.value.lower() not in landmark_lower
            ), f"Category enum {c.value} leaked into landmark of {template.family_id}"


def test_compact_ner_object_location_time_evidence_presence() -> None:
    """Verify that generated trajectories yield compact OBJ, LOC, and TIME NER entity evidence."""
    from scripts.generate_m3_dataset import ALL_SCENARIO_TEMPLATES, generate_safe_trajectory
    from scripts.train_ner import extract_trajectory_ner_samples

    trajectories = []
    for i, tmpl in enumerate(ALL_SCENARIO_TEMPLATES):
        traj = generate_safe_trajectory(
            template=tmpl,
            scenario_id=f"sc-ner-evidence-{i:04d}",
            persona="STANDARD",
            noise_level="LOW",
            multi_turn=True,
            seed=42 + i,
            split=DatasetSplit.TRAIN,
            intent="COMPLAINT",
            risk="HIGH",
            completeness="SUFFICIENT",
        )
        trajectories.append(traj)

    samples = extract_trajectory_ner_samples(trajectories)
    assert len(samples) > 0

    all_labels = {
        lbl
        for s in samples
        for _, _, lbl in s.get("spans", [])
    }
    assert "LOC" in all_labels, "Extracted NER samples must contain LOC entities"
    assert "OBJ" in all_labels, "Extracted NER samples must contain OBJ entities"
    assert "TIME" in all_labels, "Extracted NER samples must contain TIME entities"


def test_dataset_generation_preserves_zero_leakage_across_seeds() -> None:
    """Verify that dataset generation maintains zero audit violations across different seeds."""
    from scripts.generate_m3_dataset import generate_and_export_m3_dataset

    for seed in (42, 101, 2024):
        with tempfile.TemporaryDirectory() as td:
            res = generate_and_export_m3_dataset(
                output_dir=td,
                seed=seed,
                num_scenarios=72,
                corpus_target_lines=200,
                audit=True,
            )
            assert res["status"] == "SUCCESS"


def test_category_anchor_composition_injection_in_reporter_bubbles() -> None:
    """Verify that every trajectory has category-specific anchor composition in reporter bubbles."""
    from scripts.generate_m3_dataset import (
        ALL_SCENARIO_TEMPLATES,
        CATEGORY_ANCHOR_LEXICONS,
        generate_safe_trajectory,
        build_family_split_map,
    )

    family_to_cat = {t.family_id: t.category for t in ALL_SCENARIO_TEMPLATES}
    split_map = build_family_split_map(tuple(family_to_cat.keys()), seed=42, family_to_category=family_to_cat)

    drainage_cues = ("drainase", "parit", "gorong", "limpasan", "banjir", "saluran", "genangan", "luapan")
    waste_cues = ("sampah", "limbah", "kotoran", "residu")
    clean_water_cues = ("air bersih", "pdam", "kran", "leding", "perpipaan", "meteran")

    for i, tmpl in enumerate(ALL_SCENARIO_TEMPLATES):
        target_split = split_map[tmpl.family_id]
        traj = generate_safe_trajectory(
            template=tmpl,
            scenario_id=f"sc-anchor-inject-{i:04d}",
            persona="STANDARD",
            noise_level="LOW",
            multi_turn=False,
            seed=42 + i,
            split=target_split,
            intent="COMPLAINT",
            risk="MEDIUM",
            completeness="SUFFICIENT",
        )

        b1_text = traj.turns[0].bubbles[0].text.lower()
        anchors = CATEGORY_ANCHOR_LEXICONS[tmpl.category]["anchors"]
        assert any(
            a.lower() in b1_text for a in anchors
        ), f"Trajectory for {tmpl.category} missing category compact object anchor in bubble: {b1_text}"

        # Contrast drainage vs waste vs clean water
        if tmpl.category == Category.DRAINAGE_FLOOD:
            assert any(c in b1_text for c in drainage_cues), f"Drainage cue missing in: {b1_text}"
        elif tmpl.category == Category.WASTE:
            assert any(c in b1_text for c in waste_cues), f"Waste cue missing in: {b1_text}"
        elif tmpl.category == Category.CLEAN_WATER:
            assert any(c in b1_text for c in clean_water_cues), f"Clean water cue missing in: {b1_text}"

        # No IDs or Category enum in bubble text
        assert f"sc-anchor-inject-{i:04d}" not in b1_text
        assert tmpl.family_id not in b1_text
        assert "drainage_flood" not in b1_text
        assert "clean_water" not in b1_text
        assert "civil_admin" not in b1_text
        assert "health_service" not in b1_text


def test_city12_load_trajectories_with_world_truth_metadata() -> None:
    raw_city12_record = {
        "scenario_id": "city12_train_00_01_00",
        "family_id": "city12-fam-tr-00-01",
        "split": "train",
        "category": "PUBLIC_ORDER",
        "concept_id": 7,
        "risk_evidence_kind": "safety_hazard",
        "style_family": "bureaucratic",
        "world_truth": {
            "category": "PUBLIC_ORDER",
            "risk": "HIGH",
            "domain": "PUBLIC_ORDER",
            "provenance": "synthetic_hifi",
            "canonical_spans": [[0, 16, "OBJ"], [20, 39, "LOC"], [40, 49, "TIME"]],
            "bubble_canonical_spans": {
                "msg_city12_001": [[0, 16, "OBJ"], [20, 39, "LOC"], [40, 49, "TIME"]]
            },
        },
        "observable_facts": ["juru parkir liar", "pasar simpang lima", "sore hari"],
        "hidden_facts": [],
        "location_completeness": "COMPLETE",
        "duration": "OBSERVED",
        "claim_certainty": "FIRST_HAND",
        "persona": "CITIZEN",
        "noise": {"style": "informal_whatsapp"},
        "attachment_role": "NONE",
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg_city12_001",
                        "text": "juru parkir liar di pasar simpang lima sore hari meresahkan warga sekitar",
                        "offset_seconds": 0,
                        "canonical_spans": [[0, 16, "OBJ"], [20, 39, "LOC"], [40, 49, "TIME"]],
                    }
                ],
                "observable_facts": ["juru parkir liar", "pasar simpang lima", "sore hari"],
                "hidden_facts": [],
                "expected_action": {
                    "turn": 1,
                    "allowed_actions": ["EXECUTE"],
                    "missing": [],
                    "strategy": "EXECUTE",
                },
            }
        ],
        "expected_action_by_turn": [
            {
                "turn": 1,
                "allowed_actions": ["EXECUTE"],
                "missing": [],
                "strategy": "EXECUTE",
            }
        ],
        "canonical_spans": [[0, 16, "OBJ"], [20, 39, "LOC"], [40, 49, "TIME"]],
    }

    with tempfile.TemporaryDirectory() as td:
        file_path = Path(td) / "city12_sample.jsonl"
        file_path.write_text(json.dumps(raw_city12_record) + "\n", encoding="utf-8")

        trajectories = load_trajectories_from_jsonl(file_path)
        assert len(trajectories) == 1
        traj = trajectories[0]
        assert isinstance(traj, ComplaintTrajectory)
        assert traj.scenario_id == "city12_train_00_01_00"
        assert traj.family_id == "city12-fam-tr-00-01"
        assert traj.category == Category.PUBLIC_ORDER
        # Verify world_truth metadata preserved
        assert traj.world_truth["concept_id"] == 7
        assert traj.world_truth["risk_evidence_kind"] == "safety_hazard"
        assert traj.world_truth["style_family"] == "bureaucratic"
        assert traj.world_truth["domain"] == "PUBLIC_ORDER"
        assert traj.world_truth["risk"] == "HIGH"
        # Verify canonical spans preserved
        assert len(traj.canonical_spans) == 3
        assert traj.canonical_spans[0].label.value == "OBJ"
        assert traj.canonical_spans[1].label.value == "LOC"
        assert traj.canonical_spans[2].label.value == "TIME"
        assert len(traj.turns[0].bubbles[0].canonical_spans) == 3
        assert "canonical_spans" in traj.world_truth
        assert "bubble_canonical_spans" in traj.world_truth


def test_city12_load_trajectories_rejects_unsafe_top_level_extras() -> None:
    raw_record_with_unsafe_top_level = {
        "scenario_id": "city12_sc_unsafe_01",
        "family_id": "city12-fam-unsafe-01",
        "split": "train",
        "category": "ROAD",
        "concept_id": 1,
        "risk_evidence_kind": "safety_hazard",
        "style_family": "bureaucratic",
        "unsafe_injected_top_level": "malicious_payload",
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg_01",
                        "text": "Laporan aspal bolong di jalan raya",
                    }
                ],
                "expected_action": {
                    "turn": 1,
                    "allowed_actions": ["EXECUTE"],
                },
            }
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        file_path = Path(td) / "unsafe_top.jsonl"
        file_path.write_text(json.dumps(raw_record_with_unsafe_top_level) + "\n", encoding="utf-8")

        with pytest.raises(ValidationError) as exc_info:
            load_trajectories_from_jsonl(file_path)
        assert "unsafe_injected_top_level" in str(exc_info.value)
        assert "extra_forbidden" in str(exc_info.value)


def test_city12_load_trajectories_rejects_unsafe_bubble_extras() -> None:
    raw_record_with_unsafe_bubble = {
        "scenario_id": "city12_sc_unsafe_02",
        "family_id": "city12-fam-unsafe-02",
        "split": "train",
        "category": "ROAD",
        "concept_id": 2,
        "risk_evidence_kind": "cosmetic_minor",
        "style_family": "formal",
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg_02",
                        "text": "Laporan aspal bolong di jalan raya",
                        "unsafe_injected_bubble_field": "unauthorized",
                    }
                ],
                "expected_action": {
                    "turn": 1,
                    "allowed_actions": ["EXECUTE"],
                },
            }
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        file_path = Path(td) / "unsafe_bubble.jsonl"
        file_path.write_text(json.dumps(raw_record_with_unsafe_bubble) + "\n", encoding="utf-8")

        with pytest.raises(ValidationError) as exc_info:
            load_trajectories_from_jsonl(file_path)
        assert "unsafe_injected_bubble_field" in str(exc_info.value)
        assert "extra_forbidden" in str(exc_info.value)


def test_city12_load_trajectories_preserves_canonical_spans_when_nested_in_world_truth() -> None:
    raw_record = {
        "scenario_id": "city12_sc_nested_01",
        "family_id": "city12-fam-nested-01",
        "split": "dev",
        "category": "DRAINAGE_FLOOD",
        "world_truth": {
            "concept_id": 3,
            "risk_evidence_kind": "operational_disruption",
            "style_family": "narrative_slang",
            "canonical_spans": [[0, 12, "OBJ"], [16, 28, "LOC"]],
        },
        "turns": [
            {
                "turn": 1,
                "bubbles": [
                    {
                        "source_message_id": "msg_nested_01",
                        "text": "gorong-gorong mampet di jalan sudirman",
                    }
                ],
                "expected_action": {
                    "turn": 1,
                    "allowed_actions": ["EXECUTE"],
                },
            }
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        file_path = Path(td) / "nested_spans.jsonl"
        file_path.write_text(json.dumps(raw_record) + "\n", encoding="utf-8")

        trajs = load_trajectories_from_jsonl(file_path)
        assert len(trajs) == 1
        traj = trajs[0]
        assert len(traj.canonical_spans) == 2
        assert traj.canonical_spans[0].label.value == "OBJ"
        assert traj.canonical_spans[1].label.value == "LOC"
        assert traj.world_truth["concept_id"] == 3
        assert traj.world_truth["risk_evidence_kind"] == "operational_disruption"
        assert traj.world_truth["style_family"] == "narrative_slang"




