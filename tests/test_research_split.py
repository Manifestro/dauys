"""Data partition invariants for multi-excerpt research."""

from dauys.research_split import (
    leakage_free_source_splits,
    stage1_selections,
    whole_dialogue_generalization_split,
)


def test_leakage_free_selection_keeps_held_out_families():
    train = [
        {"id": "t1", "scenario_family": "shared_validation"},
        {"id": "t2", "scenario_family": "shared_test"},
        {"id": "t3", "scenario_family": "train_only"},
    ]
    validation = [
        {"id": "v1", "scenario_family": "shared_validation"},
        {"id": "v2", "scenario_family": "shared_test"},
    ]
    test = [{"id": "e1", "scenario_family": "shared_test"}]
    selected, excluded = leakage_free_source_splits(train, validation, test)
    assert [row["id"] for row in selected["train"]] == ["t3"]
    assert [row["id"] for row in selected["validation"]] == ["v1"]
    assert [row["id"] for row in selected["test"]] == ["e1"]
    assert excluded == {"train": ["t1", "t2"], "validation": ["v2"]}


def test_splits_are_nested_and_scenario_disjoint():
    rows = []
    for split, count in (("train", 130), ("validation", 20), ("test", 20)):
        for index in range(count):
            rows.append(
                {
                    "id": f"{split}-{index}",
                    "split": split,
                    "scenario": f"{split}/scenario-{index}",
                    "duration_samples": 120 * 24_000,
                }
            )
    selections = stage1_selections(
        {"rows": rows, "dataset": "sample", "revision": "r1", "manifest_sha256": "x"}
    )
    for part in ("train", "validation"):
        outer = {item["id"]: item["frames"] for item in selections["e005"][part]}
        assert all(
            outer[item["id"]] >= item["frames"] for item in selections["e004"][part]
        )
    assert not (
        {item["scenario"] for item in selections["e005"]["train"]}
        & {item["scenario"] for item in selections["e005"]["test"]}
    )


def test_generalization_keeps_whole_scenario_families_disjoint():
    rows = []
    for topic in range(70):
        source = "test" if topic < 10 else "validation" if topic < 20 else "train"
        for style in ("simple_dialogue", "interruption"):
            rows.append(
                {
                    "id": f"{topic}-{style}",
                    "split": source if style == "simple_dialogue" else "train",
                    "scenario": f"{style}/topic_{topic}",
                    "sample_rate": 24_000,
                    "duration_samples": 90 * 24_000 + topic,
                }
            )
    result = whole_dialogue_generalization_split(
        {"rows": rows, "dataset": "sample", "revision": "r1", "manifest_sha256": "x"}
    )["selection"]
    seen = set()
    for split in ("train", "validation", "test"):
        families = {row["scenario_family"] for row in result[split]}
        assert not seen & families
        seen.update(families)
        assert len(result[split]) >= 5
        assert all(
            row["source_samples"] == 90 * 24_000 + int(row["scenario_family"][6:])
            for row in result[split]
        )
    assert any(row["source_split"] == "test" for row in result["test"])
    assert any(row["source_split"] == "validation" for row in result["validation"])
