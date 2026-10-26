"""Data partition invariants for multi-excerpt research."""

from dauys.research_split import stage1_selections


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
