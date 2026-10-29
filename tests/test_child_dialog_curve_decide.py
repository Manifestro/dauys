"""Decision-rule checks for the child-dialogue scaling curve."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from child_dialog_curve_decide import (
    compute_metrics,
    decide,
    paired_bootstrap,
    parse_name,
)


def _items(pattern: list[bool], prefix: str = "i") -> list[dict]:
    return [
        {"id": f"{prefix}{index}", "top1": value} for index, value in enumerate(pattern)
    ]


def test_decide_yes_and_missing_spreads_count_as_zero():
    decision, failed = decide(
        predicted_2000=0.05,
        ci_low=0.01,
        G=0.04,
        seed_spread_h200=None,
        seed_spread_h500=0.02,
    )
    assert decision == "YES"
    assert failed == []


def test_decide_no_by_small_gain():
    decision, failed = decide(
        predicted_2000=0.02,
        ci_low=0.01,
        G=0.04,
        seed_spread_h200=0.0,
        seed_spread_h500=0.0,
    )
    assert decision == "NO"
    assert failed == ["predicted_2000_below_0.03"]


def test_decide_no_by_bootstrap_ci():
    decision, failed = decide(
        predicted_2000=0.05,
        ci_low=0.0,
        G=0.04,
        seed_spread_h200=None,
        seed_spread_h500=None,
    )
    assert decision == "NO"
    assert failed == ["bootstrap_ci_low_not_positive"]


def test_decide_no_by_seed_spread():
    decision, failed = decide(
        predicted_2000=0.05,
        ci_low=0.01,
        G=0.04,
        seed_spread_h200=0.04,
        seed_spread_h500=0.02,
    )
    assert decision == "NO"
    assert failed == ["gain_not_above_seed_spread"]


def test_compute_metrics_yes_from_paired_items():
    zero = [False] * 10
    full = [True] * 10
    half = [True] * 5 + [False] * 5
    piper = {
        "H200-s0": _items(zero),
        "H200-s1": _items(zero),
        "H500-s0": _items(full),
        "H500-s1": _items(full),
    }
    kokoro = {
        "H200-s0": _items(half),
        "H200-s1": _items(half),
        "H500-s0": _items(half),
        "H500-s1": _items(half),
    }
    result = compute_metrics(piper, kokoro, samples=2000, seed=0)
    assert result["by_hours"]["200"]["piper"] == 0.0
    assert result["by_hours"]["500"]["piper"] == 1.0
    assert result["gain"]["G"] == pytest.approx(1.0)
    assert result["gain"]["predicted_2000"] == pytest.approx(2 / 1.321928)
    assert result["seed_spread"] == {"H200": 0.0, "H500": 0.0}
    assert result["bootstrap"]["ci_low"] == pytest.approx(1.0)
    assert result["decision"] == "YES"
    assert result["conditions_failed"] == []
    assert result["memorization_flag"] is False


def test_piper_and_kokoro_may_use_different_item_ids():
    zero = [False] * 10
    full = [True] * 10
    half = [True] * 5 + [False] * 5
    piper = {
        "H200-s0": _items(zero, "p"),
        "H200-s1": _items(zero, "p"),
        "H500-s0": _items(full, "p"),
        "H500-s1": _items(full, "p"),
    }
    kokoro = {
        "H200-s0": _items(half, "k"),
        "H200-s1": _items(half, "k"),
        "H500-s0": _items(half, "k"),
        "H500-s1": _items(half, "k"),
    }
    result = compute_metrics(piper, kokoro, samples=200)
    assert result["runs"]["H200-s0"]["piper"]["items"] == 10
    assert result["runs"]["H200-s0"]["kokoro"]["items"] == 10
    assert result["decision"] == "YES"


def test_memorization_flag_does_not_change_the_decision():
    flat = _items([True] * 5 + [False] * 5)
    learned = _items([True] * 9 + [False] * 1)
    piper = {
        "H200-s0": flat,
        "H500-s0": flat,
    }
    kokoro = {
        "H200-s0": flat,
        "H500-s0": learned,
    }
    result = compute_metrics(piper, kokoro, samples=100, seed=0)
    assert result["gain"]["G"] == 0.0
    assert result["memorization_flag"] is True
    assert result["decision"] == "NO"


def test_paired_bootstrap_is_deterministic():
    first = paired_bootstrap([0.5] * 10, samples=100, seed=0)
    second = paired_bootstrap([0.5] * 10, samples=100, seed=0)
    assert first == second
    assert first["mean"] == pytest.approx(0.5)
    assert first["ci_low"] == pytest.approx(0.5)
    assert first["ci_high"] == pytest.approx(0.5)


def test_parse_name_rejects_unknown_shapes():
    assert parse_name("H500-s1") == (500.0, 1)
    with pytest.raises(ValueError, match="Invalid"):
        parse_name("500")
