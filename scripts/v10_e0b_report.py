"""Apply the E0b decision rules to two finished runs and draw the curves.

    python scripts/v10_e0b_report.py --rev2 RUN_REV2 --rev1 RUN_REV1 --output REPORT

Question 1 is only partly checked here: decoded speech and the roles on five clips
are checked by hand, and the script says so.
"""

import argparse
import json
from pathlib import Path

from dauys.v10_e0b_data import TEST_SPLITS, TRAIN_SPLIT
from dauys.v10_e0b_eval import DETECTION_GAP, check_question_1, decide_question_2

KEPT_LEAK_MAX_PCT = 5.0
PLATEAU_CHECKPOINTS = 3
PANELS = (
    ("g_silence", "G-silence"),
    ("g_voice", "G-voice"),
    ("own_history_gain", "Own history gain"),
)
COLORS = {"rev2": "#2a78d6", "rev1": "#eb6834"}
MARKERS = {"rev2": "o", "rev1": "s"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_run(root: Path) -> dict:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    evals = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "eval").glob("step-*.json"))
    ]
    if not evals:
        raise ValueError(f"{root} has no scored checkpoints")
    with (root / "metrics.jsonl").open(encoding="utf-8") as handle:
        losses = [json.loads(line) for line in handle]
    return {"metadata": metadata, "evals": evals, "losses": losses}


def curve(run: dict, name: str, split: str = TRAIN_SPLIT) -> list[float]:
    return [scores[split][name]["mean"] for scores in run["evals"]]


def plateaued_at(run: dict) -> int | None:
    """The step where training NLL stopped improving for three checkpoints, if any."""
    losses = run["losses"]
    steps = [scores["step"] for scores in run["evals"]]
    means = []
    for step in steps:
        recent = [
            row["train/nll_cb0_own"]
            for row in losses
            if step - steps[0] < row["step"] <= step
        ]
        means.append(sum(recent) / len(recent) if recent else float("inf"))
    best = float("inf")
    since = 0
    for step, value in zip(steps, means):
        since = 0 if value < best else since + 1
        best = min(best, value)
        if since >= PLATEAU_CHECKPOINTS:
            return step
    return None


def final_table(run: dict) -> dict:
    last = run["evals"][-1]
    return {
        split: {
            "places": last[split]["places"],
            "nll_real": last[split]["nll_real"],
            "g_silence": last[split]["g_silence"],
            "g_voice": last[split]["g_voice"],
            "own_history_gain": last[split]["own_history_gain"],
            "shift_change": last[split]["shift_change"],
            "echo_percent": last[split]["echo_percent"],
            "echo_percent_data": last[split]["echo_percent_data"],
            "g_silence_by_feed": last[split]["g_silence_by_feed"],
            "g_silence_leak_2_to_5": last[split]["g_silence_leak_2_to_5"],
            "g_silence_leak_under_2": last[split]["g_silence_leak_under_2"],
        }
        for split in (TRAIN_SPLIT, *TEST_SPLITS)
    }


def question_1(run: dict) -> dict:
    last = run["evals"][-1]
    losses = run["losses"]
    return check_question_1(
        train_loss_first=losses[0]["train/nll_cb0_own"],
        train_loss_last=losses[-1]["train/nll_cb0_own"],
        echo_percent=last[TRAIN_SPLIT]["echo_percent"],
        echo_baseline_percent=run["metadata"]["echo_baseline"]["percent"],
        shift_change=last[TRAIN_SPLIT]["shift_change"]["mean"],
        causality=max(scores["causality_error"] for scores in run["evals"]),
        kept_leak_max_pct=KEPT_LEAK_MAX_PCT,
    )


def draw(runs: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axes = pyplot.subplots(1, len(PANELS), figsize=(13, 4), sharex=True)
    for axis, (name, title) in zip(axes, PANELS):
        for scheme, run in runs.items():
            steps = [scores["step"] for scores in run["evals"]]
            values = curve(run, name)
            axis.plot(
                steps,
                values,
                color=COLORS[scheme],
                marker=MARKERS[scheme],
                markersize=5,
                linewidth=2,
                label=scheme,
            )
            axis.annotate(
                scheme,
                (steps[-1], values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=INK,
                fontsize=9,
                va="center",
            )
        if name == "g_silence":
            axis.axhline(DETECTION_GAP, color=MUTED, linewidth=1, linestyle=":")
        axis.axhline(0, color=MUTED, linewidth=1)
        axis.set_title(title, color=INK, fontsize=11, loc="left")
        axis.set_xlabel("step", color=MUTED)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID)
    axes[0].set_ylabel("nats per frame, training places", color=MUTED)
    axes[0].legend(frameon=False, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor="#fcfcfb")
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev2", type=Path, required=True)
    parser.add_argument("--rev1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {"rev2": load_run(args.rev2), "rev1": load_run(args.rev1)}
    steps = {
        scheme: [scores["step"] for scores in run["evals"]]
        for scheme, run in runs.items()
    }
    if steps["rev2"] != steps["rev1"]:
        raise ValueError("The two runs were not scored at the same steps")
    decision = decide_question_2(
        curve(runs["rev2"], "g_silence"),
        curve(runs["rev1"], "g_silence"),
        curve(runs["rev2"], "own_history_gain"),
        curve(runs["rev1"], "own_history_gain"),
    )
    report = {
        "question_1": {
            **{scheme: question_1(run) for scheme, run in runs.items()},
            "by_hand": [
                "decoded speech is of the right channel",
                "roles on five clips",
                "ten clips and five minutes through Mimi listened to",
            ],
        },
        "question_2": decision,
        "steps": steps["rev2"],
        "curves": {
            scheme: {
                split: {name: curve(run, name, split) for name, _ in PANELS}
                for split in (TRAIN_SPLIT, *TEST_SPLITS)
            }
            for scheme, run in runs.items()
        },
        "final": {scheme: final_table(run) for scheme, run in runs.items()},
        "voice_over_silence": {
            scheme: (
                table[TRAIN_SPLIT]["g_voice"]["mean"]
                / table[TRAIN_SPLIT]["g_silence"]["mean"]
                if table[TRAIN_SPLIT]["g_silence"]["mean"]
                else None
            )
            for scheme, table in (
                (name, final_table(run)) for name, run in runs.items()
            )
        },
        "first_step_under_nll_level": {
            scheme: run["metadata"]["first_step_under_nll_level"]
            for scheme, run in runs.items()
        },
        "plateau_step": {scheme: plateaued_at(run) for scheme, run in runs.items()},
        "parameters": {
            scheme: run["metadata"]["trainable_parameters"]
            for scheme, run in runs.items()
        },
        "git": {scheme: run["metadata"]["git"] for scheme, run in runs.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    draw(runs, args.output / "curves.png")
    print(json.dumps({k: report[k] for k in ("question_1", "question_2")}, indent=2))


if __name__ == "__main__":
    main()
