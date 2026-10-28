"""How much of what HuBERT hears is already readable from the Stage 1 ear?

Decides whether distilling a textless SSL encoder into the ear is worth a run.
Single LibriSpeech utterances (Mimi codes) are decoded back to audio and fed to
HuBERT base, which is self-supervised on audio only (never the CTC-finetuned
variant). Its layer features are pooled to Mimi's 12.5 Hz and clustered into
units on the train side. Linear probes, fitted on train speakers and scored on
held-out speakers, then read two targets from several feature sets:

- ``unit@lag``: the HuBERT unit of frame t−lag from the feature at frame t.
  HuBERT is bidirectional, so a small lag gives the causal side the context
  HuBERT used.
- ``segment_r2``: the mean HuBERT feature of the last ~2 s from the feature at
  frame t (ridge regression, R² on held-out speakers). This needs the content
  held over time, not just the sound of the current frame.

Feature sets: Mimi code embeddings of frame t alone (``codes_now``), of the
last four frames (``codes_window``, a local 320 ms baseline), and the ear
output of each checkpoint, plus a randomly initialised ear. If the ear beats
the local window on ``segment_r2``, it already integrates content and the
problem lies after the ear; if it does not, an ear distillation target is the
missing piece.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid

SEGMENT_FRAMES = 25
WINDOW_FRAMES = 4
HUBERT_PER_FRAME = 4


def load_utterances(roots: list[Path], count: int, max_frames: int, seed: int):
    """Real speech codes and activity of single-stream utterances."""
    rows = []
    for root in roots:
        for line in (root / "manifest.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append((root, row))
    rng = random.Random(seed)
    rng.shuffle(rows)
    utterances = []
    for root, row in rows:
        if len(utterances) == count:
            break
        cached = torch.load(root / row["tokens_path"], weights_only=True)
        if row.get("source", "1a_on_A") == "1a_on_B":
            codes, activity = cached["target"], cached["activity"]
        else:
            codes, activity = cached["user"], cached["user_activity"]
        speech = activity.nonzero()
        if len(speech) < SEGMENT_FRAMES * 2:
            continue
        start = int(speech[0])
        end = min(int(speech[-1]) + 1, start + max_frames)
        utterances.append(
            {
                "id": str(row["id"]),
                "speaker": str(row.get("speaker") or row["user_voice"]),
                "codes": codes[:, start:end].long(),
                "activity": activity[start:end].bool(),
            }
        )
    return utterances


@torch.inference_mode()
def hubert_frames(utterances, codec, hubert, layer: int) -> None:
    """Attach HuBERT layer features pooled to Mimi frames, [frames, dim]."""
    from scipy.signal import resample_poly

    for item in utterances:
        pcm = decode(codec, item["codes"][None].cuda())[0, 0].float().cpu().numpy()
        wav = torch.from_numpy(resample_poly(pcm, 2, 3)).float().cuda()
        wav = (wav - wav.mean()) / wav.std().clamp_min(1e-5)
        hidden = hubert(wav[None], output_hidden_states=True).hidden_states[layer][0]
        frames = item["codes"].shape[-1]
        need = frames * HUBERT_PER_FRAME
        if hidden.shape[0] < need:
            pad = hidden[-1:].expand(need - hidden.shape[0], -1)
            hidden = torch.cat((hidden, pad))
        pooled = hidden[:need].reshape(frames, HUBERT_PER_FRAME, -1).mean(1)
        item["hubert"] = pooled.float().cpu()


def kmeans(x: torch.Tensor, clusters: int, iterations: int = 25, seed: int = 0):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    centres = x[torch.randperm(len(x), generator=generator, device=x.device)[:clusters]]
    for _ in range(iterations):
        assign = torch.cdist(x, centres).argmin(1)
        sums = torch.zeros_like(centres).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=clusters).clamp_min(1)
        centres = torch.where(
            (torch.bincount(assign, minlength=clusters) > 0)[:, None],
            sums / counts[:, None],
            centres,
        )
    return centres


def assign_units(utterances, centres: torch.Tensor) -> None:
    for item in utterances:
        item["units"] = torch.cdist(item["hubert"].cuda(), centres).argmin(1).cpu()


def segment_means(hubert: torch.Tensor) -> torch.Tensor:
    """Mean of the last SEGMENT_FRAMES frames at each t (rows t < 24 unused)."""
    total = torch.cat((torch.zeros_like(hubert[:1]), hubert.cumsum(0)))
    end = torch.arange(1, len(hubert) + 1)
    start = (end - SEGMENT_FRAMES).clamp_min(0)
    return (total[end] - total[start]) / SEGMENT_FRAMES


@torch.inference_mode()
def feature_sets(utterances, models: dict[str, DauysHybrid], embedder: DauysHybrid):
    """Per utterance, per set: [frames, dim] features on the CPU."""
    for item in utterances:
        codes = item["codes"][None].cuda()
        sets = {}
        now = embedder._embed(codes, 0)[0].float()
        sets["codes_now"] = now
        padded = torch.cat((now[:1].expand(WINDOW_FRAMES - 1, -1), now))
        sets["codes_window"] = torch.cat(
            [padded[k : k + len(now)] for k in range(WINDOW_FRAMES)], dim=-1
        )
        for name, model in models.items():
            sets[name] = model.encode_listening(codes)[0].float()
        item["features"] = {key: value.half().cpu() for key, value in sets.items()}


def _pairs(utterances, name: str, lag: int):
    xs, ys = [], []
    for item in utterances:
        x = item["features"][name]
        units = item["units"]
        speech = item["activity"]
        if lag >= len(units):
            continue
        keep = speech[: len(units) - lag]
        xs.append(x[lag:][keep])
        ys.append(units[: len(units) - lag][keep])
    return torch.cat(xs).float().cuda(), torch.cat(ys).cuda()


def _segment_pairs(utterances, name: str):
    xs, ys = [], []
    for item in utterances:
        rows = torch.arange(SEGMENT_FRAMES - 1, len(item["units"]))
        xs.append(item["features"][name][rows])
        ys.append(segment_means(item["hubert"])[rows])
    return torch.cat(xs).float().cuda(), torch.cat(ys).float().cuda()


def _standardise(train: torch.Tensor, test: torch.Tensor):
    mean, std = train.mean(0), train.std(0).clamp_min(1e-5)
    return (train - mean) / std, (test - mean) / std


def unit_probe(train_x, train_y, test_x, test_y, classes: int, epochs: int = 8):
    train_x, test_x = _standardise(train_x, test_x)
    head = torch.nn.Linear(train_x.shape[1], classes).cuda()
    optimiser = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(epochs):
        order = torch.randperm(len(train_x), generator=generator, device="cuda")
        for batch in order.split(4096):
            loss = F.cross_entropy(head(train_x[batch]), train_y[batch])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    with torch.no_grad():
        return float((head(test_x).argmax(1) == test_y).float().mean())


def ridge_r2(train_x, train_y, test_x, test_y, alpha: float = 10.0) -> float:
    train_x, test_x = _standardise(train_x, test_x)
    train_y, test_y = _standardise(train_y, test_y)
    ones = torch.ones(len(train_x), 1, device=train_x.device)
    a = torch.cat((train_x, ones), 1).double()
    gram = a.T @ a + alpha * torch.eye(a.shape[1], device=a.device, dtype=a.dtype)
    weights = torch.linalg.solve(gram, a.T @ train_y.double())
    test_a = torch.cat((test_x, torch.ones(len(test_x), 1, device=test_x.device)), 1)
    prediction = test_a.double() @ weights
    residual = (prediction - test_y.double()).pow(2).sum()
    total = (test_y.double() - test_y.double().mean(0)).pow(2).sum()
    return float(1 - residual / total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--train-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    parser.add_argument("--train-items", type=int, default=1500)
    parser.add_argument("--eval-items", type=int, default=500)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--hubert", default="facebook/hubert-base-ls960")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--clusters", type=int, default=200)
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if "ft" in args.hubert.split("/")[-1]:
        raise ValueError("Use a self-supervised HuBERT, not a text-finetuned one")
    started = time.time()
    torch.manual_seed(args.seed)

    train = load_utterances(
        args.train_roots, args.train_items, args.max_frames, args.seed
    )
    test = load_utterances(args.eval_roots, args.eval_items, args.max_frames, args.seed)
    overlap = {u["speaker"] for u in train} & {u["speaker"] for u in test}
    if overlap:
        raise ValueError(f"Probe speakers overlap: {sorted(overlap)[:5]}")
    print(f"utterances train {len(train)} eval {len(test)}", flush=True)

    from transformers import HubertModel

    codec = load_mimi(args.mimi_checkpoint, "cuda")
    hubert = HubertModel.from_pretrained(args.hubert).cuda().eval()
    hubert_frames(train + test, codec, hubert, args.layer)
    del codec, hubert
    torch.cuda.empty_cache()
    print(f"hubert done {time.time() - started:.0f}s", flush=True)

    speech = torch.cat([u["hubert"][u["activity"]] for u in train]).cuda()
    centres = kmeans(speech, args.clusters, seed=args.seed)
    assign_units(train + test, centres)
    counts = torch.bincount(
        torch.cat([u["units"][u["activity"]] for u in test]), minlength=args.clusters
    )
    majority = float(counts.max() / counts.sum())

    config = ModelConfig.from_yaml(args.config)
    models = {}
    random_model = DauysHybrid(config).cuda().eval()
    models["ear_random"] = random_model
    embedder = None
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        model = DauysHybrid(config)
        model.load_state_dict(load_research_checkpoint(Path(path))["model"])
        models[f"ear_{name}"] = model.cuda().eval()
        embedder = embedder or models[f"ear_{name}"]
    feature_sets(train + test, models, embedder)
    del models, random_model
    torch.cuda.empty_cache()
    print(f"features done {time.time() - started:.0f}s", flush=True)

    rows = []
    for name in train[0]["features"]:
        row = {"features": name}
        for lag in args.lags:
            row[f"unit@{lag}"] = round(
                unit_probe(
                    *_pairs(train, name, lag), *_pairs(test, name, lag), args.clusters
                ),
                4,
            )
        row["segment_r2"] = round(
            ridge_r2(*_segment_pairs(train, name), *_segment_pairs(test, name)), 4
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
    summary = {
        "hubert": args.hubert,
        "layer": args.layer,
        "clusters": args.clusters,
        "majority_unit": round(majority, 4),
        "train_utterances": len(train),
        "eval_utterances": len(test),
        "seconds": round(time.time() - started),
        "probes": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in summary if key != "probes"}))


if __name__ == "__main__":
    main()
