"""DY-32 track D: how much did whole-window WavLM encoding (as in training/validation) help the heard part?

Same 1,200 held-out windows as dy32d_heard_use.py (same seed), 15 s true heard part + 10 s own, both modes.
  1 whole: one WavLM pass over the 25 s window; heard feats, heard ids and own ids all from it (as in training);
  2 separate: heard and own encoded alone (as at generation; = condition a of heard_use);
  3 heard alone, own ids from the whole-window pass.
CE of the own units per bucket (0-1, 1-3, 3-10 s, and the 10 s mean); paired 95 % CI of 1-2 and 1-3 per mode.
Results are appended to heard_use.json / heard_use.md (key "leak").

    python scripts/dy32d_leak.py --held /opt/dlami/nvme/heldaudio --run /data/dy32d/run \
        --centroids /data/dy32d/acb/km-corpus-K2000.npy --json /data/dy32d/heard_use.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy32d import Codebook, load_wavlm, wavlm_feats
from dy32d_heard_use import OWN, mean_ci, build_windows, read_chunk
from dauys.dy32_heard import HeardConfig, HeardLM

BUCKETS = {"0-1s": (0, 50), "1-3s": (50, 150), "3-10s": (150, 500), "0-10s": (0, 500)}
P = 750


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--held", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--centroids", required=True)
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--json", required=True)
    a = p.parse_args()
    device = torch.device("cuda")
    pieces, wins, _, _, hs, os_ = build_windows(a.held, a.n, a.seed, 15)
    ck = torch.load(Path(a.run) / "best.pt", map_location="cpu")
    model = HeardLM(HeardConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    cb = Codebook(a.centroids, "cos", device)
    wavlm = load_wavlm(device)

    @torch.no_grad()
    def ce_own(hf, hid, oid, mode):
        B = hid.shape[0]
        ids = torch.cat([hid, oid], 1)
        feats = torch.cat([hf, torch.zeros(B, OWN, hf.shape[2], device=device, dtype=hf.dtype)], 1)
        T = P + OWN
        heard = (torch.arange(T - 1, device=device) < P)[None].expand(B, -1)
        m = torch.full((B,), mode, device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(model.embed(ids[:, : T - 1], feats[:, : T - 1], heard, m))
        lg = model.logits(h[:, P - 1 :])
        return F.cross_entropy(lg.flatten(0, 1), oid.flatten(), reduction="none").view(B, OWN).cpu().numpy()

    res = {(c, m): [] for c in (1, 2, 3) for m in (0, 1)}
    for b0 in range(0, len(wins), a.batch):
        W = wins[b0 : b0 + a.batch]
        whole = np.stack([read_chunk(pieces, i, off, hs + os_) for i, off, *_ in W])
        fw = wavlm_feats(wavlm, torch.from_numpy(whole).to(device))
        assert fw.shape[1] >= P + OWN, fw.shape
        fw = fw[:, : P + OWN]
        iw = cb.ids(fw)
        fh = wavlm_feats(wavlm, torch.from_numpy(whole[:, :hs]).to(device))
        ih = cb.ids(fh)
        io = cb.ids(wavlm_feats(wavlm, torch.from_numpy(whole[:, hs:]).to(device)))
        for m in (0, 1):
            res[(1, m)].append(ce_own(fw[:, :P], iw[:, :P], iw[:, P:], m))
            res[(2, m)].append(ce_own(fh, ih, io, m))
            res[(3, m)].append(ce_own(fh, ih, iw[:, P:], m))
    res = {k: np.concatenate(v) for k, v in res.items()}

    out = {"windows": len(wins), "seed": a.seed, "checkpoint_step": ck["step"],
           "conditions": {"1": "whole-window WavLM pass (as training): heard feats, heard ids, own ids from it",
                          "2": "separate: heard alone, own alone (as generation; heard_use condition a)",
                          "3": "heard alone; own ids from the whole-window pass"}, "results": {}}
    lines = ["", "## Leak check: whole-window vs separate WavLM encoding (true heard part, 15 s + 10 s)", "",
             f"Same {len(wins)} windows. 1 = one WavLM pass over the whole 25 s window (as in training and validation); "
             "2 = heard and own encoded separately (as at generation); 3 = heard encoded alone, own ids from the "
             "whole-window pass.", "",
             "| mode | bucket | CE 1 | CE 2 | CE 3 | 1-2 [95% CI] | 1-3 [95% CI] |", "|---|---|---|---|---|---|---|"]
    for m, mn in ((0, "cont"), (1, "disc")):
        r = {}
        for bk, (s, e) in BUCKETS.items():
            c = {k: res[(k, m)][:, s:e].mean(1) for k in (1, 2, 3)}
            r[bk] = {"ce_1": round(float(c[1].mean()), 4), "ce_2": round(float(c[2].mean()), 4),
                     "ce_3": round(float(c[3].mean()), 4), "d_1_minus_2": mean_ci(c[1] - c[2]),
                     "d_1_minus_3": mean_ci(c[1] - c[3])}
            x, y = r[bk]["d_1_minus_2"], r[bk]["d_1_minus_3"]
            lines.append(f"| {mn} | {bk} | {r[bk]['ce_1']:.3f} | {r[bk]['ce_2']:.3f} | {r[bk]['ce_3']:.3f} | "
                         f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] | "
                         f"{y['mean']:+.3f} [{y['ci95'][0]:+.3f}, {y['ci95'][1]:+.3f}] |")
        out["results"][mn] = r
    j = json.loads(Path(a.json).read_text())
    j["leak"] = out
    Path(a.json).write_text(json.dumps(j, indent=2))
    md = Path(a.json).with_suffix(".md")
    md.write_text(md.read_text() + "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("LEAK DONE", flush=True)


if __name__ == "__main__":
    main()
