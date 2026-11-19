"""DY-35 evaluation pack "before an utterance", built by rule from the held-out shards before training.

Candidates: utterances of 4-15 s with >= 15 s of their piece before them, having >= 16 same-speaker same-recording
utterances >= 60 s away from them (and from the anchor) with length within x2. Episodes (shard, ep) in sorted order;
inside an episode the candidates in a seeded random order (seed 1234); round-robin over episodes, skipping a
candidate whose prompt overlaps an already chosen prompt or utterance of that episode, until 200.
For each: prompt_NNN.wav (the 15 s before the utterance start), real_NNN.wav (the utterance, cut at 10 s),
swap_NNN.wav (the first of the 16 eligible in the seeded order, cut at 10 s), 15 decoys (the next 15),
codes.npz (alone-encoded codes of real / swap / decoys from dy35_utt.py), manifest.json. Also silence.json: the silence
codes of the K2000 codebook (median frame energy <= --sil-db dBFS over the held-out shards) used by the pause rule.

    python scripts/dy35_pack.py --utt /opt/dlami/nvme/utt --audio /opt/dlami/nvme/held --out /data/t/prompts_turn
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy35_train import GAP_S, UMAX, UMIN

SR, HOP = 16000, 320


def silence_codes(files: list[Path], thr: float) -> dict:
    tok = np.concatenate([np.load(f)["tok"] for f in files]).astype(np.int64)
    db = np.concatenate([np.load(f)["edb"] for f in files]).astype(np.float64)
    cnt = np.bincount(tok, minlength=2000)
    med = np.array([np.median(db[tok == k]) if cnt[k] else 0.0 for k in range(2000)])
    codes = np.flatnonzero((med <= thr) & (cnt > 0))
    return {"rule": f"codes whose median frame RMS energy over the held-out shards is <= {thr} dBFS", "thr_db": thr,
            "codes": codes.tolist(), "n": int(len(codes)), "frame_share": float(np.isin(tok, codes).mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--utt", required=True)
    p.add_argument("--audio", required=True, help="held-out shard folders with index.npy, segs.bin, pieces.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sil-db", type=float, default=-45.0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in Path(a.utt).glob("*.npz") if held_out(f.stem))
    sil = silence_codes(files, a.sil_db)
    (out / "silence.json").write_text(json.dumps(sil))
    rng = np.random.default_rng(a.seed)
    eps = []  # list of dicts per episode with candidate lists
    data = {}
    for f in files:
        z = np.load(f)
        data[f.stem] = z
        ep_names = json.loads(str(z["eps"]))
        L = z["seg_e"] - z["seg_s"]
        m = np.flatnonzero((L >= UMIN) & (L <= UMAX))
        pe = z["piece_ep"][z["seg_piece"][m]]
        t0 = z["piece_t0"][z["seg_piece"][m]] + z["seg_s"][m] / 50
        t1 = z["piece_t0"][z["seg_piece"][m]] + z["seg_e"][m] / 50
        spk = z["seg_spk"][m]
        for ei in range(len(ep_names)):
            sel = np.flatnonzero(pe == ei)
            cands = []
            for i in sel:
                Li = L[m[i]]
                if not (200 <= Li <= 750 and z["seg_s"][m[i]] >= 750):
                    continue
                same = sel[(spk[sel] == spk[i]) & (sel != i)]
                far = (t1[same] <= t0[i] - GAP_S) | (t0[same] >= t1[i] + GAP_S)
                r = L[m[same]] / Li
                el = same[far & (r >= 0.5) & (r <= 2.0)]
                if len(el) >= 16:
                    cands.append((int(m[i]), [int(m[j]) for j in el]))
            if cands:
                order = rng.permutation(len(cands))
                eps.append({"shard": f.stem, "ep": ep_names[ei], "cands": [cands[o] for o in order]})
    chosen, ptr, used = [], [0] * len(eps), [[] for _ in eps]
    while len(chosen) < a.n and any(ptr[e] < len(eps[e]["cands"]) for e in range(len(eps))):
        for e in range(len(eps)):
            if len(chosen) >= a.n:
                break
            while ptr[e] < len(eps[e]["cands"]):
                si, el = eps[e]["cands"][ptr[e]]
                ptr[e] += 1
                z = data[eps[e]["shard"]]
                k = int(z["seg_piece"][si])
                t0 = z["piece_t0"][k] + z["seg_s"][si] / 50
                iv = (t0 - 15, z["piece_t0"][k] + z["seg_e"][si] / 50)
                if any(not (iv[1] <= u[0] or iv[0] >= u[1]) for u in used[e]):
                    continue
                used[e].append(iv)
                el = [el[j] for j in rng.permutation(len(el))]
                chosen.append((e, si, el[0], el[1:16]))
                break
    print("chosen", len(chosen), "episodes", len(eps), flush=True)
    audio = {}

    def wav(shard: str, piece: int, s: int, e: int) -> np.ndarray:
        if shard not in audio:
            d = Path(a.audio) / shard
            audio[shard] = (np.memmap(d / "segs.bin", dtype=np.int16, mode="r"), np.load(d / "index.npy"))
        segs, idx = audio[shard]
        z = data[shard]
        st, ln = idx[int(z["piece_orig"][piece])]
        assert ln // HOP == z["index"][piece][1]
        return np.asarray(segs[st + s : st + e], dtype=np.int16)

    man, codes = [], {}
    for n, (e, si, sw, dec) in enumerate(chosen):
        shard = eps[e]["shard"]
        z = data[shard]
        k = int(z["seg_piece"][si])
        s, en = int(z["seg_s"][si]), int(z["seg_e"][si])
        off = int(z["index"][k][0])
        sf.write(out / f"prompt_{n:03d}.wav", wav(shard, k, (s - 750) * HOP, s * HOP), SR)
        sf.write(out / f"real_{n:03d}.wav", wav(shard, k, s * HOP, min(en, s + 500) * HOP), SR)
        ks, ss, es = int(z["seg_piece"][sw]), int(z["seg_s"][sw]), int(z["seg_e"][sw])
        sf.write(out / f"swap_{n:03d}.wav", wav(shard, ks, ss * HOP, min(es, ss + 500) * HOP), SR)
        utt = lambda i: {"piece": int(z["seg_piece"][i]), "s": int(z["seg_s"][i]), "e": int(z["seg_e"][i]),
                         "t0": float(z["piece_t0"][z["seg_piece"][i]] + z["seg_s"][i] / 50),
                         "t1": float(z["piece_t0"][z["seg_piece"][i]] + z["seg_e"][i] / 50), "spk": int(z["seg_spk"][i])}
        tc = lambda i: z["tok"][int(z["index"][z["seg_piece"][i]][0]) + int(z["seg_s"][i]) : int(z["index"][z["seg_piece"][i]][0]) + int(z["seg_e"][i])]
        codes[f"real_{n:03d}"] = tc(si)
        codes[f"swap_{n:03d}"] = tc(sw)
        codes[f"prompt_stream_{n:03d}"] = z["tok"][off + s - 750 : off + s]
        for j, d in enumerate(dec):
            codes[f"decoy_{n:03d}_{j:02d}"] = tc(d)
        man.append({"id": f"{n:03d}", "shard": shard, "ep": eps[e]["ep"], "piece_orig": int(z["piece_orig"][k]),
                    "real": utt(si), "swap": utt(sw), "decoys": [utt(d) for d in dec],
                    "prompt_s": 15.0, "real_len_s": (en - s) / 50, "real_wav_s": min(en - s, 500) / 50})
    np.savez(out / "codes.npz", **codes)
    (out / "manifest.json").write_text(json.dumps({"rule": __doc__, "seed": a.seed, "n": len(man), "items": man}, indent=1))
    lens = np.array([m["real_len_s"] for m in man])
    print("PACK DONE", len(man), "episodes used", len({(m['shard'], m['ep']) for m in man}),
          "real len pct", np.percentile(lens, [10, 50, 90]).round(1).tolist(), "silence codes", sil["n"], flush=True)


if __name__ == "__main__":
    main()
