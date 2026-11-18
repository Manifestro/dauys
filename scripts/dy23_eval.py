"""DY-23 speech check: gate L of DY-22 on the code LM, plus listening files.

Same protocol as dy22_eval.py: 200 held-out pieces (pauses cut to 2 s), prompt = 300 grid frames, continuation
= 150 frames. ``gen``: event drawn at T 1, the 8 codes of a syllable through the chain at T 0.7, prosody sampled.
``real``: the true next frames. ``floor``: real timing, every syllable replaced by a random held-out syllable.
Rendering: a syllable's 8 codes -> z (group centroids) -> coder decoder with the voice summary of one held-out
reference voice -> Sylber feature (scaled to the DY-14 unit-mean norm) -> Sylber flow matching -> SPARC, the pitch
track from the per-syllable F0 (dy22_eval.render). Rulers as before (Whisper, GPT-2 log-PPL, rate, pause, F0
spread). Gate L: L1 gap closed >= 0.5 and rep4 <= 2 x max(real, 1e-3) (rep4 over the first-group code),
L2 rate within 15 % and pause within 30 %, L3 F0 spread within 30 %.

    python scripts/dy23_eval.py --units /opt/dlami/nvme/units23 --run /data/dy23/run --coder coder.pt \
        --means unit_means.pt --config sylber_resynthesis.yaml --ref ref.wav --out /data/dy23/eval
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy14_features import segments_of
from dy16_voice import SR, load_synth
from dy21_eval import rep4
from dy22_eval import render
from dy23_train import load_grid
from units_vinv import Coder, pool

from dauys.dy14_units import hidden_states, load_sylber
from dauys.dy21_lm import LMConfig
from dauys.dy22_lm import e_of_bin, f0_of_bin
from dauys.dy23_lm import CONT, EOS, SIL, UNIT, CodeLM

PROMPT, GEN, N = 300, 150, 200
K = 8192
T_CODE = 0.7


@torch.no_grad()
def generate(model: CodeLM, ev: np.ndarray, cd: np.ndarray, device, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    e = torch.from_numpy(ev.astype(np.int64)).to(device)
    c = torch.from_numpy(cd.astype(np.int64)).to(device)
    b = len(e)
    pros = torch.zeros(b, GEN + 1, 3, device=device)
    model.eval()
    for step in range(GEN + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(e[:, -model.cfg.ctx :], c[:, -model.cfg.ctx :])[:, -1].float()
        if step:
            pf, pe, pv = model.prosody(h)
            v = torch.multinomial(torch.softmax(pv, -1), 1, generator=g)[:, 0]
            f = f0_of_bin(torch.multinomial(torch.softmax(pf, -1), 1, generator=g)[:, 0])
            en = e_of_bin(torch.multinomial(torch.softmax(pe, -1), 1, generator=g)[:, 0])
            pros[:, step - 1] = torch.stack([f * v, en, v.float()], -1) * (e[:, -1] == UNIT)[:, None]
        if step == GEN:
            break
        lg = model.ev_head(h).float()
        lg[:, EOS] = -float("inf")
        nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=g)[:, 0]
        codes = model.sample_codes(h, T_CODE, g)
        codes = codes * (nxt == UNIT)[:, None]
        e = torch.cat([e, nxt[:, None]], 1)
        c = torch.cat([c, codes[:, None]], 1)
    return e[:, -GEN:].cpu().numpy(), c[:, -GEN:].cpu().numpy(), pros[:, :GEN].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--coder", required=True)
    p.add_argument("--means", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--other", default="", help="eval.json of DY-22 (same prompts): its gen texts get the topic test too")
    p.add_argument("--no-listen", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    (out / "listen").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    meta = json.loads((Path(args.run) / "metadata.json").read_text())
    ev, cd, pb = load_grid([Path(args.units) / f"{n}.npz" for n in meta["val_files_names"]])
    cuts = np.flatnonzero(ev == EOS)
    bounds = [(s, e) for s, e in zip(np.r_[0, cuts[:-1] + 1], cuts) if e - s >= PROMPT + GEN]
    rng = np.random.default_rng(1234)
    pick = rng.choice(len(bounds), size=min(args.n, len(bounds)), replace=False)
    sl = lambda a, o, n: np.stack([a[bounds[i][0] + o : bounds[i][0] + o + n] for i in pick])
    p_ev, p_cd = sl(ev, 0, PROMPT), sl(cd, 0, PROMPT)
    r_ev, r_cd, r_pb = sl(ev, PROMPT, GEN), sl(cd, PROMPT, GEN), sl(pb, PROMPT, GEN)
    pr_pb = sl(pb, 0, PROMPT)

    def unbin(x):
        t = torch.from_numpy(x.astype(np.int64))
        return np.stack([(f0_of_bin(t[..., 0]) * t[..., 2]).numpy(), e_of_bin(t[..., 1]).numpy(), t[..., 2].float().numpy()], -1)

    r_pr, pr_pr = unbin(r_pb), unbin(pr_pb)
    pool_syl = cd[ev == UNIT]
    f_cd = r_cd.copy()
    m = r_ev == UNIT
    f_cd[m] = pool_syl[rng.integers(0, len(pool_syl), int(m.sum()))]

    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=False)
    model = CodeLM(LMConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"], strict=False)  # DY-24 D has an extra future head
    gs = [generate(model, p_ev[i : i + 50], p_cd[i : i + 50], device, 1234 + i) for i in range(0, len(pick), 50)]
    g_ev, g_cd, g_pr = (np.concatenate(x) for x in zip(*gs))
    del model
    torch.cuda.empty_cache()
    print("generated", len(g_ev), flush=True)

    cck = torch.load(args.coder, map_location="cpu")
    coder = Coder().to(device).eval()
    coder.load_state_dict(cck["coder"])
    books = [b.to(device) for b in cck["pq"]["8x256"]]
    means = torch.load(args.means)["means"]
    scale = float(means.norm(dim=1).mean())
    syl = load_sylber(device)
    ref, _ = sf.read(args.ref, dtype="float32")
    ref16 = np.clip(ref * 32768, -32768, 32767).astype(np.int16)
    st = hidden_states(syl, ref16)
    gref = torch.from_numpy(pool(st, segments_of(st).copy()).mean(0, keepdims=True)).to(device)
    del syl

    @torch.no_grad()
    def feats_of(codes: np.ndarray) -> np.ndarray:
        c = torch.from_numpy(codes.astype(np.int64)).to(device)
        z = F.normalize(torch.cat([books[g][c[:, g]] for g in range(8)], 1), dim=1)
        f = coder.rebuild(z, gref.expand(len(z), -1))
        return (f * scale).cpu().numpy()

    from sparc import load_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    synth = load_synth(args.config, device)
    sparc = load_model("en", device="cuda", use_penn=False)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0, torch_dtype=torch.float16)
    tk = AutoTokenizer.from_pretrained("gpt2-medium")
    lm = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device).eval()

    def wav(evs: np.ndarray, codes: np.ndarray, pros: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(evs == UNIT)
        tok = np.where(evs == CONT, K, K + 1).astype(np.int64)
        tok[idx] = np.arange(len(idx))
        table = feats_of(codes[idx]) if len(idx) else np.zeros((1, 768), np.float32)
        return render(synth, sparc, tok, pros, table, ref, device)

    def text(w):
        return asr({"raw": w.astype(np.float32), "sampling_rate": SR}, generate_kwargs={"language": "en"})["text"].strip()

    @torch.no_grad()
    def logppl(t):
        ids = tk(t, return_tensors="pt").input_ids.to(device)
        return float("nan") if ids.shape[1] < 3 else float(lm(ids, labels=ids).loss)

    def stats(evs, codes, pros):
        u = evs == UNIT
        active = (evs != SIL).sum()
        stv = pros[u & (pros[:, 2] > 0.5), 0]
        return {"rate": float(u.sum() / max(active / 10, 0.1)), "pause": float((evs == SIL).mean()),
                "f0_std": float(stv.std()) if len(stv) > 2 else float("nan"), "rep4": rep4(codes[u][:, 0])}

    rows = {}
    for kind, E, Cc, P in (("gen", g_ev, g_cd, g_pr), ("real", r_ev, r_cd, r_pr), ("floor", r_ev, f_cd, r_pr)):
        r = []
        for i in range(len(E)):
            t = text(wav(E[i], Cc[i], P[i]))
            r.append({"text": t, "logppl": logppl(t), "words": len(t.split()), **stats(E[i], Cc[i], P[i])})
        rows[kind] = r
        print(kind, json.dumps(r[:2]), flush=True)
    # topic test (text models as rulers only): Whisper transcript of the 30 s prompt vs of each continuation;
    # right when a continuation is closer to its own prompt than to another random prompt (MiniLM sentence vectors)
    from transformers import AutoModel
    etk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emb = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device).eval()

    @torch.no_grad()
    def vec(texts):
        b = etk(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        h = emb(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

    prompt_text = [text(wav(p_ev[i], p_cd[i], pr_pr[i])) for i in range(len(p_ev))]
    pv = vec([t or "." for t in prompt_text])
    trng = np.random.default_rng(7)
    other_idx = np.array([(i + trng.integers(1, len(pv))) % len(pv) for i in range(len(pv))])

    def topic(texts):
        cv = vec([t or "." for t in texts])
        own = (cv * pv).sum(-1)
        oth = (cv * pv[torch.from_numpy(other_idx).to(device)]).sum(-1)
        return {"topic_acc": float(((own > oth).float() + 0.5 * (own == oth).float()).mean()),
                "sim_own": float(own.mean()), "sim_other": float(oth.mean())}

    topics = {k: topic([x["text"] for x in rows[k]]) for k in rows}
    if args.other:
        o = json.loads(Path(args.other).read_text())["rows"]["gen"]
        if len(o) >= len(pv):
            topics["dy22_gen"] = topic([x["text"] for x in o[: len(pv)]])
    print("TOPIC", json.dumps(topics), flush=True)
    for i in range(0 if args.no_listen else min(10, len(g_ev))):
        for kind, E, Cc, P in (("gen", g_ev, g_cd, g_pr), ("real", r_ev, r_cd, r_pr)):
            w = wav(np.concatenate([p_ev[i][-50:], E[i]]), np.concatenate([p_cd[i][-50:], Cc[i]]), np.concatenate([pr_pr[i][-50:], P[i]]))
            sf.write(out / "listen" / f"{i:02d}_{kind}.wav", w, SR)

    med = lambda k, key: float(np.nanmedian([x[key] for x in rows[k]]))
    summ = {k: {"median_logppl": med(k, "logppl"), "words_median": med(k, "words"),
                "no_text_share": float(np.mean([np.isnan(x["logppl"]) for x in rows[k]])),
                "rep4": float(np.mean([x["rep4"] for x in rows[k]])), "rate": med(k, "rate"),
                "pause": med(k, "pause"), "f0_std": med(k, "f0_std")} for k in rows}
    g, r, f = (summ[k]["median_logppl"] for k in ("gen", "real", "floor"))
    summ["gap_closed"] = (f - g) / (f - r) if f != r else float("nan")
    G, R = summ["gen"], summ["real"]
    summ["L1"] = bool(summ["gap_closed"] >= 0.5 and G["rep4"] <= 2 * max(R["rep4"], 1e-3))
    summ["L2"] = bool(abs(G["rate"] / R["rate"] - 1) <= 0.15 and abs(G["pause"] - R["pause"]) <= 0.3 * R["pause"])
    summ["L3"] = bool(abs(G["f0_std"] / R["f0_std"] - 1) <= 0.3)
    summ["L"] = summ["L1"] and summ["L2"] and summ["L3"]
    summ["n"] = len(g_ev)
    summ["topic"] = topics
    (out / "eval.json").write_text(json.dumps({"summary": summ, "rows": rows, "prompt_text": prompt_text}, indent=2, ensure_ascii=False))
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
