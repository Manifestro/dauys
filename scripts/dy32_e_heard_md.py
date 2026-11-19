import json
r = json.load(open("/opt/dlami/nvme/heard.json"))
L = ["# DY-32 track E: does GSLM2 use the content of what it heard? (likelihood, held-out)", "",
"Held-out shards s00197, lbig-00550 (never trained on); windows = one random point per piece >= 25 s (seed 1234), 39 episodes; own = true next 10 s.",
"a = true heard part right before own; b = same episode (pieces.jsonl ep), chunk >= 60 s (episode time) away from the whole heard+own window; c = another episode.",
"Heard and own audio encoded SEPARATELY with SpidR 2.0 (5 s heard = last 5 s of the 15 s chunk, encoded on its own), each cross-stream deduplicated separately, then concatenated (generation path).",
"G = Depformer generation CE: teacher-forced CE of each of the 6 streams given the context and the earlier streams in the generation order, mean over 6 streams (the sampler likelihood; training term gen_loss). H = per-stream backbone head CE, mean over 6 streams (multi-prediction term). Nats per deduplicated position, own part only; time bucket by the kept full-rate frame (50 Hz). Gaps: mean of paired per-window differences, +- 95% CI (1.96 SE). Shares: windows with CE(a) lower, +- 95% CI. 30 s heard not run (time).", ""]
for m, v in r.items():
    for loss in ("G", "H"):
        L += [f"## {m} ({v['lm']}), loss {loss}, {v['windows']} windows", "",
              "| heard | bucket | CE a | CE b | CE c | b-a [95% CI] | c-a [95% CI] | share a<b | share a<c |", "|---|---|---|---|---|---|---|---|---|"]
        for t in v["table"]:
            if t["loss"] != loss: continue
            f = lambda x: f"{x[0]:+.4f} [{x[0]-x[1]:+.4f}, {x[0]+x[1]:+.4f}]"
            s = lambda x: f"{x[0]:.3f} +- {x[1]:.3f}"
            L.append(f"| {t['heard_s']} s | {t['bucket']} | {t['a']:.4f} | {t['b']:.4f} | {t['c']:.4f} | {f(t['b_a'])} | {f(t['c_a'])} | {s(t['share_a_lt_b'])} | {s(t['share_a_lt_c'])} |")
        L.append("")
L.append("Models: released GSLM2 1B (Libri-Light 60K h) and 7B (conversational 6M h), CC BY-NC 4.0; ours = the 1B fine-tuned on 824 h DuplexChat units, step 9,242. Code: dauys branch dy32-e, scripts/dy32_e_heard.py.")
open("/opt/dlami/nvme/heard_use.md", "w").write("\n".join(L) + "\n")
