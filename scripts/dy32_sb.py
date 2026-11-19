"""DY-32 track A, job 2 (b): multi-layer discrete WavLM codes with SpeechBrain's ready unit vocoder.

microsoft/wavlm-large hidden layers 1, 3, 7, 12, 18, 23 -> k-means K=1000 per layer (speechbrain k-means trained on
LibriSpeech audio, no text) -> speechbrain/hifigan-wavlm-l1-3-7-12-18-23-k1000-LibriTTS (unit HiFi-GAN trained on
LibriTTS audio, attention pooling over layers; missing layers are filled with the padding token, as
DiscreteSSL.decode does). All six layers and subsets; rows go to the dy32_channel table.

    python scripts/dy32_sb.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dy32_channel import S3, W, Rec, Table, log, reals, score, sh, write

VOC = "speechbrain/hifigan-wavlm-l1-3-7-12-18-23-k1000-LibriTTS"
SUBSETS = [[1, 3, 7, 12, 18, 23], [3, 7, 12, 18, 23], [1, 3, 7, 12], [3, 7, 12], [7, 12, 18], [7, 12], [3, 7], [7], [12], [1, 3]]


def main() -> None:
    import torch
    try:
        from speechbrain.integrations.huggingface.wavlm import WavLM
    except ImportError:
        from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    from speechbrain.integrations.audio_tokenizers.discrete_ssl import DiscreteSSL

    device = torch.device("cuda")
    save = "/opt/dlami/nvme/sb"
    ssl = WavLM("microsoft/wavlm-large", save, output_all_hiddens=True).to(device)
    m = DiscreteSSL(save, ssl, kmeans_dataset="LibriSpeech", vocoder_repo_id=VOC, num_clusters=1000, device="cuda").to(device)
    m.codec_vocoder.mods.to(device)
    m.codec_vocoder.device = "cuda"
    log("layers", m.ssl_layer_ids)
    R = reals()
    ref = json.loads((W / "ref.json").read_text())
    rec, tab = Rec(device), Table()
    toks = []
    with torch.no_grad():
        for w in R:
            x = torch.from_numpy(w)[None].to(device)
            t, _, _ = m.encode(x, torch.ones(1, device=device), SSL_layers=list(m.ssl_layer_ids))
            toks.append(t)
    for sub in SUBSETS:
        name = "sb-wavlm-L" + "-".join(map(str, sub))
        idx = [list(m.ssl_layer_ids).index(l) for l in sub]
        waves = []
        with torch.no_grad():
            for t in toks:
                y = m.decode(t[..., idx], SSL_layers=sub)
                waves.append(y.squeeze().float().cpu().numpy())
        for i, w in enumerate(waves):
            write(W / "resynth" / name, i, w)
        h = rec(waves)
        sc = score(ref, h)
        tab.add(name, f"WavLM-large layers {sub}, speechbrain k-means K=1000 (LibriSpeech) -> speechbrain 6-layer unit HiFi-GAN (LibriTTS)",
                f"50 Hz x {len(sub)} x 10 bit = {500 * len(sub)} bit/s", sc, h)
        log("ROW", name, json.dumps({k: {"median": v["median"], "mean": v["mean"]} for k, v in sc.items()}))
        sh(f"aws s3 sync {W}/resynth/{name}/ {S3}/a/resynth/{name}/ --only-show-errors")
    log("SB DONE")


if __name__ == "__main__":
    main()
