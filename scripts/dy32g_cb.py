"""DY-32 track G (one-voice twin of track B): track A's k-means centroids (.npy [K, 1024], WavLM-Large layer 6, 50 Hz)
-> a codebook file in track B's format (dy32b_channel.Codebook: cent, cent_n, dec [K, 1, 1024], rate, meta), so that
track B's own unit pass, trainer and generator run unchanged with only the centroid file swapped.

Assignment is argmax cosine to cent_n (as track A's "cosine assignment"); decoding is the centroid itself (track A's
"centroid lookup -> HiFi-GAN").

    python scripts/dy32g_cb.py --npy /data/a/centroids/km-lj-K2000.npy --out /data/g/cb/a_lj_k2000_r50.pt
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rate", type=int, default=50)
    a = p.parse_args()
    c = torch.from_numpy(np.load(a.npy).astype(np.float32))
    assert c.ndim == 2 and c.shape[1] == 1024 and a.rate == 50, c.shape
    torch.save({"cent": c, "cent_n": F.normalize(c, dim=-1), "dec": c[:, None].clone(), "rate": a.rate,
                "meta": {"source": a.npy, "K": int(c.shape[0]), "rate_hz": a.rate,
                         "fit": "track A k-means on LJSpeech only (one voice), WavLM-Large layer 6",
                         "assign": "argmax cosine", "decode": "centroid lookup"}}, a.out)
    print("CODEBOOK", a.out, tuple(c.shape))


if __name__ == "__main__":
    main()
