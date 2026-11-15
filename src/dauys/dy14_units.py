"""Syllable units for DY-14: Sylber segments of each headset track, cleaned and gridded.

Sylber (Cho et al., ICLR 2025, https://github.com/Berkeley-Speech-Group/sylber, MIT)
is a HuBERT-base cut at layer 9 whose frame norms and cosine merges give one segment
per syllable at 50 Hz. Only its inference is needed, so the model is built here from
``transformers`` and the checkpoint, and ``get_segment`` is copied from
``sylber/utils/segment_utils.py`` (0.1.4) instead of installing the package and its
training dependencies.

Per track this module keeps a segment only when the own headset is at least
``BLEED_DB`` louder than the other one over it: a headset also picks up the other
person, and those segments would put the partner's words into the own channel.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

SR = 16000
HOP = 320  # HuBERT frame: 20 ms at 16 kHz
SYLBER_REPO = "cheoljun95/sylber"
SYLBER_REVISION = "39451274641088bacc52c78a4bdc223086e6e92c"
HUBERT = "facebook/hubert-base-ls960"
LAYER = 9
NORM_THRESHOLD = 2.6
MERGE_THRESHOLD = 0.8
CHUNK_S = 20
BLEED_DB = 6.0
F0_MIN, F0_MAX = 50.0, 400.0
VOICED_CORR = 0.45
GRID = 5  # Sylber frames per grid frame: 10 Hz


def load_sylber(device: str | torch.device) -> torch.nn.Module:
    """HuBERT-base with 9 layers and the Sylber weights, in eval mode."""
    from huggingface_hub import hf_hub_download
    from transformers import HubertConfig, HubertModel

    config = HubertConfig.from_pretrained(HUBERT, num_hidden_layers=LAYER)
    model = HubertModel(config)
    path = hf_hub_download(SYLBER_REPO, "sylber.ckpt", revision=SYLBER_REVISION)
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Sylber's checkpoint is the speech model only; a large mismatch means a wrong file
    loaded = len(model.state_dict()) - len(missing)
    if loaded < 0.95 * len(model.state_dict()):
        raise RuntimeError(
            f"Sylber checkpoint loaded {loaded} tensors, missing {missing[:5]}"
        )
    print(
        f"sylber: {loaded} tensors loaded, {len(missing)} missing, {len(unexpected)} unexpected"
    )
    return model.eval().to(device)


@torch.no_grad()
def hidden_states(
    model: torch.nn.Module, wav: np.ndarray, batch: int = 16
) -> np.ndarray:
    """Layer-9 states of one track at 50 Hz, ``[frames, 768]`` float32 on the CPU.

    The track is cut into 20 s chunks, each normalised to zero mean and unit variance
    as Sylber's ``Segmenter`` does with a whole file. A chunk edge can split a syllable.
    """
    device = next(model.parameters()).device
    x = torch.from_numpy(wav.astype(np.float32) / 32768.0)
    size = CHUNK_S * SR
    frames = len(x) // HOP
    out = np.zeros((frames, 768), np.float32)
    starts = list(range(0, len(x), size))
    for i in range(0, len(starts), batch):
        group = starts[i : i + batch]
        chunks = [x[s : s + size] for s in group]
        chunks = [(c - c.mean()) / (c.std() + 1e-5) for c in chunks]
        lengths = [len(c) for c in chunks]
        padded = torch.stack([F.pad(c, (0, size - len(c))) for c in chunks]).to(device)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            states = model(padded).last_hidden_state.float().cpu().numpy()
        for s, n, st in zip(group, lengths, states):
            first = s // HOP
            count = min(n // HOP, st.shape[0], frames - first)
            out[first : first + count] = st[:count]
    return out


def _cossim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        (x * y).sum(-1)
        / (((x**2).sum(-1) + 1e-8) ** 0.5)
        / (((y**2).sum(-1) + 1e-8) ** 0.5)
    )


def get_segment(
    states: np.ndarray,
    normthreshold: float = NORM_THRESHOLD,
    mergethreshold: float = MERGE_THRESHOLD,
) -> np.ndarray:
    """Sylber's segmentation, copied unchanged in logic: ``[n, 2]`` start/end frames."""
    norms = ((states**2).sum(-1) + 1e-8) ** 0.5
    mask = norms >= normthreshold
    curr = 0
    seg_cnt = 0
    segments = []
    s = -1
    midboundaries = []
    for i in range(len(states)):
        if not mask[i]:
            if s > -1:
                segments.append([s, i])
            s = -1
            curr = 0
            seg_cnt = 0
        else:
            if seg_cnt == 0:
                curr = states[i]
                seg_cnt += 1
                s = i
            else:
                sim = _cossim(curr, states[i])
                if sim >= mergethreshold:
                    curr = (curr * seg_cnt + states[i]) / (seg_cnt + 1)
                    seg_cnt += 1
                else:
                    curr = states[i]
                    seg_cnt += 1
                    segments.append([s, i])
                    midboundaries.append([i, len(segments) - 1])
                    s = i
    if s > -1:
        segments.append([s, len(states)])
    merged = []
    for bd, segi in midboundaries:
        if segi >= len(segments) - 1:
            continue
        if (
            _cossim(
                states[segments[segi][0] : segments[segi][1]].mean(0),
                states[segments[segi + 1][0] : segments[segi + 1][1]].mean(0),
            )
            >= mergethreshold
        ):
            segments[segi + 1] = [segments[segi][0], segments[segi + 1][1]]
            merged.append(segi)
            continue
        s = max(
            segments[segi][0], bd - max(1, (segments[segi][1] - segments[segi][0]) // 2)
        )
        bd = min(
            segments[segi + 1][1],
            bd + max(1, (segments[segi + 1][1] - segments[segi + 1][0]) // 2),
        )
        prev_center = states[segments[segi][0] : segments[segi][1]].mean(0)
        next_center = states[segments[segi + 1][0] : segments[segi + 1][1]].mean(0)
        sim_prev = _cossim(states[s:bd], prev_center[None, :])
        sim_next = _cossim(states[s:bd], next_center[None, :])
        sim_sweep = [(sim_prev[:i].sum() + sim_next[i:].sum()) for i in range(bd - s)]
        opt_b = np.arange(s, bd)[np.argmax(sim_sweep)]
        segments[segi] = [segments[segi][0], opt_b]
        segments[segi + 1] = [opt_b, segments[segi + 1][1]]
    segments = [seg for segi, seg in enumerate(segments) if segi not in merged]
    return np.array(segments, dtype=np.int64).reshape(-1, 2)


def segment_rms(wav: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """RMS of int16 audio over each segment's samples."""
    x = wav.astype(np.float64) / 32768.0
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    lo = np.clip(segments[:, 0] * HOP, 0, len(x))
    hi = np.clip(segments[:, 1] * HOP, 0, len(x))
    n = np.maximum(hi - lo, 1)
    return np.sqrt((c[hi] - c[lo]) / n)


@torch.no_grad()
def segment_f0(
    wav: np.ndarray, segments: np.ndarray, device: str | torch.device, batch: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Median F0 (Hz) and a voiced flag per segment from normalised autocorrelation.

    40 ms frames every 20 ms over at most the middle 0.5 s of the segment; a frame is
    voiced when its best lag between 2.5 and 20 ms correlates at least ``VOICED_CORR``.
    """
    win, hop, cap = 640, 320, SR // 2
    lo_lag, hi_lag = int(SR / F0_MAX), int(SR / F0_MIN)
    x = torch.from_numpy(wav.astype(np.float32) / 32768.0)
    f0 = np.zeros(len(segments), np.float32)
    voiced = np.zeros(len(segments), bool)
    for i in range(0, len(segments), batch):
        seg = segments[i : i + batch]
        mids = (seg[:, 0] + seg[:, 1]) * HOP // 2
        lens = np.minimum((seg[:, 1] - seg[:, 0]) * HOP, cap)
        lens = np.maximum(lens, win)
        starts = np.clip(mids - lens // 2, 0, max(len(x) - win, 0))
        clips = torch.zeros(len(seg), cap)
        for j, (s, n) in enumerate(zip(starts, lens)):
            piece = x[s : s + n]
            clips[j, : len(piece)] = piece
        clips = clips.to(device)
        frames = clips.unfold(1, win, hop)  # [S, F, win]
        frames = frames - frames.mean(-1, keepdim=True)
        spec = torch.fft.rfft(frames, n=2 * win)
        ac = torch.fft.irfft(spec.abs() ** 2, n=2 * win)[..., : hi_lag + 1]
        ac = ac / (ac[..., :1] + 1e-8)
        best, lag = ac[..., lo_lag:].max(-1)
        lag = lag + lo_lag
        valid = (
            torch.arange(frames.shape[1], device=device)[None, :] * hop + win
            <= torch.as_tensor(lens, device=device)[:, None]
        )
        ok = (best >= VOICED_CORR) & valid
        hz = SR / lag.float()
        hz = torch.where(ok, hz, torch.nan)
        med = torch.nanmedian(hz, dim=1).values
        f0[i : i + batch] = torch.nan_to_num(med, nan=0.0).cpu().numpy()
        voiced[i : i + batch] = (ok.sum(1) >= 2).cpu().numpy()
    return f0, voiced


@dataclass
class Track:
    """Sylber segments of one headset track (frames at 50 Hz)."""

    segments: np.ndarray  # [n, 2] int
    features: np.ndarray  # [n, 768] float16, mean layer-9 state
    keep: np.ndarray  # [n] bool: own speech, not the other person's bleed
    energy_db: np.ndarray  # [n] float32, relative to the median of kept segments
    f0_st: np.ndarray  # [n] float32 semitones vs the median voiced F0 of kept segments
    voiced: np.ndarray  # [n] bool
    frames: int


def build_track(
    states: np.ndarray,
    segments: np.ndarray,
    own: np.ndarray,
    other: np.ndarray,
    f0: np.ndarray,
    voiced: np.ndarray,
) -> Track:
    if len(segments):
        # sums over [start, end) by reduceat on interleaved bounds; a zero row keeps
        # every index inside the array when the last segment ends at the last frame
        padded = np.concatenate([states, np.zeros((1, states.shape[1]), states.dtype)])
        sums = np.add.reduceat(padded, segments.reshape(-1), axis=0)[0::2]
        n = (segments[:, 1] - segments[:, 0])[:, None]
        feats = (sums / n).astype(np.float16)
    else:
        feats = np.zeros((0, 768), np.float16)
    rms_own = segment_rms(own, segments)
    rms_other = segment_rms(other[: len(own)], segments) if len(segments) else rms_own
    level = 20 * np.log10(rms_own + 1e-9)
    keep = level - 20 * np.log10(rms_other + 1e-9) >= BLEED_DB
    energy = level - (np.median(level[keep]) if keep.any() else 0.0)
    good = keep & voiced & (f0 > 0)
    ref = np.median(f0[good]) if good.any() else 1.0
    f0_st = np.where(voiced & (f0 > 0), 12 * np.log2(np.maximum(f0, 1e-3) / ref), 0.0)
    return Track(
        segments,
        feats,
        keep,
        energy.astype(np.float32),
        f0_st.astype(np.float32),
        voiced,
        len(states),
    )


def grid_track(
    segments: np.ndarray,
    keep: np.ndarray,
    units: np.ndarray,
    energy: np.ndarray,
    f0_st: np.ndarray,
    voiced: np.ndarray,
    frames50: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One track on the 10 Hz grid.

    Token per grid frame: the unit of a kept segment starting in it, else ``k`` (speech
    continues) while a kept segment covers the frame's middle, else ``k + 1`` (silence).
    Prosody per frame ``[energy dB, F0 semitones, voiced]`` at unit frames, else zeros.
    When two segments start in one grid frame the first wins; that is a collision.
    """
    n = -(-frames50 // GRID)
    tok = np.full(n, k + 1, np.int16)
    pros = np.zeros((n, 3), np.float32)
    seg = segments[keep]
    u = units[keep]
    for s, e in seg:
        mid_first = (s - GRID // 2 + GRID - 1) // GRID
        tok[max(mid_first, 0) : -(-(e - GRID // 2) // GRID)] = k
    starts = seg[:, 0] // GRID if len(seg) else np.zeros(0, np.int64)
    collisions = 0
    taken = np.zeros(n, bool)
    for j, g in enumerate(starts):
        if taken[g]:
            collisions += 1
            continue
        taken[g] = True
        tok[g] = u[j]
        idx = np.flatnonzero(keep)[j]
        pros[g] = (energy[idx], f0_st[idx], float(voiced[idx]))
    active_s = float((tok != k + 1).sum()) / 10
    stats = {
        "units": len(seg),
        "collisions": int(collisions),
        "active_s": active_s,
        "bleed_dropped": int((~keep).sum()),
    }
    return tok, pros, stats
