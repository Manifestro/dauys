"""DY-32 track G: the classic textless GSLM pipeline (Lakhotia et al. 2021) under modern PyTorch.

speech -> HuBERT-base-ls960 (self-supervised only, HF ``facebook/hubert-base-ls960``) layer L features -> k-means
(released GSLM ``km.bin``) -> units -> unit language model -> units -> unit vocoder -> 16 kHz speech.

Pieces:
- ``Hubert``: HF HubertModel, ``hidden_states[L]`` = output of transformer layer L (fairseq ``output_layer=L``;
  HuBERT base is post-norm, so no final norm is applied in either). Raw waveform in, no normalisation (as GSLM).
- ``KMeans``: cluster centres of a sklearn ``km.bin``; nearest centre on the GPU.
- ``CodeHiFiGAN``: unit HiFi-GAN generator (code from fairseq / textlesslib, MIT licence), optional duration
  predictor (for deduplicated units), optional speaker/style embeddings (Expresso vocoder).
- ``FairseqULM``: re-implementation of fairseq ``transformer_lm_big`` (pre-norm, sinusoidal positions, tied
  embeddings) that loads the released GSLM unit-LM weights; dictionary: 0 <s>, 1 <pad>, 2 </s>, 3 <unk>, unit u -> u + 4.
- ``Tacotron``: the released GSLM unit Tacotron2 + WaveGlow through textlesslib's own modules.
"""

import json
import math
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm

SR = 16000
HOP = 320  # 50 Hz units


# ---------------------------------------------------------------- encoder + quantiser

class Hubert:
    def __init__(self, layer: int, device, name: str = "facebook/hubert-base-ls960") -> None:
        from transformers import HubertModel

        self.layer = layer
        self.device = device
        self.model = HubertModel.from_pretrained(name).to(device).eval()
        self.model.encoder.layers = self.model.encoder.layers[:layer]  # later layers are never used

    @torch.no_grad()
    def __call__(self, wavs: torch.Tensor) -> torch.Tensor:
        """wavs [B, N] float in [-1, 1], all the same length (HuBERT base uses group norm: no padding) -> [B, T, 768]."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(wavs.to(self.device), output_hidden_states=True)
        return out.hidden_states[self.layer].float()


class KMeans:
    def __init__(self, path: str, device) -> None:
        import joblib

        km = joblib.load(path)
        self.c = torch.from_numpy(np.asarray(km.cluster_centers_, np.float32)).to(device)
        self.c2 = (self.c**2).sum(-1)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., D] -> nearest centre ids [...]."""
        d = self.c2 - 2 * x @ self.c.T
        return d.argmin(-1)

    @torch.no_grad()
    def inertia(self, x: torch.Tensor) -> float:
        d = (x**2).sum(-1, keepdim=True) + self.c2 - 2 * x @ self.c.T
        return float(d.min(-1).values.mean())


def dedup(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Runs of equal units -> (unit per run, run length in frames)."""
    if len(u) == 0:
        return u, np.zeros(0, np.int64)
    cut = np.r_[0, np.nonzero(np.diff(u))[0] + 1]
    return u[cut], np.diff(np.r_[cut, len(u)])


def encode(enc: Hubert, km: KMeans, wav: np.ndarray, win_s: float = 20.0) -> np.ndarray:
    """One float waveform -> full-rate (50 Hz) unit ids, in windows of win_s seconds."""
    win = int(win_s * SR)
    out = []
    for i in range(0, len(wav), win):
        w = wav[i : i + win]
        if len(w) < 400:
            break
        out.append(km(enc(torch.from_numpy(w.astype(np.float32))[None]))[0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.int64)


# ---------------------------------------------------------------- unit HiFi-GAN (fairseq / textlesslib, MIT)

LRELU = 0.1


def _pad(k: int, d: int = 1) -> int:
    return (k * d - d) // 2


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dil) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(weight_norm(Conv1d(ch, ch, k, 1, dilation=d, padding=_pad(k, d))) for d in dil)
        self.convs2 = nn.ModuleList(weight_norm(Conv1d(ch, ch, k, 1, dilation=1, padding=_pad(k))) for _ in dil)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
        return x


class VariancePredictor(nn.Module):
    def __init__(self, d_in: int, d: int, k: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(d_in, d, k, padding=(k - 1) // 2), nn.ReLU())
        self.ln1 = nn.LayerNorm(d)
        self.conv2 = nn.Sequential(nn.Conv1d(d, d, k, padding=1), nn.ReLU())
        self.ln2 = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, x):  # [B, T, C] -> [B, T]
        x = self.ln1(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        x = self.ln2(self.conv2(x.transpose(1, 2)).transpose(1, 2))
        return self.proj(x).squeeze(2)


class CodeHiFiGAN(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.nk = len(cfg["resblock_kernel_sizes"])
        ch0 = cfg["upsample_initial_channel"]
        self.conv_pre = weight_norm(Conv1d(cfg.get("model_in_dim", 80), ch0, 7, 1, padding=3))
        self.ups = nn.ModuleList(
            weight_norm(ConvTranspose1d(ch0 // 2**i, ch0 // 2 ** (i + 1), k, u, padding=(k - u) // 2))
            for i, (u, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = ch0 // 2 ** (i + 1)
            for k, d in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.dict = nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"])
        self.spkr = nn.Embedding(cfg["num_speakers"], cfg["embedding_dim"]) if cfg.get("multispkr") else None
        self.style = nn.Embedding(cfg["num_styles"], cfg["embedding_dim"]) if cfg.get("multistyle") else None
        dp = cfg.get("dur_predictor_params")
        self.dur_predictor = VariancePredictor(dp["encoder_embed_dim"], dp["var_pred_hidden_dim"],
                                               dp["var_pred_kernel_size"]) if dp else None

    @classmethod
    def load(cls, ckpt: str, cfg_path: str, device) -> "CodeHiFiGAN":
        cfg = json.load(open(cfg_path))
        m = cls(cfg)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["generator"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        return m.to(device).eval()

    def frames(self, units: torch.Tensor) -> torch.Tensor:
        """Deduplicated units [T] -> units repeated by the predicted durations (as fairseq dur_prediction)."""
        x = self.dict(units[None])
        dur = torch.clamp(torch.round(torch.exp(self.dur_predictor(x)) - 1).long(), min=1)
        return torch.repeat_interleave(units, dur[0])

    @torch.no_grad()
    def forward(self, units: np.ndarray, dur_prediction: bool = False, spkr: int = 0, style: int = 0) -> np.ndarray:
        dev = self.dict.weight.device
        u = torch.as_tensor(np.asarray(units, np.int64), device=dev)
        if dur_prediction:
            u = self.frames(u)
        x = self.dict(u[None]).transpose(1, 2)
        T = x.shape[-1]
        if self.spkr is not None:
            x = torch.cat([x, self.spkr.weight[spkr][None, :, None].expand(1, -1, T)], 1)
        if self.style is not None:
            x = torch.cat([x, self.style.weight[style][None, :, None].expand(1, -1, T)], 1)
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU))
            xs = sum(self.resblocks[i * self.nk + j](x) for j in range(self.nk))
            x = xs / self.nk
        x = torch.tanh(self.conv_post(F.leaky_relu(x)))
        return x[0, 0].float().cpu().numpy()


# ---------------------------------------------------------------- released GSLM unit LM (fairseq transformer_lm_big)

def sinusoidal(n: int, d: int, padding_idx: int = 1) -> torch.Tensor:
    half = d // 2
    f = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000) / (half - 1)))
    a = torch.arange(n, dtype=torch.float32)[:, None] * f[None]
    e = torch.cat([torch.sin(a), torch.cos(a)], 1)
    e[padding_idx] = 0
    return e


class FairseqULM(nn.Module):
    BOS, PAD, EOS, UNK, OFF = 0, 1, 2, 3, 4

    def __init__(self, sd: dict, args: dict) -> None:
        super().__init__()
        self.d, self.h, self.L = args["decoder_embed_dim"], args["decoder_attention_heads"], args["decoder_layers"]
        assert args["decoder_normalize_before"] and not args["decoder_learned_pos"] and args["activation_fn"] == "relu"
        sd = {k.removeprefix("decoder."): v for k, v in sd.items()}
        self.emb = nn.Embedding.from_pretrained(sd["embed_tokens.weight"].float(), freeze=True)
        self.vocab = self.emb.weight.shape[0]
        self.pos = sinusoidal(args["max_target_positions"] + 2, self.d)
        P = nn.ParameterDict()
        for k, v in sd.items():
            if k.startswith("layers.") or k.startswith("layer_norm.") or k == "output_projection.weight":
                P[k.replace(".", "__")] = nn.Parameter(v.float(), requires_grad=False)
        self.P = P

    @classmethod
    def load(cls, path: str, device) -> "FairseqULM":
        s = torch.load(path, map_location="cpu", weights_only=False)
        args = s["cfg"]["model"]
        args = dict(args) if not hasattr(args, "__dict__") else vars(args)
        m = cls(s["model"], args)
        m.pos = m.pos.to(device)
        return m.to(device).eval()

    def p(self, k: str) -> torch.Tensor:
        return self.P[k.replace(".", "__")]

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B, T] (no padding) -> next-token logits [B, T, V] (float32)."""
        B, T = ids.shape
        x = self.emb(ids) * math.sqrt(self.d) + self.pos[2 : 2 + T][None]
        dh = self.d // self.h
        for i in range(self.L):
            pre = f"layers.{i}."
            h = F.layer_norm(x, (self.d,), self.p(pre + "self_attn_layer_norm.weight"), self.p(pre + "self_attn_layer_norm.bias"))
            q = F.linear(h, self.p(pre + "self_attn.q_proj.weight"), self.p(pre + "self_attn.q_proj.bias"))
            k = F.linear(h, self.p(pre + "self_attn.k_proj.weight"), self.p(pre + "self_attn.k_proj.bias"))
            v = F.linear(h, self.p(pre + "self_attn.v_proj.weight"), self.p(pre + "self_attn.v_proj.bias"))
            q, k, v = (t.view(B, T, self.h, dh).transpose(1, 2) for t in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, self.d)
            x = x + F.linear(a, self.p(pre + "self_attn.out_proj.weight"), self.p(pre + "self_attn.out_proj.bias"))
            h = F.layer_norm(x, (self.d,), self.p(pre + "final_layer_norm.weight"), self.p(pre + "final_layer_norm.bias"))
            h = F.linear(F.relu(F.linear(h, self.p(pre + "fc1.weight"), self.p(pre + "fc1.bias"))),
                         self.p(pre + "fc2.weight"), self.p(pre + "fc2.bias"))
            x = x + h
        x = F.layer_norm(x, (self.d,), self.p("layer_norm.weight"), self.p("layer_norm.bias"))
        return F.linear(x, self.p("output_projection.weight")).float()


def sample_top_p(lg: torch.Tensor, temp: float, top_p: float, g: torch.Generator) -> torch.Tensor:
    """lg [B, V] -> sampled ids [B] (temperature, then nucleus)."""
    p = torch.softmax(lg / temp, -1)
    ps, ix = p.sort(-1, descending=True)
    keep = ps.cumsum(-1) - ps < top_p
    ps = ps * keep
    ps = ps / ps.sum(-1, keepdim=True)
    j = torch.multinomial(ps, 1, generator=g)
    return ix.gather(-1, j)[:, 0]


# ---------------------------------------------------------------- released GSLM Tacotron2 + WaveGlow (textlesslib)

class Tacotron:
    """textlesslib's TacotronVocoder without running textless/__init__ (which imports fairseq)."""

    def __init__(self, textlesslib_dir: str, tts: str, code_dict: str, waveglow: str, device) -> None:
        pkg = types.ModuleType("textless")
        pkg.__path__ = [f"{textlesslib_dir}/textless"]
        sys.modules.setdefault("textless", pkg)
        import librosa.util

        if not getattr(librosa.util, "_positional", False):  # textlesslib calls the pre-0.10 positional signature
            _pc = librosa.util.pad_center
            librosa.util.pad_center = lambda data, size=None, **k: _pc(data, size=size, **k)
            librosa.util._positional = True
        from textless.vocoders.tacotron2.vocoder import TacotronVocoder

        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
        try:
            self.v = TacotronVocoder(tts, code_dict, waveglow, max_decoder_steps=3000, denoiser_strength=0.1).to(device)
        finally:
            torch.load = _load
        self.sr = self.v.output_sample_rate

    @torch.no_grad()
    def __call__(self, units: np.ndarray) -> np.ndarray:
        """Deduplicated units -> 16 kHz float audio (resampled if the Tacotron rate differs)."""
        a = self.v(torch.as_tensor(np.asarray(units, np.int64))).float().cpu()
        if self.sr != SR:
            import torchaudio

            a = torchaudio.functional.resample(a, self.sr, SR)
        return a.numpy()
