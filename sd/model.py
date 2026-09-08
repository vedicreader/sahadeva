"Sahadeva acoustic model: non-autoregressive phone→mel with durations taken from forced alignment."
import torch, torch.nn as nn, torch.nn.functional as F
from . import cfg

__all__ = ['Sahadeva', 'expand']

class ConvBlock(nn.Module):
    "Depth-friendly 1d conv block: conv → GELU → LayerNorm → dropout, residual."
    def __init__(s, d, k=cfg.KERNEL, dil=1, p=cfg.DROPOUT):
        super().__init__()
        s.c = nn.Conv1d(d, d, k, padding=dil * (k - 1) // 2, dilation=dil)
        s.n, s.dp = nn.LayerNorm(d), nn.Dropout(p)
    def forward(s, x):                                  # x: (B, T, D)
        h = s.c(x.transpose(1, 2)).transpose(1, 2)
        return s.n(x + s.dp(F.gelu(h)))

def expand(h, d):
    "Length regulator: repeat each token's state d[i] times (per batch item, padded to the max)."
    out = [torch.repeat_interleave(h[b], d[b].clamp(min=0), dim=0) for b in range(h.shape[0])]
    n = max(o.shape[0] for o in out)
    return torch.stack([F.pad(o, (0, 0, 0, n - o.shape[0])) for o in out]), n

class Sahadeva(nn.Module):
    def __init__(s, n_tok, n_spk, d=cfg.D_MODEL, n_mel=cfg.N_MEL, enc=cfg.ENC_DILS, dils=cfg.DEC_DILS):
        super().__init__()
        s.emb, s.spk = nn.Embedding(n_tok, d, padding_idx=0), nn.Embedding(n_spk, d)
        s.enc = nn.ModuleList([ConvBlock(d, dil=x) for x in enc])
        s.dur = nn.Sequential(ConvBlock(d, k=3), ConvBlock(d, k=3))
        s.dur_out = nn.Linear(d, 1)
        s.dec = nn.ModuleList([ConvBlock(d, dil=x) for x in dils])
        s.out = nn.Linear(d, n_mel)
        s.post = nn.Sequential(*[ConvBlock(n_mel, k=5, dil=1) for _ in range(cfg.POST_BLOCKS)], nn.Linear(n_mel, n_mel))

    def encode(s, tok, spk):
        h = s.emb(tok) + s.spk(spk)[:, None]
        for b in s.enc: h = b(h)
        return h

    def forward(s, tok, spk, dur=None, mask=None, ratio=1.0):
        "dur given → teacher-forced training; dur None → predict durations (inference)."
        h = s.encode(tok, spk)
        logd = s.dur_out(s.dur(h.detach() if dur is not None else h)).squeeze(-1)
        if dur is None:
            dur = torch.clamp((logd.exp() - 1) * ratio, min=1).round().long()
            if mask is not None: dur = dur * mask.long()
        z, _ = expand(h + s.spk(spk)[:, None], dur)
        for b in s.dec: z = b(z)
        pre = s.out(z)
        return pre, pre + s.post(pre), logd, dur
