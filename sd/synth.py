"Synthesise mel with the trained model and waveform with the pretrained HiFi-GAN vocoder."
import numpy as np, torch
from pathlib import Path
from . import cfg
from .g2p import g2p, tok2id, SIL, SP, UNK, n_vocab
from .model import Sahadeva

__all__ = ['load', 'text_tokens', 'mel_from_tokens', 'vocode', 'save_wav', 'Voice']

def load(ck=None):
    st = torch.load(Path(ck or (cfg.RUN_DIR / 'best.pt')), map_location='cpu', weights_only=False)
    m = Sahadeva(n_vocab(), len(st['spk'])); m.load_state_dict(st['model']); m.eval()
    return m, st['mu'], st['sd'], st['spk']

def text_tokens(text, gap=SIL):
    "Token ids for plain Devanagari text (no timings) — the duration predictor supplies the rhythm."
    out = [SIL]
    for w in (text or '').split():
        p = g2p(w)
        if p: out += p + [gap]
    return np.array([tok2id.get(t, tok2id[UNK]) for t in (out or [SIL])], np.int64)

@torch.no_grad()
def mel_from_tokens(m, mu, sd, tok, spk, dur=None, ratio=1.0):
    "Denormalised log-mel (frames, 80) for one token sequence."
    t = torch.as_tensor(tok)[None]
    d = torch.as_tensor(dur)[None].long() if dur is not None else None
    _, post, _, used = m(t, torch.tensor([spk]), d, mask=torch.ones_like(t), ratio=ratio)
    return post[0].numpy() * sd + mu, used[0].numpy()

_VOC = {}
@torch.no_grad()
def vocode(mel, name=cfg.VOCODER):
    "Pretrained HiFi-GAN (frozen) — no vocoder training needed, and it fixes Griffin-Lim's buzz."
    if name not in _VOC:
        from transformers import SpeechT5HifiGan
        _VOC[name] = SpeechT5HifiGan.from_pretrained(name).eval()
    return _VOC[name](torch.as_tensor(np.asarray(mel, np.float32))).numpy()

def save_wav(path, x, sr=cfg.SR):
    import soundfile as sf
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, np.clip(np.asarray(x, np.float32), -1, 1), sr); return p

class Voice:
    "Convenience wrapper: text (+ speaker) → waveform."
    def __init__(s, ck=None): s.m, s.mu, s.sd, s.spk = load(ck)
    def ids(s): return s.spk
    def say(s, text, speaker, ratio=1.0):
        i = s.spk[speaker] if isinstance(speaker, str) else speaker
        mel, _ = mel_from_tokens(s.m, s.mu, s.sd, text_tokens(text), i, ratio=ratio)
        return vocode(mel)
