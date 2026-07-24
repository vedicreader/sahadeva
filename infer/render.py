"""Render Sanskrit text through a fine-tuned F5/IndicF5 checkpoint.

Thin wrapper over `f5_tts.api.F5TTS` — the model code is theirs, this adds the three things
that make it work for accented Sanskrit:

  1. Text goes through dhvani, never raw. Feeding Devanagari straight in triggers Hindi
     schwa-deletion, which is the single most common way this pipeline sounds wrong.
  2. Reference clips are chosen by accent register. F5 has no duration or pitch head, so the
     reference clip is the only real prosody lever; a Vedic accented verse rendered against an
     unaccented stotra reference gets the wrong melody no matter how good the checkpoint is.
  3. Seeds are pinned by default, because F5 samples and unpinned eval takes are not comparable.

Runs on CUDA, MPS or CPU — F5-TTS's own device detection already covers Apple Silicon, so the
M3 path needs no patching.
"""
import json, argparse, random
from pathlib import Path

def pick_reference(manifest, accented=None, speaker=None, min_s=4.0, max_s=10.0, seed=0):
    """Choose a reference clip from the corpus.

    Prefers clips matching the target's accent register, then QC-clean mid-length ones. F5
    conditions heavily on this clip, so a mismatched reference costs more than a slightly
    undertrained checkpoint.
    """
    utts = [json.loads(l) for l in open(manifest)]
    pool = [u for u in utts if u.get("qc_pass", True) and min_s <= u["dur_s"] <= max_s
            and Path(u["audio"]).exists()]
    if speaker:  pool = [u for u in pool if u["speaker_id"] == speaker] or pool
    if accented is not None:
        match = [u for u in pool if bool(u["accented"]) == bool(accented)]
        if match: pool = match
    if not pool: raise SystemExit("no usable reference clips — run sd-segment and sd-qc first")
    pool.sort(key=lambda u: -u["qc"].get("snr_db", 0) if u.get("qc") else 0)
    return random.Random(seed).choice(pool[:max(1, len(pool) // 4)])   # clean quartile

class Renderer:
    """Loads the model once; render many.

    `ckpt` and `vocab` should point at your fine-tuned checkpoint and the vocab it was trained
    with — the same vocab prep_f5.py checked coverage against.
    """
    def __init__(self, ckpt, vocab, model="F5TTS_v1_Base", device=None, frontend=None):
        from f5_tts.api import F5TTS
        from dhvani import Frontend
        self.fe = frontend or Frontend()
        self.tts = F5TTS(model=model, ckpt_file=ckpt, vocab_file=vocab, device=device)
        self.device = self.tts.device

    def render(self, text, ref_audio, ref_text, out_wav=None, seed=0, nfe_step=32,
               cfg_strength=2.0, speed=1.0, sway_sampling_coef=-1.0):
        "Render one utterance. Returns (wav, sample_rate, Prepped)."
        p = self.fe(text)
        ref_p = self.fe(ref_text) if ref_text else None
        wav, sr, _ = self.tts.infer(
            ref_file=str(ref_audio), ref_text=ref_p.model_text if ref_p else "",
            gen_text=p.model_text, seed=seed, nfe_step=nfe_step, cfg_strength=cfg_strength,
            speed=speed, sway_sampling_coef=sway_sampling_coef,
            file_wave=str(out_wav) if out_wav else None)
        return wav, sr, p

def main():
    ap = argparse.ArgumentParser(description="Render Sanskrit text with a fine-tuned F5 checkpoint")
    ap.add_argument("text", nargs="?", help="Devanagari (or any Brahmic) input")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--model", default="F5TTS_v1_Base")
    ap.add_argument("--device", help="cuda | mps | cpu (auto-detected if omitted)")
    ap.add_argument("--manifest", default="data/manifest.qc.jsonl", help="source of reference clips")
    ap.add_argument("--ref-audio", help="explicit reference wav (overrides --manifest selection)")
    ap.add_argument("--ref-text", default="", help="transcript of the reference wav")
    ap.add_argument("--speaker", help="prefer reference clips from this speaker")
    ap.add_argument("--out", default="out/render.wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--no-svara", action="store_true", help="strip accent marks before rendering")
    a = ap.parse_args()

    from dhvani import Frontend
    fe = Frontend(keep_svara=not a.no_svara)
    p = fe(a.text)

    ref_audio, ref_text = a.ref_audio, a.ref_text
    if not ref_audio:
        r = pick_reference(a.manifest, accented=p.accented, speaker=a.speaker, seed=a.seed)
        ref_audio, ref_text = r["audio"], r["text_deva"]
        print(f"reference: {r['id']} ({r['speaker_id']}, accented={bool(r['accented'])}, {r['dur_s']}s)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    r = Renderer(a.ckpt, a.vocab, a.model, a.device, frontend=fe)
    print(f"device   : {r.device}")
    print(f"model txt: {p.model_text}")
    if p.accented: print(f"svara    : {p.svara_str}")
    r.render(a.text, ref_audio, ref_text, out_wav=a.out, seed=a.seed, nfe_step=a.nfe)
    print(f"wrote {a.out}  (seed {a.seed})")

if __name__ == "__main__": main()
