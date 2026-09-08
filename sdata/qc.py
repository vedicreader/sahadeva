"""Quality gates over the segmented wavs.

The timings are hand-verified, so this is not hunting for gross misalignment. It is looking
for the things that survive good timings and still poison a fine-tune: clipped source audio,
segments that are mostly silence, and — the useful one — utterances whose speaking rate is far
from the corpus norm, which is the signature of a text/audio mismatch that the eye misses.

Nothing is deleted. Every utterance gets a `qc` block and a `qc_pass` flag; `splits.py` decides
what to do about it, so a threshold change never means re-cutting audio.
"""
import json, argparse
from pathlib import Path
import numpy as np, soundfile as sf

MIN_DUR, MAX_DUR = 1.0, 20.0        # hard bounds; the 3-15s preference is a split-time concern
MAX_SILENCE_FRAC = 0.60
MIN_SNR_DB = 12.0
CLIP_THRESH, MAX_CLIP_FRAC = 0.99, 0.001
RATE_Z = 3.0                        # speaking-rate outlier cutoff, in MADs

def _frames(x, sr, win=0.02):
    n = max(1, int(win * sr))
    if len(x) < n: return np.array([np.sqrt((x ** 2).mean())]) if len(x) else np.array([0.0])
    return np.sqrt((x[:len(x) // n * n].reshape(-1, n) ** 2).mean(axis=1))

def measure(path):
    "Per-file acoustic measurements. Returns a dict, or {'error': ...} if unreadable."
    try: x, sr = sf.read(str(path), dtype="float32")
    except Exception as e: return dict(error=str(e))
    return measure_array(x, sr)

def measure_array(x, sr):
    "Same measurements over an in-memory mono signal — used by the forced-aligned clip builder."
    if x.ndim > 1: x = x.mean(axis=1)
    if len(x) == 0: return dict(error="empty")
    r = _frames(x, sr)
    noise, speech = np.percentile(r, 10), np.percentile(r, 90)
    snr = 20 * np.log10(speech / noise) if noise > 1e-8 and speech > 0 else 60.0
    thr = max(noise * 2, speech * 0.05)
    return dict(dur_s=round(len(x) / sr, 3), peak=round(float(np.abs(x).max()), 4),
                clip_frac=round(float((np.abs(x) >= CLIP_THRESH).mean()), 6),
                silence_frac=round(float((r < thr).mean()), 4),
                snr_db=round(float(min(snr, 60.0)), 2),
                rms=round(float(np.sqrt((x ** 2).mean())), 5))

def _rate_outliers(utts):
    """Flag utterances whose aksharas-per-second is far from the corpus median.

    Computed per speaker, since recitation tempo varies a lot between reciters — a global
    threshold would flag an entire slow recording rather than its bad lines.
    """
    flags = {}
    by_spk = {}
    for u in utts:
        d = u.get("qc", {}).get("dur_s") or u["dur_s"]
        if d and u["n_aksharas"]: by_spk.setdefault(u["speaker_id"], []).append((u["id"], u["n_aksharas"] / d))
    for spk, pairs in by_spk.items():
        rates = np.array([r for _, r in pairs])
        med = np.median(rates)
        mad = np.median(np.abs(rates - med)) or 1e-6
        for (uid, r) in pairs:
            z = abs(r - med) / (1.4826 * mad)
            if z > RATE_Z: flags[uid] = round(float(z), 2)
    return flags

def run(manifest, out=None):
    "Measure every segmented wav, annotate the manifest, return the annotated records."
    utts = [json.loads(l) for l in open(manifest)]
    for u in utts:
        p = Path(u["audio"])
        u["qc"] = measure(p) if p.exists() else dict(error="missing")

    outliers = _rate_outliers(utts)
    for u in utts:
        q = u["qc"]
        u["qc"]["rate_z"] = outliers.get(u["id"], 0.0)
        reasons = []
        if "error" in q: reasons.append(q["error"])
        else:
            if not (MIN_DUR <= q["dur_s"] <= MAX_DUR): reasons.append("duration")
            if q["clip_frac"] > MAX_CLIP_FRAC:         reasons.append("clipping")
            if q["silence_frac"] > MAX_SILENCE_FRAC:   reasons.append("silence")
            if q["snr_db"] < MIN_SNR_DB:               reasons.append("snr")
            if u["qc"]["rate_z"] > RATE_Z:             reasons.append("speaking_rate")
        u["qc"]["reasons"] = reasons
        u["qc_pass"] = not reasons

    if out:
        with open(out, "w") as f:
            for u in utts: f.write(json.dumps(u, ensure_ascii=False) + "\n")
    return utts

def summarize(utts):
    ok = [u for u in utts if u["qc_pass"]]
    print(f"\n{len(ok)}/{len(utts)} passed QC ({sum(u['dur_s'] for u in ok)/3600:.2f}h usable)\n")
    counts = {}
    for u in utts:
        for r in u["qc"]["reasons"]: counts[r] = counts.get(r, 0) + 1
    for r, n in sorted(counts.items(), key=lambda kv: -kv[1]): print(f"  {r:16}{n:>5}")
    good = [u["qc"] for u in utts if "error" not in u["qc"]]
    if good:
        snr = np.array([g["snr_db"] for g in good])
        print(f"\n  SNR dB  median {np.median(snr):.1f}  p10 {np.percentile(snr,10):.1f}")

def main():
    ap = argparse.ArgumentParser(description="Quality-gate the segmented utterances")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", default="data/manifest.qc.jsonl")
    a = ap.parse_args()
    utts = run(a.manifest, a.out)
    summarize(utts)
    print(f"\nwrote {a.out}")

if __name__ == "__main__": main()
