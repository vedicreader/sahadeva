"""Does the synthesized audio actually follow the prescribed Vedic accent?

This is the metric the project exists to move, and there is no established benchmark for it —
vagdhenu reports conjunct accuracy and MOS, neither of which can see accent, because its corpus
has none.

WHAT IS MEASURED. The text prescribes a pitch target per syllable: anudātta low, udātta mid,
svarita high-falling. The synthesized f0 track is compared against that prescription two ways:

  `r`         Pearson correlation between the measured f0 contour (in semitones, relative to the
              utterance median, so it is speaker-independent) and the prescribed contour.
  `separation` mean f0 of anudātta syllables minus mean f0 of svarita syllables, in semitones.
              Negative is correct — anudātta really is the low tone. This is the more
              interpretable of the two, and the harder one to satisfy by accident.

HONEST LIMITATION. Syllable boundaries in synthesized audio are unknown, so the label sequence
is warped uniformly across the voiced span. Recitation is close to isochronous, which is why
this is workable at all, but it is an approximation: a syllable-aligned version (force-align the
output, then measure per syllable) would be strictly better and is the obvious next step. Treat
these numbers as comparative between arms, not as absolute physical measurements.

Always report `shuffled_r` alongside `r`. It is the same computation against a shuffled label
sequence, and it is the control that tells you whether the correlation means anything.

CALIBRATION — READ BEFORE INTERPRETING ANY NUMBER FROM THIS MODULE.
Measured on 60 utterances of the *human* Rudram Namakam recording, i.e. on ground truth that
is correct by definition:

    mean r  +0.126     shuffled control  -0.018
    mean separation  -0.489 st           ordering correct  70.0%

So the ceiling of this metric is roughly r≈0.13 and 70% ordering, not 1.0 and 100%. The gap is
the metric's, not the reciter's: the uniform time warp smears syllable boundaries, and svarita
is a *falling* tone that a flat high target models poorly. A synthesized sample scoring near
70% ordering is matching human performance here; one scoring 50% is at chance.

Run `--audio-field audio` on the same split to regenerate this baseline for your own corpus
before comparing any checkpoint against it. Absolute values are not portable across corpora.
"""
import json, argparse, random
import numpy as np

FMIN, FMAX = 70.0, 400.0
# Pitch targets in arbitrary units; only the ordering and relative spacing matter.
TARGET = {"A": -1.0, "U": 0.0, "S": 1.0, "D": 1.0}

def extract_f0(path, sr=24000, fmin=FMIN, fmax=FMAX):
    "f0 track in Hz with NaN for unvoiced frames, plus the frame times."
    import librosa
    y, sr = librosa.load(str(path), sr=sr)
    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024)
    t = librosa.times_like(f0, sr=sr)
    return f0, t

def fix_octave_jumps(f0):
    """Damp pyin's halving/doubling errors.

    A frame sitting within a few percent of exactly half or double the running median is almost
    always a tracker artifact rather than a real octave leap, and left alone these dominate the
    correlation.
    """
    f = f0.copy()
    v = f[~np.isnan(f)]
    if len(v) < 8: return f
    med = np.median(v)
    for k, mult in ((0.5, 2.0), (2.0, 0.5)):
        idx = ~np.isnan(f) & (np.abs(f - med * k) < 0.12 * med * k)
        f[idx] = f[idx] * mult
    return f

def to_semitones(f0):
    "Convert to semitones relative to the utterance median — removes speaker pitch range."
    v = f0[~np.isnan(f0)]
    if len(v) == 0: return f0
    return 12.0 * np.log2(f0 / np.median(v))

def prescribed_contour(svara_str, n):
    "Expand a per-syllable svara string to `n` frames by uniform time warp."
    if not svara_str or n <= 0: return np.zeros(max(n, 0))
    vals = np.array([TARGET.get(c, 0.0) for c in svara_str])
    idx = np.clip((np.arange(n) * len(vals) // max(n, 1)), 0, len(vals) - 1)
    return vals[idx]

def score(path, svara_str, seed=0):
    """Score one rendered utterance against its prescribed accent.

    Returns a dict; `r` and `separation` are the headline numbers, `shuffled_r` is the control.
    """
    if not svara_str or set(svara_str) <= {"U"}:
        return dict(skipped="unaccented")
    f0, _ = extract_f0(path)
    f0 = fix_octave_jumps(f0)
    st = to_semitones(f0)
    voiced = ~np.isnan(st)
    if voiced.sum() < 16: return dict(skipped="insufficient voiced frames")

    st_v = st[voiced]
    pres = prescribed_contour(svara_str, len(st))[voiced]

    def _r(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6: return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    shuf = list(svara_str); random.Random(seed).shuffle(shuf)
    shuffled = prescribed_contour("".join(shuf), len(st))[voiced]

    lo = st_v[pres < -0.5]
    hi = st_v[pres > 0.5]
    sep = float(lo.mean() - hi.mean()) if len(lo) and len(hi) else float("nan")
    return dict(r=round(_r(st_v, pres), 4), shuffled_r=round(_r(st_v, shuffled), 4),
                separation_st=round(sep, 3) if sep == sep else None,
                ordering_correct=bool(sep < 0) if sep == sep else None,
                n_voiced=int(voiced.sum()), n_syllables=len(svara_str),
                marked_syllables=sum(1 for c in svara_str if c != "U"))

def run(manifest, audio_field="audio", limit=None):
    "Score every accented utterance in a manifest. Returns (per-utterance rows, aggregate)."
    rows = []
    for line in open(manifest):
        u = json.loads(line)
        if not u.get("accented"): continue
        s = score(u[audio_field], u.get("svara", ""))
        if "skipped" in s: continue
        rows.append({**s, "id": u["id"], "speaker_id": u["speaker_id"]})
        if limit and len(rows) >= limit: break
    if not rows: return rows, {}
    agg = dict(
        n=len(rows),
        mean_r=round(float(np.mean([r["r"] for r in rows])), 4),
        mean_shuffled_r=round(float(np.mean([r["shuffled_r"] for r in rows])), 4),
        mean_separation_st=round(float(np.nanmean([r["separation_st"] for r in rows
                                                   if r["separation_st"] is not None])), 3),
        ordering_correct_frac=round(float(np.mean([bool(r["ordering_correct"]) for r in rows])), 4))
    return rows, agg

def main():
    ap = argparse.ArgumentParser(description="Svara contour adherence of rendered audio")
    ap.add_argument("--manifest", default="data/hard_svara.jsonl")
    ap.add_argument("--audio-field", default="audio",
                    help="'audio' scores the reference recordings; point at a field holding your renders to score the model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows, agg = run(a.manifest, a.audio_field, a.limit)
    if not rows: raise SystemExit("no accented utterances scored — is this the hard_svara split?")
    print(f"\nscored {agg['n']} accented utterances\n")
    print(f"  mean r              {agg['mean_r']:+.4f}")
    print(f"  mean shuffled r     {agg['mean_shuffled_r']:+.4f}   <- control")
    print(f"  mean separation     {agg['mean_separation_st']:+.3f} st   (negative is correct)")
    print(f"  ordering correct    {100*agg['ordering_correct_frac']:.1f}%")
    if agg["mean_r"] <= abs(agg["mean_shuffled_r"]):
        print("\n  r is not above the shuffled control — no accent signal detected.")
    if a.out:
        with open(a.out, "w") as f: json.dump(dict(aggregate=agg, rows=rows), f, indent=2)
        print(f"\nwrote {a.out}")

if __name__ == "__main__": main()
