"""Slice source recordings into per-utterance wavs using the manifest's timings.

Pure Python — soundfile decodes mp3 with random access, so there is no ffmpeg dependency and
this behaves identically on Linux and macOS. Each source file is decoded once and held while
all of its utterances are cut, rather than reopened per slice.

Output is 24 kHz mono float32 wav, loudness-normalized, which is what both F5/IndicF5 and
BigVGAN-v2 expect.
"""
import json, argparse
from pathlib import Path
import numpy as np, soundfile as sf, soxr, pyloudnorm

SR = 24000
TARGET_LUFS = -23.0        # EBU R128 speech target; keeps headroom for the chant's dynamics
PAD_MS = 30                # small pad each side — line boundaries are marked at onsets

def load_source(path, sr=SR):
    "Decode a source recording to mono float32 at `sr`."
    d, in_sr = sf.read(str(path), dtype="float32", always_2d=True)
    d = d.mean(axis=1)
    return d if in_sr == sr else soxr.resample(d, in_sr, sr)

def normalize_loudness(x, target=TARGET_LUFS, sr=SR):
    """Loudness-normalize, then peak-limit.

    Segments under ~0.4s are shorter than the BS.1770 gating block, so they are peak-normalized
    instead — measuring loudness on them returns -inf and would blow up the gain.
    """
    if len(x) < int(0.4 * sr):
        peak = float(np.abs(x).max())
        return x * (0.7 / peak) if peak > 0 else x
    loud = pyloudnorm.Meter(sr).integrated_loudness(x)
    if not np.isfinite(loud): return x
    x = x * (10.0 ** ((target - loud) / 20.0))
    peak = float(np.abs(x).max())
    return x * (0.99 / peak) if peak > 0.99 else x

def cut(src, start_ms, end_ms, sr=SR, pad_ms=PAD_MS):
    "Extract [start_ms, end_ms) with padding, clamped to the source bounds."
    a = max(0, int((start_ms - pad_ms) * sr / 1000))
    b = min(len(src), int((end_ms + pad_ms) * sr / 1000))
    return src[a:b] if b > a else np.zeros(0, dtype="float32")

def run(manifest, limit=None, overwrite=False, pad_ms=PAD_MS):
    "Segment every utterance in the manifest. Returns (written, skipped, missing_sources)."
    utts = [json.loads(l) for l in open(manifest)]
    if limit: utts = utts[:limit]

    by_src = {}
    for u in utts: by_src.setdefault(u["src_audio"], []).append(u)

    written = skipped = 0
    missing = []
    for src_path, group in sorted(by_src.items()):
        p = Path(src_path)
        if not p.exists(): missing.append(src_path); continue
        if p.stat().st_size < 1024 and p.read_bytes().startswith(b"version https://git-lfs"):
            missing.append(f"{src_path} (git-lfs pointer — run: git lfs pull)"); continue

        print(f"  {p.name} -> {len(group)} utts", flush=True)
        src = load_source(p)
        for u in group:
            out = Path(u["audio"])
            if out.exists() and not overwrite: skipped += 1; continue
            seg = cut(src, u["start_ms"], u["end_ms"], pad_ms=pad_ms)
            if len(seg) == 0: skipped += 1; continue
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), normalize_loudness(seg), SR, subtype="PCM_16")
            written += 1
    return written, skipped, missing

def main():
    ap = argparse.ArgumentParser(description="Cut source recordings into per-utterance wavs")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pad-ms", type=int, default=PAD_MS)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    written, skipped, missing = run(a.manifest, a.limit, a.overwrite, a.pad_ms)
    print(f"\nwrote {written}, skipped {skipped}")
    if missing:
        print(f"\n{len(missing)} source recordings unavailable:")
        for m in missing[:10]: print(f"  {m}")
        print("\nThe VedicReader mp3s are stored in git-lfs. Fetch them with:\n"
              "  cd <vedicreader> && git lfs pull --include='static/vedic_texts/**/*.mp3'")

if __name__ == "__main__": main()
