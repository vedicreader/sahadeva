"""Turn a sahadeva split into the dataset F5-TTS's fine-tune path expects.

Format is taken from f5_tts/train/datasets/prepare_csv_wavs.py (F5-TTS 1.1.22):

    audio_file|text          <- header required, "|" delimiter, absolute audio paths

The vocab preflight is the point of this module. In fine-tune mode `prepare_csv_wavs.py`
*copies the pretrained vocab* and ignores the vocab set it just computed from your text — so a
character your text uses and the checkpoint does not know is silently dropped rather than
reported. For accented Sanskrit that is exactly the failure you cannot afford: the accent marks
are the signal, and they would vanish without a single warning.

Checked against the IndicF5 vocab, U+0951 and U+0952 are present, so Vedic svara survives
tokenization and no vocab extension is needed. U+1CDA is absent, which is why dhvani folds it
to U+0951 upstream. This module re-verifies that against whatever vocab you actually point it
at, rather than trusting the note above.
"""
import json, csv, argparse, unicodedata
from collections import Counter
from pathlib import Path

def load_vocab(path):
    "Read an F5-TTS vocab.txt — one token per line, order is the token id."
    with open(path, encoding="utf-8") as f:
        return [ln[:-1] if ln.endswith("\n") else ln for ln in f]

def vocab_coverage(rows, vocab):
    """Which characters of the model text are missing from the vocab.

    Returns (oov_counter, n_affected_rows). Space is treated as covered — F5's vocab carries it
    explicitly, but a trailing newline strip can make it look absent.
    """
    vset = set(vocab) | {" "}
    oov, affected = Counter(), 0
    for r in rows:
        bad = [c for c in r["text"] if c not in vset]
        if bad:
            affected += 1
            oov.update(bad)
    return oov, affected

def describe(chars):
    "Human-readable OOV report — codepoint and name, since these are invisible combining marks."
    return [f"U+{ord(c):04X} {unicodedata.name(c, '?')}" for c in chars]

def build(split_path, out_csv, text_field="model_text", abs_paths=True, require_audio=True):
    "Write metadata.csv from a split JSONL. Returns the rows written."
    rows, missing = [], []
    for line in open(split_path):
        u = json.loads(line)
        text = (u.get(text_field) or "").strip()
        if not text: continue
        p = Path(u["audio"])
        if require_audio and not p.exists(): missing.append(str(p)); continue
        rows.append(dict(audio_file=str(p.resolve() if abs_paths else p), text=text))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["audio_file", "text"])
        for r in rows: w.writerow([r["audio_file"], r["text"]])
    return rows, missing

def main():
    ap = argparse.ArgumentParser(description="sahadeva split -> F5-TTS metadata.csv")
    ap.add_argument("--split", default="data/train.jsonl")
    ap.add_argument("--out", default="data/f5/metadata.csv")
    ap.add_argument("--vocab", help="path to the checkpoint's vocab.txt — strongly recommended")
    ap.add_argument("--text-field", default="model_text",
                    help="'model_text' is Kannada-routed; use 'text_deva' only to reproduce the schwa-deletion bug")
    ap.add_argument("--allow-missing-audio", action="store_true")
    a = ap.parse_args()

    rows, missing = build(a.split, a.out, a.text_field, require_audio=not a.allow_missing_audio)
    print(f"wrote {a.out}: {len(rows)} rows")
    if missing:
        print(f"\n  {len(missing)} utterances have no segmented wav — run sd-segment first.")
        for m in missing[:5]: print(f"    {m}")

    if a.vocab:
        vocab = load_vocab(a.vocab)
        oov, affected = vocab_coverage(rows, vocab)
        print(f"\nvocab: {len(vocab)} tokens")
        if not oov:
            print("  full coverage — every character in the training text is in the checkpoint vocab")
        else:
            print(f"  {len(oov)} out-of-vocab characters across {affected} rows:")
            for c, n in oov.most_common(20):
                print(f"    {describe([c])[0]:<52} x{n}")
            print("\n  F5-TTS's fine-tune path copies the pretrained vocab and drops these silently.\n"
                  "  Either normalize them out in dhvani, or extend the vocab and resize the\n"
                  "  embedding before training.")
    else:
        print("\n  no --vocab given, so OOV characters were not checked. For accented text this is\n"
              "  the check that matters — accent marks drop silently if the checkpoint lacks them.")

    print(f"\nnext:\n"
          f"  python -m f5_tts.train.datasets.prepare_csv_wavs {a.out} data/f5/sahadeva_char\n"
          f"  # then see configs/finetune_16gb.md for the training invocation")

if __name__ == "__main__": main()
