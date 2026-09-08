"""CER/WER for Sanskrit ASR round-trip, computed on SLP1.

Scoring in SLP1 rather than in whatever script the ASR emits is what makes the numbers mean
anything: SLP1 is one character per phone, so edit distance is phone distance. Comparing raw
Devanagari against raw Kannada output would score script differences as errors, and comparing
in IAST would count the combining accent marks as separate characters.

Accent marks are stripped before scoring by default. No Sanskrit ASR transcribes svara, so
leaving them in would charge the model for information the reference transcript cannot contain
— `sahaeval.svara` is where accent is scored instead.
"""
import json, argparse
from dhvani import to_slp1, to_deva
from dhvani.svara import strip_svara

def _norm(text, strip_accent=True):
    "Any script -> SLP1, accent optionally removed, whitespace collapsed."
    t = to_deva(text)
    if strip_accent: t = strip_svara(t)
    return " ".join(to_slp1(t, fix_vocalic_rr=False, fix_om=True).split())

def edit_distance(a, b):
    "Levenshtein, O(min(len)) memory."
    if len(a) < len(b): a, b = b, a
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def cer(ref, hyp, strip_accent=True):
    "Character error rate on SLP1 phones."
    r, h = _norm(ref, strip_accent).replace(" ", ""), _norm(hyp, strip_accent).replace(" ", "")
    return edit_distance(r, h) / max(len(r), 1)

def wer(ref, hyp, strip_accent=True):
    "Word error rate on SLP1 tokens."
    r, h = _norm(ref, strip_accent).split(), _norm(hyp, strip_accent).split()
    return edit_distance(r, h) / max(len(r), 1)

def run(pairs, strip_accent=True):
    """Score (reference_text, hypothesis_text) pairs.

    Aggregate CER is length-weighted, not a mean of per-utterance rates — a mean over
    utterances lets short ones dominate and is the usual way this number gets quietly inflated.
    """
    rows, num_c, den_c, num_w, den_w = [], 0, 0, 0, 0
    for ref, hyp in pairs:
        r = _norm(ref, strip_accent).replace(" ", "")
        h = _norm(hyp, strip_accent).replace(" ", "")
        rw, hw = _norm(ref, strip_accent).split(), _norm(hyp, strip_accent).split()
        dc, dw = edit_distance(r, h), edit_distance(rw, hw)
        num_c += dc; den_c += len(r); num_w += dw; den_w += len(rw)
        rows.append(dict(cer=round(dc / max(len(r), 1), 4), wer=round(dw / max(len(rw), 1), 4),
                         ref=ref, hyp=hyp))
    return rows, dict(cer=round(num_c / max(den_c, 1), 4), wer=round(num_w / max(den_w, 1), 4),
                      n=len(rows))

def main():
    ap = argparse.ArgumentParser(description="CER/WER over ASR round-trip transcripts")
    ap.add_argument("--pairs", required=True,
                    help="JSONL with {ref, hyp} per line — produced by sahaeval.conjunct or your own ASR run")
    ap.add_argument("--keep-accent", action="store_true")
    a = ap.parse_args()
    pairs = [(d["ref"], d["hyp"]) for d in map(json.loads, open(a.pairs))]
    rows, agg = run(pairs, strip_accent=not a.keep_accent)
    print(f"\n{agg['n']} utterances\n  CER {100*agg['cer']:.2f}%\n  WER {100*agg['wer']:.2f}%")
    worst = sorted(rows, key=lambda r: -r["cer"])[:5]
    if worst:
        print("\nworst by CER:")
        for w in worst: print(f"  {100*w['cer']:5.1f}%  {w['ref'][:60]}")

if __name__ == "__main__": main()
