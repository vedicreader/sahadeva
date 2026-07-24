"""Conjunct rendering accuracy — vagdhenu's headline metric, and the bar to clear.

Vagdhenu reports 100% correct rendering of conjuncts including retroflex aspirates. That claim
is about a specific, checkable thing: given a verse containing kṣ / jñ / ṣṭ / ḍḍh, does the
synthesized audio actually contain that cluster, or does the model smooth it into something
easier?

Method is an ASR round-trip: synthesize the hard-set text, transcribe it back, and check
cluster survival in SLP1 space. Two numbers come out —

  cluster_recall   fraction of clusters present in the reference text that survive into the
                   transcript. This is the headline.
  cer              overall character error rate, for context; a model can preserve clusters
                   while being broadly unintelligible, and recall alone would hide that.

ASR IS NOT BUNDLED. No Sanskrit ASR ships with this repo — pass a callable, or use
`--transcripts` with a JSONL you produced separately. IndicWhisper and the Vedavani-benchmarked
models are the usual choices. Whichever you use, transcribe the *reference recordings* too and
report that number alongside: ASR error is large for Sanskrit, and a cluster the recognizer
never gets right on human audio tells you nothing about the model.
"""
import json, argparse
from collections import Counter
from dhvani import to_slp1, to_deva
from dhvani.svara import strip_svara
from sdata.splits import HARD_CLUSTERS, RETROFLEX_ASPIRATE

TRACKED = list(HARD_CLUSTERS) + sorted(RETROFLEX_ASPIRATE)

def _slp(text):
    return to_slp1(strip_svara(to_deva(text)), fix_vocalic_rr=False, fix_om=True)

def clusters_in(text):
    "Multiset of tracked clusters present in a string."
    s = _slp(text)
    c = Counter()
    for cl in TRACKED:
        n = s.count(cl)
        if n: c[cl] = n
    return c

def score_pair(ref_text, hyp_text):
    "Cluster survival for one utterance."
    ref, hyp = clusters_in(ref_text), clusters_in(hyp_text)
    total = sum(ref.values())
    kept = sum(min(n, hyp.get(cl, 0)) for cl, n in ref.items())
    missed = {cl: n - min(n, hyp.get(cl, 0)) for cl, n in ref.items() if n > hyp.get(cl, 0)}
    return dict(n_clusters=total, kept=kept,
                recall=round(kept / total, 4) if total else None, missed=missed)

def run(pairs):
    """Score (reference_text, transcript) pairs. Returns (rows, aggregate).

    Recall is pooled over clusters rather than averaged over utterances, so a verse with six
    hard clusters counts six times as much as one with a single cluster.
    """
    from sahaeval.cer import run as cer_run
    rows, kept, total = [], 0, 0
    per_cluster = Counter(); per_cluster_kept = Counter()
    for ref, hyp in pairs:
        s = score_pair(ref, hyp)
        if s["n_clusters"]:
            kept += s["kept"]; total += s["n_clusters"]
            for cl, n in clusters_in(ref).items():
                per_cluster[cl] += n
                per_cluster_kept[cl] += min(n, clusters_in(hyp).get(cl, 0))
        rows.append({**s, "ref": ref, "hyp": hyp})
    _, cer_agg = cer_run(pairs)
    agg = dict(n=len(rows), n_clusters=total,
               cluster_recall=round(kept / total, 4) if total else None,
               cer=cer_agg["cer"], wer=cer_agg["wer"],
               per_cluster={cl: round(per_cluster_kept[cl] / n, 3)
                            for cl, n in per_cluster.most_common()})
    return rows, agg

def main():
    ap = argparse.ArgumentParser(description="Conjunct rendering accuracy via ASR round-trip")
    ap.add_argument("--transcripts", required=True,
                    help="JSONL with {id, ref, hyp} — hyp being your ASR transcript of the rendered audio")
    a = ap.parse_args()
    pairs = [(d["ref"], d["hyp"]) for d in map(json.loads, open(a.transcripts))]
    rows, agg = run(pairs)
    print(f"\n{agg['n']} utterances, {agg['n_clusters']} tracked clusters\n")
    print(f"  cluster recall  {100*agg['cluster_recall']:.1f}%" if agg["cluster_recall"] is not None
          else "  no tracked clusters found")
    print(f"  CER             {100*agg['cer']:.2f}%")
    print(f"  WER             {100*agg['wer']:.2f}%")
    if agg["per_cluster"]:
        print("\n  per cluster (SLP1):")
        for cl, r in agg["per_cluster"].items(): print(f"    {cl:4} {100*r:5.1f}%")
    print("\n  Report the same numbers for the human reference recordings before drawing a\n"
          "  conclusion — Sanskrit ASR error is large and asymmetric across clusters.")

if __name__ == "__main__": main()
