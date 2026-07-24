"""Train/dev/test splits, plus the two held-out hard sets the evaluation is built around.

Standard TTS practice is a speaker-disjoint split, but that is the wrong instrument here: with
six speakers and 2.1h, holding a speaker out means holding out a sixth of the corpus and one
entire text. Splits are stratified *within* speaker instead, so every voice is represented on
both sides, and generalization is measured by the hard sets rather than by speaker disjointness.

Two hard sets, both held out of training entirely:

  conjunct — utterances dense in retroflex aspirates and difficult clusters (kṣ, jñ, ṣṭ, ḍḍh).
             This is vagdhenu's headline metric; 100% correct rendering is the bar.
  svara    — utterances dense in Vedic accent marks. This is the metric this project adds, and
             it is the one the corpus is uniquely able to support.
"""
import json, argparse, random
from collections import defaultdict
from dhvani import to_slp1

# SLP1: w=ṭ W=ṭh q=ḍ Q=ḍh R=ṇ z=ṣ Y=ñ. Weighted by how reliably each breaks a weak model.
HARD_CLUSTERS = {"kz": 3, "jY": 3, "zw": 3, "zW": 4, "qQ": 4, "wW": 3, "Rw": 2, "Rq": 2,
                 "hR": 2, "hn": 2, "kD": 2, "tT": 1, "dD": 1, "pP": 1, "bB": 1}
RETROFLEX_ASPIRATE = set("WQ")

def conjunct_score(text_deva):
    "How punishing an utterance is for conjunct rendering. Higher is harder."
    slp = to_slp1(text_deva, fix_vocalic_rr=False, fix_om=False)
    s = sum(w * slp.count(c) for c, w in HARD_CLUSTERS.items())
    s += 4 * sum(slp.count(c) for c in RETROFLEX_ASPIRATE)
    return s

def svara_score(svara_str):
    "Count of marked syllables — unmarked (udātta) syllables carry no accent information."
    return sum(1 for c in svara_str if c != "U")

def _pick_hard(utts, score_fn, n, min_score=1):
    "Take the top-n by score, preferring mid-length utterances so eval is not dominated by outliers."
    cands = [(score_fn(u), u) for u in utts if 3.0 <= u["dur_s"] <= 15.0]
    cands = [(s, u) for s, u in cands if s >= min_score]
    cands.sort(key=lambda su: -su[0])
    return [u for _, u in cands[:n]]

def run(manifest, out_dir="data", n_conjunct=60, n_svara=60, dev_frac=0.05, seed=0,
        require_qc=True):
    utts = [json.loads(l) for l in open(manifest)]
    pool = [u for u in utts if u.get("qc_pass", True)] if require_qc else list(utts)

    hard_c = _pick_hard(pool, lambda u: conjunct_score(u["text_deva"]), n_conjunct)
    taken = {u["id"] for u in hard_c}
    hard_s = _pick_hard([u for u in pool if u["id"] not in taken],
                        lambda u: svara_score(u["svara"]), n_svara)
    taken |= {u["id"] for u in hard_s}

    rest = [u for u in pool if u["id"] not in taken]
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for u in rest: by_spk[u["speaker_id"]].append(u)

    train, dev = [], []
    for spk, us in sorted(by_spk.items()):
        us = sorted(us, key=lambda u: u["id"]); rng.shuffle(us)
        k = max(1, int(len(us) * dev_frac))
        dev += us[:k]; train += us[k:]

    splits = dict(train=train, dev=dev, hard_conjunct=hard_c, hard_svara=hard_s)
    for name, rows in splits.items():
        rows.sort(key=lambda u: u["id"])
        with open(f"{out_dir}/{name}.jsonl", "w") as f:
            for u in rows: f.write(json.dumps(u, ensure_ascii=False) + "\n")
    return splits

def summarize(splits):
    print()
    for name, rows in splits.items():
        h = sum(u["dur_s"] for u in rows) / 3600
        acc = sum(u["accented"] for u in rows)
        print(f"{name:16}{len(rows):>6} utts{h:>7.2f}h   accented {acc:>4}   speakers {len({u['speaker_id'] for u in rows})}")
    hs = splits["hard_svara"]
    if hs:
        marked = sum(svara_score(u["svara"]) for u in hs)
        print(f"\nhard_svara holds {marked} marked syllables across {len(hs)} utterances")
    hc = splits["hard_conjunct"]
    if hc:
        print(f"hard_conjunct top score {conjunct_score(hc[0]['text_deva'])}, "
              f"median {conjunct_score(hc[len(hc)//2]['text_deva'])}")

def main():
    ap = argparse.ArgumentParser(description="Build splits and hard-set holdouts")
    ap.add_argument("--manifest", default="data/manifest.qc.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--n-conjunct", type=int, default=60)
    ap.add_argument("--n-svara", type=int, default=60)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-qc", action="store_true", help="ignore qc_pass when pooling")
    a = ap.parse_args()
    splits = run(a.manifest, a.out_dir, a.n_conjunct, a.n_svara, a.dev_frac, a.seed,
                 require_qc=not a.no_qc)
    summarize(splits)
    print(f"\nwrote {a.out_dir}/{{train,dev,hard_conjunct,hard_svara}}.jsonl")

if __name__ == "__main__": main()
