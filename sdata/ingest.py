"""Build a training manifest from the VedicReader SQLite databases.

The alignment stage most TTS pipelines spend their effort on does not exist here, because
VedicReader's XML already carries hand-corrected line timings (`timestamp_fixed="true"`).
Those are better than anything MFA or aeneas would produce from scratch, so this stage reads
them directly rather than re-deriving them. Every line in the corpus is timed.

SPEAKER DISCIPLINE. The recordings are separately sourced, so each title is treated as its own
speaker until proven otherwise — five recordings is five speakers, not 2.2h of one voice.
Mixing them into a single-speaker target would produce an averaged, blurry voice. They are
labelled for multi-speaker adaptation, and `consent` defaults to "unverified" so that nothing
can be used as a voice-cloning target without someone explicitly recording provenance.
"""
import json, sqlite3, argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from dhvani import Frontend, SandhiT

# Categories that carry audio. namavalis parse fine but have no recordings, so they are
# text-only: useful for frontend regression, useless for training.
AUDIO_CATEGORIES = ("stotras", "japas", "mantras")

@dataclass
class Utt:
    "One training utterance. Field names are stable — downstream stages key off them."
    id: str
    title_id: str
    title: str
    category: str
    deity: str
    section: str
    speaker_id: str
    src_audio: str
    audio: str
    start_ms: int
    end_ms: int
    dur_s: float
    text_deva: str
    text_iast: str
    model_text: str
    svara: str
    weights: str
    accented: bool
    n_aksharas: int
    license: str = "unknown"
    consent: str = "unverified"
    source: str = "vedicreader"
    qc: dict = field(default_factory=dict)

def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")

def read_db(db_path, vr_root, fe, out_dir):
    "Yield Utt records for one shlokas database."
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    titles = {r["id"]: r for r in con.execute("select * from titles")}
    audio = {r["title_id"]: r for r in con.execute("select * from files where type='audio'")}

    q = """select rowid, * from shloka_lines
           where ignore_sync=0 and exclude_from_display=0
             and start_time_ms is not null and end_time_ms is not null
           order by title_id, section_order, rowid"""
    seen = {}
    for r in con.execute(q):
        tid = r["title_id"]
        if tid not in audio: continue                    # text-only title, e.g. namavalis
        t = titles.get(tid)
        if t is None: continue
        txt = (r["text"] or "").strip()
        if not txt: continue

        n = seen[tid] = seen.get(tid, -1) + 1
        name = _slug(t["name"])
        cat = t["category"]
        p = fe(txt)
        start, end = int(r["start_time_ms"]), int(r["end_time_ms"])
        yield Utt(
            id=f"{cat}/{name}/{n:05d}", title_id=tid, title=t["name"], category=cat,
            deity=t["deity"] or "", section=r["section_name"] or "",
            speaker_id=f"spk_{name}",                     # one speaker per recording — see module docstring
            src_audio=str(Path(vr_root) / audio[tid]["path"]),
            audio=str(Path(out_dir) / cat / name / f"{n:05d}.wav"),
            start_ms=start, end_ms=end, dur_s=round((end - start) / 1000.0, 3),
            text_deva=txt, text_iast=r["iast"] or "",
            model_text=p.model_text, svara=p.svara_str, weights=p.weight_str,
            accented=p.accented, n_aksharas=p.n_aksharas)

def build(vr_root, out_manifest, out_dir="data/wav", keep_svara=True, sandhi=SandhiT.PLAIN):
    "Scan every shlokas DB under `vr_root` and write a JSONL manifest."
    fe = Frontend(sandhi=sandhi, keep_svara=keep_svara)
    dbs = sorted((Path(vr_root) / "assets/db/shlokas").glob("*.db"))
    if not dbs: raise SystemExit(f"no shlokas databases under {vr_root}")
    utts = []
    for db in dbs:
        try: utts.extend(read_db(db, vr_root, fe, out_dir))
        except sqlite3.Error as e: print(f"  skip {db.name}: {e}")
    utts.sort(key=lambda u: u.id)
    Path(out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w") as f:
        for u in utts: f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
    return utts

def summarize(utts):
    "Print a data card — speakers, durations, accent coverage."
    by_spk = {}
    for u in utts: by_spk.setdefault(u.speaker_id, []).append(u)
    tot = sum(u.dur_s for u in utts)
    print(f"\n{len(utts)} utterances, {tot/3600:.2f}h across {len(by_spk)} speakers\n")
    print(f"{'speaker':34}{'utts':>6}{'hours':>8}{'accented':>10}{'med_s':>8}")
    for spk, us in sorted(by_spk.items()):
        d = sorted(u.dur_s for u in us)
        print(f"{spk:34}{len(us):>6}{sum(d)/3600:>8.2f}{sum(u.accented for u in us):>10}{d[len(d)//2]:>8.1f}")
    ok = sum(1 for u in utts if 3 <= u.dur_s <= 15)
    print(f"\nin 3-15s window: {ok}/{len(utts)} ({100*ok/max(len(utts),1):.0f}%)")
    print(f"accented utts   : {sum(u.accented for u in utts)}")
    unverified = sum(1 for u in utts if u.consent != "verified")
    if unverified:
        print(f"\n  {unverified} utterances have consent='unverified'. Record provenance in the\n"
              f"  manifest before using any of this as a voice-cloning target.")

def main():
    ap = argparse.ArgumentParser(description="VedicReader databases -> training manifest")
    ap.add_argument("--vr-root", default="/home/user/vedicreader")
    ap.add_argument("--out", default="data/manifest.jsonl")
    ap.add_argument("--wav-dir", default="data/wav")
    ap.add_argument("--no-svara", action="store_true", help="ablation arm: strip accent marks from model text")
    ap.add_argument("--sandhi", default="plain", choices=[s.value for s in SandhiT])
    a = ap.parse_args()
    utts = build(a.vr_root, a.out, a.wav_dir, keep_svara=not a.no_svara, sandhi=SandhiT(a.sandhi))
    summarize(utts)
    print(f"\nwrote {a.out}")

if __name__ == "__main__": main()
