"""Corpus-level regression tests against the real VedicReader databases.

Hand-written goldens catch what you thought to write down; the corpus catches what you didn't.
Every bug found in the frontend so far came from this file, not from test_dhvani.py.

Skips cleanly when the VedicReader checkout isn't present, so CI stays green without it.
"""
import glob, sqlite3
import pytest
from dhvani import prep
from dhvani.svara import extract_svara, restore_svara, normalize
from dhvani.meter import aksharas

DBS = sorted(glob.glob("/home/user/vedicreader/assets/db/shlokas/*.db"))
pytestmark = pytest.mark.skipif(not DBS, reason="VedicReader databases not available")

# 2 of 2,619 lines carry an accent mark inside a conjunct cluster — see
# test_dhvani.test_known_lossy_case_mark_inside_conjunct. Ratchet: this may go down, never up.
MAX_ROUNDTRIP_MISMATCHES = 2

def _lines():
    for db in DBS:
        try: rows = sqlite3.connect(db).execute("select text from shloka_lines")
        except sqlite3.Error: continue
        for (t,) in rows:
            if t and t.strip(): yield t

def test_corpus_is_present():
    assert sum(1 for _ in _lines()) > 2000

def test_svara_labels_align_on_every_line():
    """The invariant the whole svara design rests on: one label per akshara, every line.

    This is what caught the virāma-lookback bug, where accented and stripped copies of the
    same text segmented into different numbers of aksharas.
    """
    bad = [t for t in _lines() if (p := prep(t)).n_aksharas != len(p.svara)]
    assert not bad, f"{len(bad)} lines misaligned, e.g. {bad[:3]}"

def test_corpus_roundtrip_within_ratchet():
    bad = [t for t in _lines()
           if restore_svara(*extract_svara(s := normalize(t))) != s]
    assert len(bad) <= MAX_ROUNDTRIP_MISMATCHES, f"{len(bad)} mismatches, e.g. {bad[:3]}"

def test_no_line_crashes_the_frontend():
    """Whole corpus through the full pipeline — no exceptions, no empty model text.

    Note what is deliberately NOT asserted: len(weight_str) == n_aksharas. Metrical weight is
    computed on the phonemic grid and svara on the orthographic akshara grid, and the two do
    not coincide in general — 'तर्पणम् मन्त्रः' is 6 aksharas but scans as 5 syllables
    (tar-pa-ṇam-man-traḥ), which is the correct scansion. Only the svara channel needs to be
    index-aligned to aksharas, and that is asserted above.
    """
    for t in _lines():
        p = prep(t)
        assert p.model_text is not None
        assert p.weight_str or not p.n_aksharas

def test_danda_never_reaches_model_text():
    "Daṇḍas are punctuation, not phones — they must not survive into what the model reads."
    assert not any(c in prep(t).model_text for t in _lines() for c in "।॥|")
