"""Vedic accent (svara) — extraction, stripping, normalization, restoration.

Why this module exists: svara *is* the melody of Vedic recitation, not decoration. Rudram
Namakam in the VedicReader corpus carries ~1,600 accent marks. Neither the vagdhenu frontend
nor any general Indic TTS frontend does anything with them — vagdhenu trains on Bhāgavatam,
which is classical metrical verse with no accent. So they pass through as inert characters.

A NOTE ON UNICODE NAMES, WHICH ARE MISLEADING HERE.
Unicode calls U+0951 "DEVANAGARI STRESS SIGN UDATTA". In the Taittirīya (Kṛṣṇa Yajurveda)
printing convention that Rudram and virtually all South Indian Vedic texts use, that vertical
stroke above marks *svarita*, not udātta — and udātta itself is left unmarked. Reading U+0951
as udātta is a real and easy mistake. This module follows the Taittirīya reading by default;
pass `tradition="unicode"` if you have a text that genuinely follows the Unicode gloss.

VOCAB NOTE. U+0951 and U+0952 are both present in the IndicF5 vocab, so they survive
tokenization — but IndicF5 trained on IndicVoices-R, which contains no accented Vedic text, so
those embeddings are effectively untrained. U+1CDA (dīrgha svarita) is *absent* from the vocab
entirely. `normalize()` handles the OOV case; teaching the in-vocab marks is what fine-tuning
on this corpus is for.
"""
from enum import Enum
from .meter import aksharas

ANUDATTA = "॒"        # ॒  line below
SVARITA = "॑"         # ॑  vertical line above (Taittirīya: svarita; Unicode gloss: udātta)
DIRGHA_SVARITA = "᳚"  # ᳚  double vertical above — independent/long svarita. OOV in IndicF5.
GRAVE = "॔"           # ॔  rare; treated as svarita-class

from .translit import VEDIC_MARKS as MARKS   # single source of truth; the syllabifier needs it too

class SvaraT(str, Enum):
    UDATTA = "U"          # unmarked in Taittirīya texts
    ANUDATTA = "A"
    SVARITA = "S"
    DIRGHA_SVARITA = "D"

_TAITTIRIYA = {ANUDATTA: SvaraT.ANUDATTA, SVARITA: SvaraT.SVARITA,
               DIRGHA_SVARITA: SvaraT.DIRGHA_SVARITA, GRAVE: SvaraT.SVARITA}
# The literal Unicode reading, for texts that actually follow it.
_UNICODE = {ANUDATTA: SvaraT.ANUDATTA, SVARITA: SvaraT.UDATTA,
            DIRGHA_SVARITA: SvaraT.DIRGHA_SVARITA, GRAVE: SvaraT.SVARITA}

# Canonical mark for each label, per tradition. Built explicitly rather than by inverting the
# tables above, which would collide: GRAVE and SVARITA both read as SvaraT.SVARITA.
# Under the 'unicode' reading, UDATTA is deliberately absent — an explicitly marked udātta and an
# unmarked syllable both label as UDATTA there, so that reading is not round-trippable.
_CANON = {
    "taittiriya": {SvaraT.ANUDATTA: ANUDATTA, SvaraT.SVARITA: SVARITA,
                   SvaraT.DIRGHA_SVARITA: DIRGHA_SVARITA},
    "unicode":    {SvaraT.ANUDATTA: ANUDATTA, SvaraT.SVARITA: GRAVE,
                   SvaraT.DIRGHA_SVARITA: DIRGHA_SVARITA},
}

def _table(tradition):
    if tradition == "taittiriya": return _TAITTIRIYA
    if tradition == "unicode": return _UNICODE
    raise ValueError(f"unknown tradition {tradition!r} — use 'taittiriya' or 'unicode'")

_DEVA_PUNCT = frozenset("।॥॰ऽ")   # avagraha trails the syllable; the mark goes before it

def _is_core(c):
    "True for letters/matras of the Brahmic blocks we handle — i.e. the syllable proper."
    o = ord(c)
    return c not in _DEVA_PUNCT and (
        0x0900 <= o <= 0x097F or 0x0C80 <= o <= 0x0CFF or
        0x1CD0 <= o <= 0x1CFF or 0xA8E0 <= o <= 0xA8FF)

def _attach(ak, mark):
    """Insert a mark after the syllable core.

    An akshara sweeps up whatever trails it — spaces, hyphens, daṇḍas — and the mark belongs
    before all of that, otherwise 'ति॒-' round-trips as 'ति-॒'.
    """
    if not mark: return ak
    i = len(ak)
    while i > 0 and not _is_core(ak[i-1]): i -= 1
    return ak[:i] + mark + ak[i:]

def has_svara(txt):
    "True if the text carries any Vedic accent mark."
    return any(c in MARKS for c in txt)

def count_svara(txt):
    "Per-mark counts — useful for corpus profiling and for spotting OOV dīrgha svarita."
    return {m: txt.count(m) for m in MARKS if m in txt}

def strip_svara(txt):
    "Remove all accent marks. This is the ablation arm: same text, no melody."
    return "".join(c for c in txt if c not in MARKS)

def svara_spans(txt, tradition="taittiriya"):
    """Per-akshara accent labels: list of (akshara_without_marks, SvaraT).

    This is the aligned label channel — one entry per syllable, unmarked syllables reported
    as UDATTA rather than dropped, so the sequence lines up with the audio syllable-for-syllable.
    """
    tbl = _table(tradition)
    out = []
    for ak in aksharas(txt):
        mark = next((c for c in ak if c in MARKS), None)
        out.append((strip_svara(ak), tbl[mark] if mark else SvaraT.UDATTA))
    return out

def extract_svara(txt, tradition="taittiriya"):
    """Split text into (bare_text, labels).

    `labels` is one SvaraT per akshara of `bare_text`, so `restore_svara` round-trips it.
    """
    spans = svara_spans(txt, tradition)
    return "".join(a for a, _ in spans), [s for _, s in spans]

def restore_svara(bare, labels, tradition="taittiriya"):
    """Inverse of `extract_svara` — reattach accent marks to a stripped string.

    Round-trips exactly under the default Taittirīya reading. Under `tradition="unicode"` an
    explicitly marked udātta is indistinguishable from an unmarked syllable, so it is not
    recovered (see `_CANON`).
    """
    canon = _CANON[tradition]
    aks = aksharas(bare)
    if len(aks) != len(labels):
        raise ValueError(f"akshara/label length mismatch: {len(aks)} vs {len(labels)}")
    return "".join(_attach(ak, canon.get(lab)) for ak, lab in zip(aks, labels))

def normalize(txt, oov_dirgha=True):
    """Make accent marks safe for the IndicF5 vocab.

    `oov_dirgha=True` folds U+1CDA (absent from the vocab) down to U+0951, keeping the tonal
    category and losing only the length distinction — the least-lossy in-vocab mapping. Also
    folds the rare U+0954 into U+0951.
    """
    if oov_dirgha: txt = txt.replace(DIRGHA_SVARITA, SVARITA)
    return txt.replace(GRAVE, SVARITA)

def to_tags(labels, skip_udatta=True):
    """Render labels as explicit conditioning tags, e.g. ['<A>', '<S>'].

    For the arm where accent is fed as side-channel tokens rather than inline diacritics.
    `skip_udatta=True` emits nothing for unmarked syllables, keeping the tag stream sparse.
    """
    return [f"<{l.value}>" for l in labels if not (skip_udatta and l is SvaraT.UDATTA)]

def contour(labels):
    """Numeric pitch-target contour: anudātta low, udātta mid, svarita falling-from-high.

    A coarse 3-level proxy for scoring synthesized f0 against the text's prescribed accent.
    Returns one float per syllable in [0,1].
    """
    lut = {SvaraT.ANUDATTA: 0.0, SvaraT.UDATTA: 0.5, SvaraT.SVARITA: 1.0, SvaraT.DIRGHA_SVARITA: 1.0}
    return [lut[l] for l in labels]
