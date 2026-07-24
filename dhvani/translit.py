"""Script detection and transliteration: any Brahmic -> Devanagari -> SLP1 -> Kannada.

Adapted from prathoshap/vagdhenu `src/prep_text.py` (Apache-2.0). The Deva->SLP1->Kannada
route is deliberate: feeding Devanagari straight to an Indic model triggers Hindi
schwa-deletion. Kannada is the script IndicF5 renders Sanskrit conjuncts correctly in.

Everything here is svara-transparent — accent marks pass through untouched. Deciding what
to *do* with them is `svara.py`'s job, not this module's.
"""
import re
from indic_transliteration import sanscript

VIRAMA, VISARGA, ANUSVARA = "्", "ं", "ं"
VISARGA = "ः"
JIHVA, UPADH = "ᳵ", "ᳶ"          # jihvāmūlīya, upadhmānīya

# Vedic accent codepoints live here, at the bottom of the import graph, because both the
# syllabifier and the svara module need them and svara.py already depends on meter.py.
VEDIC_MARKS = frozenset("॒॑᳚॔")   # anudātta, svarita, dīrgha svarita, grave

# daṇḍas, pipes, quotes, parens, ZWJ/ZWNJ. Avagraha (ऽ) and ॐ are kept — they are pronounced.
# NB: Vedic accent marks are deliberately absent from this set.
PUNCT_DROP = set("।॥|/\\—–\"'“”‘’„«»‹›*•·().,;!?‌‍")
SKIP = set(" \t\n-") | PUNCT_DROP | set("0123456789०१२३४५६७८९")

_SCRIPT_BLOCKS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI), (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),   (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),      (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),     (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),  (0x11300, 0x1137F, sanscript.GRANTHA)]

def detect_script(txt):
    "Script of the first in-block char. Roman input is NOT auto-detected — pass it pre-transliterated."
    for c in txt:
        o = ord(c)
        for lo, hi, scheme in _SCRIPT_BLOCKS:
            if lo <= o <= hi: return scheme
    return sanscript.DEVANAGARI

def to_deva(txt):
    "Normalize any supported Brahmic script to Devanagari; the rest of the pipeline works in Deva."
    src = detect_script(txt)
    return txt if src == sanscript.DEVANAGARI else sanscript.transliterate(txt, src, sanscript.DEVANAGARI)

def fix_colon(deva):
    "Stray Latin colon used as visarga: 'गुरु:-' / 'गुरु:' -> 'गुरुः'."
    return deva.replace(":-", VISARGA).replace(":", VISARGA)

def strip_punct(deva):
    "Colon->visarga, drop daṇḍas/pipes/quotes/digits, hyphen joins (a space would break alignment)."
    out = []
    for c in fix_colon(deva):
        if c in PUNCT_DROP or c.isdigit() or ("०" <= c <= "९") or c in "-–—": continue
        out.append(c)
    return re.sub(r"\s+", " ", "".join(out)).strip()

def to_slp1(deva, fix_vocalic_rr=True, fix_om=True):
    """Devanagari -> SLP1, with two pronunciation patches applied at this layer.

    `fix_vocalic_rr` rewrites long vocalic ṝ (F) as repha+ū. IndicF5 mispronounces Kannada ೄ
    (U+0CC4), so vagdhenu patches it here, where tF -> trU -> ತ್ರೂ.

    `fix_om` rewrites ॐ from AUM to OM. sanscript emits the etymological analysis a+u+m, which
    scans as two syllables and puts the metrical grid out of step with the akshara grid; ॐ is
    pronounced /oːm/, one syllable. Both render as ಓಂ, so this changes the internal
    representation and the scansion, not what the model reads.

    Turn both off when you need faithful, reversible SLP1.
    """
    slp = sanscript.transliterate(deva, sanscript.DEVANAGARI, sanscript.SLP1)
    if fix_om: slp = slp.replace("AUM", "OM")
    return slp.replace("F", "rU") if fix_vocalic_rr else slp

def to_kannada(slp):
    "SLP1 -> Kannada (the script the model actually reads)."
    return sanscript.transliterate(slp, sanscript.SLP1, sanscript.KANNADA)

def n_aksharas(s):
    "Count syllables (aksharas) in Devanagari or Kannada text."
    n, L = 0, len(s)
    for i, c in enumerate(s):
        o = ord(c)
        indep = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94)
        cons = (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        if indep: n += 1
        elif cons and (s[i+1] if i+1 < L else "") not in ("्", "್"): n += 1
    return n
