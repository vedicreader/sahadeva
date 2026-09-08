"""Syllabification and metrical weight (laghu/guru).

Two different splits, for two different jobs:
  - `aksharas()`  orthographic, on Devanagari/Kannada. Used to attach svara marks to syllables.
  - `syllabify()` phonemic, on SLP1. Used for metrical weight, where 1 SLP1 char == 1 phone.
"""
from enum import Enum
from .translit import n_aksharas, VEDIC_MARKS  # noqa: F401  (n_aksharas re-exported via __init__)

VOWELS = set("aAiIuUfFxXeEoO")
LONG = set("AIUFXeEoO")        # Sanskrit has no short e/o — SLP1 e,o are inherently long
SHORT = set("aiufx")
CODA = set("MH")               # anusvāra, visarga

class WeightT(str, Enum):
    LAGHU = "L"
    GURU = "G"

_VIRAMAS = ("्", "್")

def aksharas(s):
    """Split Devanagari/Kannada text into orthographic syllables.

    A new akshara starts at an independent vowel or a consonant that is not preceded by a
    virāma. Matras, anusvāra, visarga, nuktas and Vedic accent marks all cling to the
    syllable they follow — which is exactly what makes this the right unit for svara.

    The virāma lookback deliberately skips accent marks. In 'ग्॑श्च' the mark sits between the
    virāma and the next consonant; without the skip, accented and accent-stripped copies of the
    same text segment into different numbers of aksharas and the svara labels stop lining up.
    """
    out, cur = [], ""
    for i, c in enumerate(s):
        o = ord(c)
        base = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94) or \
               (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        j = i - 1
        while j >= 0 and s[j] in VEDIC_MARKS: j -= 1
        prev = s[j] if j >= 0 else ""
        if base and prev not in _VIRAMAS:
            if cur: out.append(cur)
            cur = c
        else:
            cur += c
    if cur: out.append(cur)
    return out

def syllabify(slp):
    "Split an SLP1 string into phonemic syllables: onset consonants + vowel + optional M/H coda."
    out, cur = [], ""
    for c in slp:
        if c.isspace(): continue
        cur += c
        if c in VOWELS:
            out.append(cur); cur = ""
        elif c in CODA and out:
            out[-1] += cur; cur = ""      # coda attaches to the syllable just closed
    if cur: out.append(cur)               # trailing consonants (halant-final)
    return out

def _onset_len(syl):
    "Number of leading consonants in a syllable."
    n = 0
    for c in syl:
        if c in VOWELS: break
        if c not in CODA: n += 1
    return n

def weights(slp, final_anceps=False):
    """Laghu/guru per syllable.

    Guru when the vowel is long, or the syllable is closed by anusvāra/visarga, or the next
    syllable opens with a conjunct (>=2 consonants). `final_anceps=True` returns None for the
    last syllable, which in most vṛttas may be counted either way.
    """
    syls = syllabify(slp)
    out = []
    for i, syl in enumerate(syls):
        v = next((c for c in syl if c in VOWELS), None)
        if v is None:                                   # consonant-only tail
            out.append(WeightT.GURU); continue
        heavy = v in LONG or any(c in CODA for c in syl)
        if not heavy and i + 1 < len(syls) and _onset_len(syls[i+1]) >= 2: heavy = True
        out.append(WeightT.GURU if heavy else WeightT.LAGHU)
    if final_anceps and out: out[-1] = None
    return out

def weight_str(slp):
    "Compact L/G string, e.g. 'LGLGGLLG' — the shape you match against vṛtta templates."
    return "".join(w.value if w else "?" for w in weights(slp))
