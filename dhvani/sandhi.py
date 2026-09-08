"""Sandhi policies — switchable, because the obvious choice is empirically the wrong one.

The intuition is that a TTS frontend should resolve every pronunciation-affecting sandhi.
Vagdhenu A/B-tested exactly that and found the opposite: their 4.6-MOS champion feeds text
with visarga and anusvāra left PLAIN and lets the model learn the realizations acoustically.
Their `prep_text.py` states it outright — "A/B 2026-06-15: plain > resolved for satva".

So PLAIN is the default here and the others are opt-in arms, not an ablation ladder where
more is better.

For accented Vedic text specifically, PLAIN is not merely the default but the correct choice:
saṃhitāpāṭha already *has* its sandhi applied in the received text. Re-resolving it corrupts
the transmitted form. `apply_sandhi` refuses to do that silently.
"""
from enum import Enum
from .svara import MARKS, has_svara

VIRAMA, VISARGA, ANUSVARA = "्", "ः", "ं"
JIHVA, UPADH = "ᳵ", "ᳶ"

class SandhiT(str, Enum):
    PLAIN = "plain"                  # champion: no sandhi at all
    UTVA_RUTVA_LOPA = "utva"         # vagdhenu production: word-boundary visarga only
    FULL = "full"                    # + satva + homorganic anusvāra (A/B'd worse — for ablation)

KA_V, CA_V, TTA_V = set("कखगघङ"), set("चछजझञ"), set("टठडढण")
TA_V, PA_V = set("तथदधन"), set("पफबभम")
STOP_NASAL = {**{c: "ङ" for c in KA_V}, **{c: "ञ" for c in CA_V}, **{c: "ण" for c in TTA_V},
              **{c: "न" for c in TA_V}, **{c: "म" for c in PA_V}}
K_UNVOICED, P_UNVOICED = set("कख"), set("पफ")
VIS_SIB = {**{c: "स" for c in "सतथ"}, **{c: "श" for c in "शचछ"}, **{c: "ष" for c in "षटठ"}}
_SKIP = set(" \t\n-") | MARKS      # accent marks never block a sandhi context lookahead

def _next_real(s, i):
    "Index of the next char that can condition sandhi, skipping spaces and accent marks."
    j = i + 1
    while j < len(s) and s[j] in _SKIP: j += 1
    return j if j < len(s) else None

def homorganic_anusvara(deva):
    "Anusvāra -> homorganic nasal + virāma before a stop; left as ं before sibilant/semivowel/h/end."
    out = []
    for i, c in enumerate(deva):
        if c == ANUSVARA:
            j = _next_real(deva, i)
            nxt = deva[j] if j is not None else None
            out.append(STOP_NASAL[nxt] + VIRAMA if nxt in STOP_NASAL else ANUSVARA)
        else: out.append(c)
    return "".join(out)

def satva(deva, kannada_safe=True):
    """Visarga -> sibilant assimilation before stops/sibilants; segment-final visarga preserved.

    `kannada_safe=True` keeps plain ः before k/p instead of emitting jihvāmūlīya ᳵ /
    upadhmānīya ᳶ, which are out-of-vocab for the Kannada-routed model.
    """
    last_final = None
    for i, c in enumerate(deva):
        if c == VISARGA and _next_real(deva, i) is None: last_final = i
    out = []
    for i, c in enumerate(deva):
        if c != VISARGA: out.append(c); continue
        if i == last_final: out.append(VISARGA); continue
        j = _next_real(deva, i)
        nxt = deva[j] if j is not None else None
        if nxt in K_UNVOICED:   out.append(VISARGA if kannada_safe else JIHVA)
        elif nxt in P_UNVOICED: out.append(VISARGA if kannada_safe else UPADH)
        elif nxt in VIS_SIB:    out.append(VIS_SIB[nxt] + VIRAMA)
        else:                   out.append(VISARGA)
    return "".join(out)

# ── word-boundary visarga sandhi, on SLP1 ────────────────────────────────────────────
_VS_VOICED = set("gGjJqQdDbBNYRnmyrlvh")
_VS_OTHERV = set("iIuUfFxXeEoO")
_VS_ALLV = set("aAiIuUfFxXeEoO")
_VS_LEN = {"a": "A", "i": "I", "u": "U", "f": "F", "A": "A", "I": "I", "U": "U"}

def visarga_sandhi(slp):
    """Word-boundary visarga: utva / rutva / lopa only.

    satva and jihvāmūlīya/upadhmānīya contexts are deliberately left plain — see module docstring.
      utva : aH + a -> o' ; aH + voiced -> o
      rutva: (i/u/e/o…)H + vowel/voiced -> r
      lopa : āH + vowel/voiced -> ā ; aH + (vowel != a) -> a ; saḥ/eṣaḥ + (!= a) -> sa/eṣa
    """
    ws = slp.split(" "); i = 0; out = []
    while i < len(ws):
        w = ws[i]
        if not (w.endswith("H") and i < len(ws) - 1 and len(w) >= 2):
            out.append(w); i += 1; continue
        V, base, nxt = w[-2], w[:-1], ws[i+1]
        F = nxt[0] if nxt else ""
        if F == "r":                                  out.append(base[:-1] + _VS_LEN.get(V, V))
        elif w in ("saH", "ezaH") and F != "a":       out.append(base)
        elif F not in _VS_ALLV and F not in _VS_VOICED: out.append(w)      # keep plain
        elif V == "a":
            if F == "a":   out.append(base[:-1] + "o"); ws[i+1] = "'" + nxt[1:]
            elif F in _VS_VOICED: out.append(base[:-1] + "o")
            else:          out.append(base)
        elif V == "A":                                out.append(base)
        elif V in _VS_OTHERV:                         out.append(base + "r")
        else:                                         out.append(w)
        i += 1
    return " ".join(out)

_VS_VOWELS = "aAiIuUfFxXeEoO"

def visarga_echo_final(slp):
    """Chant echo-vowel on the clip-final visarga: rāmaḥ -> rāmaha, guruḥ -> guruhu.

    This is a recitation convention, and it also fixes the garbled clip-final visarga the
    vocoder otherwise produces. Applies to the last word only.
    """
    ws = slp.split(" ")
    if ws and ws[-1].endswith("H") and len(ws[-1]) >= 2 and ws[-1][-2] in _VS_VOWELS:
        ws[-1] = ws[-1][:-1] + "h" + ws[-1][-2]
    return " ".join(ws)

def apply_sandhi(deva, policy=SandhiT.PLAIN, allow_on_accented=False):
    """Apply a Devanagari-level sandhi policy. Returns Devanagari.

    Raises on accented input for non-PLAIN policies unless `allow_on_accented=True`:
    saṃhitāpāṭha is already sandhi-resolved, so re-applying corrupts the received text.
    """
    if policy is SandhiT.PLAIN: return deva
    if has_svara(deva) and not allow_on_accented:
        raise ValueError(
            "refusing to apply sandhi to accented Vedic text — saṃhitāpāṭha already carries its "
            "sandhi. Use SandhiT.PLAIN, or pass allow_on_accented=True if you know better.")
    if policy is SandhiT.UTVA_RUTVA_LOPA: return deva          # handled at SLP1 layer, see frontend
    if policy is SandhiT.FULL:            return satva(homorganic_anusvara(deva))
    raise ValueError(f"unknown policy {policy!r}")
