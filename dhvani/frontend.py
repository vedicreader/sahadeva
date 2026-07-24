"""Top-level frontend: raw text in, model-ready text plus aligned metadata out.

    >>> p = prep("अ॒ग्निमी॑ळे पु॒रोहि॑तं")
    >>> p.model_text        # Kannada, what IndicF5 reads
    >>> p.svara             # one SvaraT per akshara — the conditioning/eval channel
    >>> p.weight_str        # 'GGLGLLG…' laghu/guru

The two arms worth A/B-ing on this corpus are `keep_svara=True` (accent marks inline, so a
fine-tune can learn them) versus `keep_svara=False` (stripped — the honest baseline that shows
what the marks are actually buying you).
"""
from dataclasses import dataclass, field
from .translit import to_deva, strip_punct, to_slp1, to_kannada
from .svara import SvaraT, extract_svara, strip_svara, has_svara, normalize as norm_svara
from .sandhi import SandhiT, apply_sandhi, visarga_sandhi, visarga_echo_final
from .meter import weights, weight_str, aksharas

@dataclass
class Prepped:
    """Everything downstream stages need from one utterance of text.

    Two per-syllable channels on two different grids, deliberately:
      `svara`      is index-aligned to the aksharas of `bare_deva` (orthographic).
      `weight_str` is on the phonemic grid, which is what scansion actually operates on.
    They usually coincide but need not — 'तर्पणम् मन्त्रः' is 6 aksharas and scans as 5
    syllables. Only `svara` is guaranteed aligned, and that is what conditioning uses.
    """
    src: str
    deva: str                                   # normalized Devanagari, accent marks intact
    bare_deva: str                              # accent-stripped
    slp: str
    model_text: str                             # Kannada — the string handed to the model
    svara: list = field(default_factory=list)   # one SvaraT per akshara of bare_deva
    weight_str: str = ""
    n_aksharas: int = 0
    accented: bool = False

    @property
    def svara_str(self):
        "Compact accent string aligned to aksharas, e.g. 'AUSUU' — handy for manifests and diffs."
        return "".join(s.value for s in self.svara)

@dataclass
class Frontend:
    """Configured text frontend.

    sandhi:      PLAIN is the champion path; the others are opt-in arms (see sandhi.py).
    keep_svara:  keep accent marks inline in `model_text`.
    echo_final:  chant echo-vowel on the clip-final visarga (ḥ -> ha/hi/hu).
    fix_vocalic_rr: rewrite ṝ as repha+ū — IndicF5 mispronounces Kannada ೄ.
    """
    sandhi: SandhiT = SandhiT.PLAIN
    keep_svara: bool = True
    echo_final: bool = True
    fix_vocalic_rr: bool = True
    tradition: str = "taittiriya"
    allow_sandhi_on_accented: bool = False

    def __call__(self, text):
        deva = strip_punct(to_deva(text))
        deva = norm_svara(deva)                       # fold OOV dīrgha svarita into U+0951
        bare, labels = extract_svara(deva, self.tradition)

        # Guards live in apply_sandhi: it refuses non-PLAIN policies on accented text.
        deva_s = apply_sandhi(deva, self.sandhi, self.allow_sandhi_on_accented)
        bare_s = strip_svara(deva_s)

        src_for_model = deva_s if self.keep_svara else bare_s
        slp = to_slp1(src_for_model, self.fix_vocalic_rr)
        if self.sandhi is SandhiT.UTVA_RUTVA_LOPA:
            slp = visarga_sandhi(slp)
        if self.echo_final:
            slp = visarga_echo_final(slp)

        bare_slp = to_slp1(bare_s, self.fix_vocalic_rr)
        return Prepped(src=text, deva=deva_s, bare_deva=bare_s, slp=slp,
                       model_text=to_kannada(slp), svara=labels,
                       weight_str=weight_str(bare_slp), n_aksharas=len(aksharas(bare_s)),
                       accented=has_svara(deva))

_default = Frontend()

def prep(text, **kw):
    "One-shot convenience. Pass Frontend kwargs to override the defaults."
    return (Frontend(**kw) if kw else _default)(text)
