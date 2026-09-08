"Phone tokens for the acoustic model, built on dhvani so svara semantics match the rest of the repo."
import numpy as np
from dhvani.meter import aksharas
from dhvani.svara import SvaraT, extract_svara, normalize as norm_svara
from dhvani.translit import to_deva, to_slp1, strip_punct

__all__ = ['phones', 'text_ids', 'VOCAB', 'tok2id', 'id2tok', 'WT', 'PAD', 'SIL', 'SP', 'UNK', 'n_vocab', 'n_aksharas']

PAD, SIL, SP, UNK = '<pad>', '<sil>', '<sp>', '<unk>'

# Vedic accent rides the phone grid as its own token, attached after the akshara it marks.
# Names follow the Taittiriya reading dhvani uses: U+0951 is svarita, not udatta (see dhvani/svara.py).
SVARA_TOK = {SvaraT.ANUDATTA: '<anudatta>', SvaraT.SVARITA: '<svarita>', SvaraT.DIRGHA_SVARITA: '<svarita2>'}

SHORT_V, LONG_V = set('aiufx'), set('AIUFXeEoO')
VOWELS = SHORT_V | LONG_V
CODA = set('MH')
CONS = set('kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshL') | set("'")   # avagraha is pronounced, keep it
SLP = sorted(VOWELS | CODA | CONS)

VOCAB = [PAD, SIL, SP, UNK] + sorted(SVARA_TOK.values()) + SLP
tok2id = {t: i for i, t in enumerate(VOCAB)}
id2tok = {i: t for t, i in tok2id.items()}
def n_vocab(): return len(VOCAB)

# relative duration weights used to split a word's measured span across its phones
WT = {c: (2.0 if c in LONG_V else 1.25 if c in SHORT_V else 0.7 if c in CODA else 1.0) for c in SLP}
WT.update({t: 0.25 for t in SVARA_TOK.values()})
WT.update({SIL: 1.0, SP: 1.0, UNK: 1.0, PAD: 0.0})

def phones(word):
    """SLP1 phone tokens for one word, with svara tokens attached after the akshara they mark.

    Transliteration goes through dhvani (indic_transliteration), not a hand-rolled table, and
    dirgha svarita is folded to svarita first so the tonal category survives.
    """
    d = norm_svara(strip_punct(to_deva(word or '')))
    if not d: return []
    bare, labels = extract_svara(d)
    out = []
    for ak, lab in zip(aksharas(bare), list(labels) + [SvaraT.UDATTA] * len(bare)):
        out += [c for c in to_slp1(ak) if c in tok2id]
        if lab in SVARA_TOK: out.append(SVARA_TOK[lab])
    return out

def n_aksharas(text): return len(aksharas(strip_punct(to_deva(text or ''))))

def text_ids(toks): return np.array([tok2id.get(t, tok2id[UNK]) for t in toks], np.int64)
