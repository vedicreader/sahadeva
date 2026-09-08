"Devanagari → Sanskrit phone tokens. Deterministic, no dependencies; Vedic svara marks are kept as tokens."
import re, unicodedata

__all__ = ['g2p', 'VOCAB', 'tok2id', 'id2tok', 'WT', 'PAD', 'SIL', 'SP', 'UNK', 'n_vocab']

PAD, SIL, SP, UNK = '<pad>', '<sil>', '<sp>', '<unk>'

VOW = dict(zip('अआइईउऊऋॠऌॡएऐओऔ', ['a', 'aa', 'i', 'ii', 'u', 'uu', 'ri', 'rii', 'li', 'lii', 'e', 'ai', 'o', 'au']))
MAT = dict(zip('ािीुूृॄॢॣेैोौ', ['aa', 'i', 'ii', 'u', 'uu', 'ri', 'rii', 'li', 'lii', 'e', 'ai', 'o', 'au']))
CON = dict(zip('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ',
               ['k', 'kh', 'g', 'gh', 'ng', 'c', 'ch', 'j', 'jh', 'ny', 'tt', 'tth', 'dd', 'ddh', 'nn',
                't', 'th', 'd', 'dh', 'n', 'p', 'ph', 'b', 'bh', 'm', 'y', 'r', 'l', 'v', 'sh', 'ss', 's', 'h', 'll']))
SGN = {'ं': 'M', 'ँ': 'M', 'ः': 'H'}
ACC = {'॑': 'ud', '॒': 'an', '॓': 'ud', '॔': 'ud', '᳚': 'ud', '᳒': 'ud'}
VIR, NUKTA = '्', '़'
OM = 'ॐ'
DROP = set('ऽ।॥॰|.,;:!?"\'()[]{}-–—0123456789०१२३४५६७८९‌‍')

VOWELS = set(VOW.values()) | set(MAT.values())
LONG = {'aa', 'ii', 'uu', 'rii', 'lii', 'e', 'ai', 'o', 'au'}
PHONES = sorted(set(VOW.values()) | set(MAT.values()) | set(CON.values()) | set(SGN.values()) | set(ACC.values()))
VOCAB = [PAD, SIL, SP, UNK] + PHONES
tok2id = {t: i for i, t in enumerate(VOCAB)}
id2tok = {i: t for t, i in tok2id.items()}
def n_vocab(): return len(VOCAB)

# relative duration weights used to split a word's measured span across its phones
WT = {t: (2.0 if t in LONG else 1.25) if t in VOWELS else 1.0 for t in PHONES}
WT.update({v: 0.7 for v in SGN.values()}); WT.update({v: 0.25 for v in ACC.values()})
WT.update({SIL: 1.0, SP: 1.0, PAD: 0.0, UNK: 1.0})

_NUK = str.maketrans(dict(zip('\u0958\u0959\u095a\u095b\u095c\u095d\u095e\u095f', '\u0915\u0916\u0917\u091c\u0921\u0922\u092b\u092f')))

def g2p(word):
    "Phone tokens for one Devanagari word (inherent 'a' inserted unless a virama or matra follows)."
    w = unicodedata.normalize('NFC', word or '').translate(_NUK).replace(NUKTA, '')
    out, i, n = [], 0, len(w)
    while i < n:
        ch = w[i]
        if ch in CON:
            out.append(CON[ch]); i += 1
            if i < n and w[i] == VIR: i += 1
            elif i < n and w[i] in MAT: out.append(MAT[w[i]]); i += 1
            else: out.append('a')
        elif ch == OM: out += ['o', 'M']; i += 1
        elif ch in VOW: out.append(VOW[ch]); i += 1
        elif ch in MAT: out.append(MAT[ch]); i += 1
        elif ch in SGN: out.append(SGN[ch]); i += 1
        elif ch in ACC: out.append(ACC[ch]); i += 1
        else: i += 1
    return out
