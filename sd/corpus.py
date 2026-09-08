"Standardise vedicreader lyrics XML and audio_alignment aeneas JSON into one align schema."
import json, re
import xml.etree.ElementTree as ET
from pathlib import Path
from . import cfg

# Unified record:
#   dict(source, corpus, id, title, speaker, audio, lines=[dict(i, text, s, e, words=[dict(t, s, e)])])
# All times are integer milliseconds.

ID_RE = re.compile(r'p(\d+)s(\d+)w(\d+)')
PUNCT = set('।॥|.,;:!?—-–')

def _ms(v): return int(round(float(v) * 1000))
def _txt(fr): return ' '.join(x.strip() for x in fr.get('lines', []) if x.strip()).strip()
def is_word(t): return bool(t) and not all(c in PUNCT for c in t)

# === audio_alignment (aeneas mplain, levels=3) ===
def words_by_sentence(word_json):
    "Group level-3 word fragments into sentences keyed by (paragraph, sentence) from their aeneas ids."
    out = {}
    for fr in json.load(open(word_json, encoding='utf-8'))['fragments']:
        m, t = ID_RE.match(fr.get('id', '') or ''), _txt(fr)
        if not (m and 'end' in fr and is_word(t)): continue
        out.setdefault((int(m[1]), int(m[2])), []).append(dict(t=t, s=_ms(fr['begin']), e=_ms(fr['end'])))
    return out

def from_audio_alignment(corpus, entry, speaker, audio):
    "Build a record from an audio_alignment data.json entry (word alignment is the source of truth)."
    base = cfg.ALIGN_REF / 'data' / corpus
    grps = words_by_sentence(base / entry['word_alignment'])
    lines = []
    for k in sorted(grps):
        ws = sorted(grps[k], key=lambda w: w['s'])
        if not ws: continue
        lines.append(dict(i=len(lines), text=' '.join(w['t'] for w in ws), s=ws[0]['s'], e=ws[-1]['e'], words=ws))
    return dict(source='audio_alignment', corpus=corpus, id=entry['id'], title=entry.get('name', entry['id']),
                speaker=speaker, audio=str(audio), lines=lines)

# === vedicreader library content: lyrics XML or the JSON content format ===
def _rec(x, audio, lines):
    return dict(source='vedicreader', corpus='vedicreader', id=x.parent.name, title=x.parent.name,
                speaker=f'{cfg.VR_SPK_PREFIX}_{x.parent.name}', audio=str(audio), lines=lines)

def _xml_words(l):
    "Word timings from the aligner's `<w>` children, when the published XML carries them."
    out = [dict(t=_one(w.text), s=int(w.get('start_time_ms', 0)), e=int(w.get('end_time_ms', 0))) for w in l.iterfind('w')]
    return [w for w in out if is_word(w['t']) and w['e'] > w['s']]

def _one(t): return re.sub(r'\s+', ' ', t or '').strip()

def from_vr_xml(xml_pth, audio, words=None):
    "Build a record from a vedicreader lyrics XML. Word timings come from `<w>` children, else from `words` by alignment_id."
    x = Path(xml_pth)
    root, lines = ET.parse(x).getroot(), []
    for l in root.iter('line'):
        aid = l.get('alignment_id')
        if aid is None or l.get('ignore') == 'true': continue
        s, e = int(l.get('start_time_ms', 0)), int(l.get('end_time_ms', 0))
        t = _one(l.text)
        if not (t and e > s): continue
        lines.append(dict(i=len(lines), text=t, s=s, e=e, words=_xml_words(l) or (words or {}).get(str(aid), [])))
    return _rec(x, audio, lines)

def _json_words(l):
    ws = [dict(t=_one(w[0]), s=int(w[1]), e=int(w[2])) if isinstance(w, (list, tuple))
          else dict(t=_one(w.get('t')), s=int(w.get('s', 0)), e=int(w.get('e', 0))) for w in (l.get('w') or l.get('words') or [])]
    return [w for w in ws if is_word(w['t']) and w['e'] > w['s']]

def from_vr_json(json_pth, audio):
    "Build a record from a vedicreader JSON content file; `heading` lines are printed, not recited."
    x = Path(json_pth)
    d, lines = json.loads(x.read_text(encoding='utf-8')), []
    for sec in d.get('sections') or []:
        for l in sec.get('lines') or []:
            if str(l.get('role') or 'verse') in ('heading', 'skip'): continue
            t, s, e = _one(l.get('t') or l.get('text')), int(l.get('s', 0)), int(l.get('e', 0))
            if not (t and e > s): continue
            lines.append(dict(i=len(lines), text=t, s=s, e=e, words=_json_words(l)))
    return _rec(x, audio, lines)

def from_vr(pth, audio, words=None):
    "Read either vedicreader content format."
    return from_vr_json(pth, audio) if str(pth).endswith('.json') else from_vr_xml(pth, audio, words)

# === io ===
def key(rec): return f"{rec['corpus']}__{re.sub(r'[^A-Za-z0-9._-]+', '_', rec['id'])}"
def save(rec, d=None):
    p = Path(d or cfg.ALN_DIR) / f'{key(rec)}.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8'); return p
def load_all(d=None): return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(Path(d or cfg.ALN_DIR).glob('*.json'))]

def stats(recs):
    "Per-speaker line counts and voiced hours."
    out = {}
    for r in recs:
        a = out.setdefault(r['speaker'], dict(files=0, lines=0, secs=0.0))
        a['files'] += 1; a['lines'] += len(r['lines'])
        a['secs'] += sum(l['e'] - l['s'] for l in r['lines']) / 1000
    return out

def import_vr_sidecars(d, dst=None):
    "Copy vedicreader Align Studio sidecar records (already in this schema) into the dataset."
    out = []
    for f in sorted(Path(d).glob('*.words.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if not Path(r['audio']).is_absolute(): r['audio'] = str(cfg.VR_ROOT / r['audio'])
        r['speaker'] = f"vr_{f.name.split('.')[0]}"
        r['lines'] = [l for l in r['lines'] if l.get('words')]
        if r['lines']: save(r, dst); out.append(r)
    return out
