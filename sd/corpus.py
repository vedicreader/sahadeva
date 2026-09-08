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

# === vedicreader lyrics XML ===
def from_vr_xml(xml_pth, audio, words=None):
    "Build a record from a vedicreader lyrics XML; words: {alignment_id: [dict(t,s,e)]} from the enhanced aligner."
    x = Path(xml_pth)
    root, lines = ET.parse(x).getroot(), []
    for l in root.iter('line'):
        aid = l.get('alignment_id')
        if aid is None or l.get('ignore') == 'true': continue
        s, e = int(l.get('start_time_ms', 0)), int(l.get('end_time_ms', 0))
        t = re.sub(r'\s+', ' ', (l.text or '')).strip()
        if not (t and e > s): continue
        lines.append(dict(i=len(lines), text=t, s=s, e=e, words=(words or {}).get(str(aid), [])))
    return dict(source='vedicreader', corpus='vedicreader', id=x.parent.name, title=x.parent.name,
                speaker=f'{cfg.VR_SPK_PREFIX}_{x.parent.name}', audio=str(audio), lines=lines)

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
