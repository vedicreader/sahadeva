"Turn word-level alignments into training clips: phone tokens, per-token frame durations and log-mel targets."
import json, subprocess, sys
from functools import lru_cache
import numpy as np
from pathlib import Path
from sdata.qc import measure_array, _rate_outliers, MIN_DUR, MAX_DUR, MAX_SILENCE_FRAC, MIN_SNR_DB, MAX_CLIP_FRAC
from . import cfg, corpus
from .text import phones, tok2id, WT, SIL, SP, UNK, n_aksharas

__all__ = ['decode', 'clip_lines', 'clip_tokens', 'build', 'load_split', 'speakers']

# === audio ===
def decode(path, sr=cfg.SR):
    "Decode any audio file to mono float32 at sr via ffmpeg."
    r = subprocess.run(['ffmpeg', '-v', 'quiet', '-i', str(path), '-f', 's16le', '-ac', '1', '-ar', str(sr), '-'],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0

# === clipping: merge short lines, split over-long ones at their widest word gap ===
def _split_words(ws, mx):
    "Recursively split a word list at its widest internal gap until every piece fits mx ms."
    if not ws or ws[-1]['e'] - ws[0]['s'] <= mx or len(ws) < 2: return [ws] if ws else []
    k = max(range(1, len(ws)), key=lambda i: ws[i]['s'] - ws[i - 1]['e'])
    return _split_words(ws[:k], mx) + _split_words(ws[k:], mx)

def clip_lines(rec, mn=cfg.CLIP_MIN_MS, mx=cfg.CLIP_MAX_MS, join_gap=1200):
    """Word groups that make good training clips: each within [mn, mx] ms where the audio allows.

    Words that transliterate to nothing (a bare danda that aeneas timed as its own fragment) are
    dropped, so their span widens the neighbouring gap and becomes an explicit pause instead.
    """
    groups, cur = [], []
    for l in rec['lines']:
        ws = [w for w in l.get('words') or [] if w['e'] > w['s'] and _word_toks(w['t'])]
        if not ws: continue
        for piece in _split_words(ws, mx):
            if cur and (piece[0]['s'] - cur[-1]['e'] > join_gap or piece[-1]['e'] - cur[0]['s'] > mx):
                groups.append(cur); cur = []
            cur += piece
            if cur[-1]['e'] - cur[0]['s'] >= mn: groups.append(cur); cur = []
    if cur: groups.append(cur)
    return [g for g in groups if g and mn <= g[-1]['e'] - g[0]['s'] <= mx]

# === tokens + durations ===
@lru_cache(maxsize=1 << 16)
def _word_toks(t): return tuple(phones(t))

def clip_tokens(ws, pad=cfg.PAD_MS, gap_sil=cfg.GAP_SIL_MS):
    "Tokens and per-token millisecond durations for one clip, plus its audio span."
    s0, e1 = max(ws[0]['s'] - pad, 0), ws[-1]['e'] + pad
    toks, dur = [SIL], [ws[0]['s'] - s0]
    for i, w in enumerate(ws):
        ph = list(_word_toks(w['t']))
        wt = np.array([WT.get(x, 1.0) for x in ph], float); wt = wt if wt.sum() else np.ones(len(ph))
        span = max(w['e'] - w['s'], len(ph))
        toks += ph; dur += list(wt / wt.sum() * span)
        gap = (ws[i + 1]['s'] if i + 1 < len(ws) else e1) - w['e']
        if gap > 10: toks.append(SIL if gap >= gap_sil else SP); dur.append(gap)
    return toks, np.array(dur, float), s0, e1

def _to_frames(dur_ms, n_frames):
    "Quantise millisecond durations to frames summing exactly to n_frames, every token at least 1."
    d = np.maximum(np.round(dur_ms / 1000 * cfg.FPS), 1).astype(np.int64)
    while d.sum() != n_frames:
        gap = n_frames - d.sum()
        if gap > 0: d[np.argsort(-dur_ms)[:gap]] += 1
        else:
            cand = np.flatnonzero(d > 1)
            if not len(cand): return None
            d[cand[np.argsort(-d[cand])[:min(-gap, len(cand))]]] -= 1
    return d

# === build ===
def _mel(fe, x):
    o = fe(audio_target=x, sampling_rate=cfg.SR, return_tensors='np')
    return np.asarray(list(o.values())[0][0], np.float32)

def build(align_dir=None, out=None, log=print):
    "Write one npz of packed (tokens, durations, mel) per source recording, plus a dataset index."
    from transformers import SpeechT5FeatureExtractor
    fe = SpeechT5FeatureExtractor()
    out = Path(out or cfg.FEAT_DIR); out.mkdir(parents=True, exist_ok=True)
    index, spk = [], {}
    for rec in corpus.load_all(align_dir):
        k = corpus.key(rec)
        if not Path(rec['audio']).is_file(): log(f'  skip {k}: audio missing'); continue
        x = decode(rec['audio'])
        mels, toks, durs, mo, to, meta = [], [], [], [0], [0], []
        for ws in clip_lines(rec):
            t, dms, s0, e1 = clip_tokens(ws)
            a, b = int(s0 * cfg.SR / 1000), int(e1 * cfg.SR / 1000)
            if b - a < cfg.SR // 2 or b > len(x): continue
            m = _mel(fe, x[a:b])
            d = _to_frames(dms, len(m))
            if d is None or len(d) != len(t): continue
            mels.append(m.astype(np.float16)); mo.append(mo[-1] + len(m))
            toks += [tok2id.get(z, tok2id[UNK]) for z in t]; durs += list(d); to.append(len(toks))
            txt = ' '.join(w['t'] for w in ws)
            meta.append(dict(s=int(s0), e=int(e1), text=txt, nf=len(m), nt=len(t),
                             qc=_qc(x[a:b], cfg.SR), n_aksharas=n_aksharas(txt)))
        if not mels: log(f'  skip {k}: no usable clips'); continue
        spk.setdefault(rec['speaker'], len(spk))
        np.savez(out / f'{k}.npz', mel=np.concatenate(mels), mel_off=np.array(mo), tok=np.array(toks, np.int16),
                 tok_off=np.array(to), dur=np.array(durs, np.int16))
        index.append(dict(key=k, speaker=rec['speaker'], corpus=rec['corpus'], title=rec['title'],
                          audio=rec['audio'], clips=meta))
        log(f'  {k}: {len(mels)} clips, {mo[-1]/cfg.FPS/60:.1f} min, {len(toks)} tokens')
    qc_gate(index, log)
    (out / 'index.json').write_text(json.dumps(dict(speakers=spk, records=index), ensure_ascii=False))
    n = sum(len(r['clips']) for r in index)
    ok = sum(1 for r in index for c in r['clips'] if c['qc_pass'])
    log(f'{len(index)} records, {n} clips ({ok} pass QC), {len(spk)} speakers → {out}')
    return index

# === QC (sdata.qc thresholds, applied to in-memory clips) ===
def _qc(x, sr): return measure_array(np.asarray(x, np.float32), sr)

def qc_gate(index, log=print):
    """Flag clips rather than dropping them, so a threshold change never means re-cutting audio.

    The speaking-rate outlier test is the useful one: an aksharas-per-second far from the
    reciter's own median is the signature of a text/audio mismatch that survives good timings.
    """
    utts = [dict(id=f"{r['key']}#{i}", speaker_id=r['speaker'], n_aksharas=c['n_aksharas'],
                 dur_s=c['qc'].get('dur_s') or (c['e'] - c['s']) / 1000, qc=c['qc'])
            for r in index for i, c in enumerate(r['clips'])]
    z = _rate_outliers(utts)
    counts = {}
    for u, (r, c) in zip(utts, [(r, c) for r in index for c in r['clips']]):
        q = c['qc']; reasons = []
        if 'error' in q: reasons.append(q['error'])
        else:
            if not (MIN_DUR <= q['dur_s'] <= MAX_DUR): reasons.append('duration')
            if q['clip_frac'] > MAX_CLIP_FRAC: reasons.append('clipping')
            if q['silence_frac'] > MAX_SILENCE_FRAC: reasons.append('silence')
            if q['snr_db'] < MIN_SNR_DB: reasons.append('snr')
            if z.get(u['id'], 0.0): reasons.append('speaking_rate')
        c['qc']['rate_z'] = z.get(u['id'], 0.0); c['qc']['reasons'] = reasons; c['qc_pass'] = not reasons
        for x in reasons: counts[x] = counts.get(x, 0) + 1
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]): log(f'  QC fail {k:14}{v:>6}')
    return index

def speakers(idx=None): return (idx or json.loads((Path(cfg.FEAT_DIR) / 'index.json').read_text()))['speakers']

def load_split(n_test=8, n_val=4, seed=cfg.SEED, feat_dir=None, speakers=None):
    """Load every clip into memory and split per speaker.

    Test clips are held out whole, so original-vs-generated is a fair comparison on unseen audio.
    """
    d = Path(feat_dir or cfg.FEAT_DIR)
    idx = json.loads((d / 'index.json').read_text())
    rng, items = np.random.default_rng(seed), []
    for r in idx['records']:
        if speakers and r['speaker'] not in speakers: continue
        z = np.load(d / f'{r["key"]}.npz')
        mel, mo, tk, to, du = z['mel'], z['mel_off'], z['tok'], z['tok_off'], z['dur']
        for i, c in enumerate(r['clips']):
            if not c.get('qc_pass', True): continue
            items.append(dict(key=r['key'], ci=i, spk=sorted(speakers).index(r['speaker']) if speakers else idx['speakers'][r['speaker']], speaker=r['speaker'],
                              audio=r['audio'], text=c['text'], s=c['s'], e=c['e'],
                              mel=mel[mo[i]:mo[i + 1]], tok=tk[to[i]:to[i + 1]].astype(np.int64), dur=du[to[i]:to[i + 1]].astype(np.int64)))
    tr, va, te = [], [], []
    for s in sorted({it['speaker'] for it in items}):
        g = [it for it in items if it['speaker'] == s]
        # prefer mid-length clips for the test set: long enough to judge, short enough to vocode fast
        order = rng.permutation(len(g))
        pick = [i for i in order if 2500 <= g[i]['e'] - g[i]['s'] <= 7000][:n_test + n_val]
        te += [g[i] for i in pick[:n_test]]; va += [g[i] for i in pick[n_test:]]
        held = set(pick[:n_test + n_val]); tr += [g[i] for i in range(len(g)) if i not in held]
    return tr, va, te, idx['speakers']

if __name__ == '__main__': build()
