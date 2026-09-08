"Test-set evaluation: original vs generated audio, plus a self-contained comparison page."
import base64, io, json, sys, time
import numpy as np, torch
from pathlib import Path
from . import cfg, data, synth

__all__ = ['originals', 'evaluate', 'page']

def originals(items):
    "Cut each test clip out of its source recording (decoding every file once)."
    out, cache = {}, {}
    for it in items:
        a = it['audio']
        if a not in cache: cache[a] = data.decode(a)
        x = cache[a]
        out[(it['key'], it['ci'])] = x[int(it['s'] * cfg.SR / 1000):int(it['e'] * cfg.SR / 1000)]
    return out

def _mcd(a, b):
    "Mean absolute log-mel distance over the overlapping frames."
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).mean()) if n else float('nan')

def evaluate(ck=None, out=None, limit=None, log=print):
    "Write original / copy-synth / generated wavs per test clip and return per-clip metrics."
    m, mu, sd, spk = synth.load(ck)
    _, _, te, _ = data.load_split()
    if limit: te = te[:limit]
    d = Path(out or cfg.OUT_DIR); d.mkdir(parents=True, exist_ok=True)
    orig, rows = originals(te), []
    for n, it in enumerate(te):
        ref = it['mel'].astype(np.float32)
        gt, _ = synth.mel_from_tokens(m, mu, sd, it['tok'], it['spk'], dur=it['dur'])       # aligner durations
        pr, dp = synth.mel_from_tokens(m, mu, sd, it['tok'], it['spk'])                     # predicted durations
        nm = f"{it['speaker']}_{it['key']}_{it['ci']:03}"
        w = dict(orig=synth.save_wav(d / f'{nm}.orig.wav', orig[(it['key'], it['ci'])]),
                 copy=synth.save_wav(d / f'{nm}.copysynth.wav', synth.vocode(ref)),
                 gen=synth.save_wav(d / f'{nm}.gen.wav', synth.vocode(pr)),
                 gen_gt=synth.save_wav(d / f'{nm}.gen_gtdur.wav', synth.vocode(gt)))
        rows.append(dict(name=nm, speaker=it['speaker'], text=it['text'], ms=it['e'] - it['s'],
                         mel_l1=_mcd(ref, gt), dur_rmse=float(np.sqrt(((dp - it['dur']) ** 2).mean())),
                         len_ratio=float(dp.sum() / max(it['dur'].sum(), 1)),
                         wav={k: v.name for k, v in w.items()}))
        log(f'  [{n+1}/{len(te)}] {nm} mel_l1={rows[-1]["mel_l1"]:.3f} dur_rmse={rows[-1]["dur_rmse"]:.2f}', flush=True)
    (d / 'metrics.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return rows

def _b64(p):
    return 'data:audio/wav;base64,' + base64.b64encode(Path(p).read_bytes()).decode()

def page(rows=None, out=None, d=None):
    "One self-contained HTML file with every clip's audio inlined — openable offline."
    d = Path(d or cfg.OUT_DIR)
    rows = rows or json.loads((d / 'metrics.json').read_text())
    css = ("body{font:14px/1.6 system-ui;margin:2rem;max-width:1100px}table{border-collapse:collapse;width:100%}"
           "td,th{border-bottom:1px solid #e5e5e5;padding:8px 6px;vertical-align:top}audio{width:210px;height:32px}"
           "th{text-align:left;background:#fafafa}.dv{font-family:'Noto Serif Devanagari',serif;font-size:15px}"
           ".s{color:#666;font-size:12px}h1{margin-bottom:0}")
    ms = lambda k: np.mean([r[k] for r in rows]) if rows else 0
    h = [f'<!doctype html><meta charset="utf-8"><title>sahadeva — test set</title><style>{css}</style>',
         '<h1>sahadeva — original vs generated</h1>',
         f'<p class="s">{len(rows)} held-out clips · mean mel L1 {ms("mel_l1"):.3f} · mean duration RMSE {ms("dur_rmse"):.2f} frames'
         f' · mean length ratio {ms("len_ratio"):.2f}</p>',
         '<p class="s"><b>original</b> = source recording · <b>copy-synth</b> = original mel through the vocoder '
         '(the ceiling this model can reach) · <b>generated</b> = model from text alone · '
         '<b>generated (aligner durations)</b> = model with the forced-aligner timings, isolating spectral quality from rhythm.</p>',
         '<table><tr><th>clip</th><th>original</th><th>copy-synth</th><th>generated</th><th>generated (aligner durations)</th></tr>']
    for r in rows:
        au = lambda k: f'<audio controls preload="none" src="{_b64(d / r["wav"][k])}"></audio>'
        h.append(f'<tr><td><div class="dv">{r["text"][:70]}</div><div class="s">{r["speaker"]} · {r["ms"]/1000:.1f}s · '
                 f'mel L1 {r["mel_l1"]:.3f} · dur RMSE {r["dur_rmse"]:.2f}</div></td>'
                 f'<td>{au("orig")}</td><td>{au("copy")}</td><td>{au("gen")}</td><td>{au("gen_gt")}</td></tr>')
    h.append('</table>')
    f = Path(out or (d / 'sahadeva_test_set.html')); f.write_text('\n'.join(h)); return f

def main():
    "sd-eval [--ck=path] [--limit=n] — synthesise the test set and write the comparison page."
    kw = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:] if '=' in a)
    r = evaluate(ck=kw.get('ck'), limit=int(kw['limit']) if 'limit' in kw else None)
    print('page →', page(r))

if __name__ == '__main__': main()
