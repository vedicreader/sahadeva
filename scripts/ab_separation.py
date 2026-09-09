"""A/B: does vocal separation improve the acoustic model?

Same clips, same architecture, same schedule — only the source audio differs. Mel L1 is not
comparable across arms because separation changes the targets, so the headline number is
gen / copy-synth ceiling: the fraction of achievable quality each arm reaches.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sd import cfg, data, train, synth
from sd.model import Sahadeva
from sd.text import n_vocab

ARMS = dict(base='data/align_ab_base', sep='data/align_ab_sep')

def ceiling(items, fe):
    "Copy-synth mel L1: reference mel -> vocoder -> re-analysed mel. The best any model could do."
    out = []
    for it in items:
        ref = it['mel'].astype(np.float32)
        w = synth.vocode(ref)
        re_ = np.asarray(list(fe(audio_target=w, sampling_rate=cfg.SR, return_tensors='np').values())[0][0], np.float32)
        n = min(len(ref), len(re_)); out.append(float(np.abs(ref[:n] - re_[:n]).mean()))
    return float(np.mean(out))

def arm(name, align_dir, epochs, n_test, log=print):
    feats = Path(f'data/feats_{name}')
    if not (feats / 'index.json').is_file():
        log(f'[{name}] building features from {align_dir}')
        data.build(align_dir=align_dir, out=feats, log=lambda *a: None)
    tr, va, te, spk = data.load_split(feat_dir=feats)
    te = te[:n_test]
    mu, sd = train.norm_stats(tr)
    log(f'[{name}] train={len(tr)} val={len(va)} test={len(te)} spk={len(spk)} '
        f'hours={sum(len(i["mel"]) for i in tr)/cfg.FPS/3600:.2f}')
    torch.manual_seed(cfg.SEED)
    m = Sahadeva(n_vocab(), len(spk))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.LR, weight_decay=1e-6)
    nb = len(train.batches(tr, shuffle=False)); t0 = time.time()
    for ep in range(epochs):
        m.train()
        for k, b in enumerate(train.batches(tr, seed=ep)):
            for g in opt.param_groups: g['lr'] = train.one_cycle(ep * nb + k, epochs * nb)
            loss, _, _ = train.loss_fn(m, train.collate(b, mu, sd))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    vb = train.batches(va, shuffle=False)
    with torch.no_grad(): val = float(np.mean([train.loss_fn(m, train.collate(b, mu, sd))[1] for b in vb])) if vb else float('nan')
    gen = []
    for it in te:
        mel, _ = synth.mel_from_tokens(m, mu, sd, it['tok'], it['spk'], dur=it['dur'])
        ref = it['mel'].astype(np.float32); n = min(len(ref), len(mel))
        gen.append(float(np.abs(ref[:n] - mel[:n]).mean()))
    from transformers import SpeechT5FeatureExtractor
    cl = ceiling(te, SpeechT5FeatureExtractor())
    g = float(np.mean(gen))
    log(f'[{name}] {epochs} epochs in {time.time()-t0:.0f}s · val_l1={val:.4f} · gen_mel_l1={g:.4f} '
        f'· ceiling={cl:.4f} · gen/ceiling={g/cl:.2f}')
    return dict(arm=name, val=val, gen=g, ceiling=cl, ratio=g / cl, clips=len(tr), per_clip=gen)

def main(epochs=10, n_test=12):
    res = [arm(k, v, epochs, n_test) for k, v in ARMS.items()]
    print(f'\n{"arm":6} {"clips":>6} {"val_l1":>8} {"gen_l1":>8} {"ceiling":>8} {"gen/ceiling":>12}')
    for r in res: print(f'{r["arm"]:6} {r["clips"]:6} {r["val"]:8.4f} {r["gen"]:8.4f} {r["ceiling"]:8.4f} {r["ratio"]:12.2f}')
    b, s = res[0], res[1]
    print(f'\nseparation changes gen/ceiling by {100*(s["ratio"]-b["ratio"])/b["ratio"]:+.1f}%  (lower ratio is better)')
    d = np.array(s['per_clip']) - np.array(b['per_clip'])
    if len(d) > 1:
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f'paired per-clip diff (sep - base): mean={d.mean():+.4f} sd={d.std(ddof=1):.4f} '
              f'n={len(d)} t={t:+.2f} · sep better on {int((d<0).sum())}/{len(d)} clips')
    Path('data/out').mkdir(parents=True, exist_ok=True)
    Path('data/out/ab_separation.json').write_text(json.dumps(res, indent=1))
    return res

if __name__ == '__main__':
    kw = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:] if '=' in a)
    main(**{k: int(v) for k, v in kw.items()})
