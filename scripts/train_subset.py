"""Train on a speaker subset — the single-speaker arm, without rebuilding features.

Reuses data/feats; `speakers` just filters the split, so this is directly comparable to the
multi-speaker run on the same clips.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sd import cfg, data, train, synth
from sd.model import Sahadeva
from sd.text import n_vocab

def run(speakers, epochs=30, out=None, threads=4, n_test=8, log=print):
    torch.set_num_threads(threads); torch.manual_seed(cfg.SEED)
    sp = set(speakers.split(','))
    out = Path(out or (cfg.RUN_DIR.parent / f'run_{"_".join(sorted(sp))[:40]}')); out.mkdir(parents=True, exist_ok=True)
    tr, va, te, _ = data.load_split(speakers=sp)
    mu, sd = train.norm_stats(tr)
    log(f'speakers={sorted(sp)} train={len(tr)} val={len(va)} test={len(te)} '
        f'hours={sum(len(i["mel"]) for i in tr)/cfg.FPS/3600:.2f}', flush=True)
    m = Sahadeva(n_vocab(), len(sp))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.LR, weight_decay=1e-6)
    nb, best = len(train.batches(tr, shuffle=False)), math.inf
    vb = train.batches(va, shuffle=False)
    for ep in range(epochs):
        m.train(); t0, tot, k0 = time.time(), 0.0, 0
        for k, b in enumerate(train.batches(tr, seed=ep)):
            for g in opt.param_groups: g['lr'] = train.one_cycle(ep * nb + k, epochs * nb)
            loss, l1, _ = train.loss_fn(m, train.collate(b, mu, sd))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += l1; k0 += 1
        m.eval()
        with torch.no_grad(): vs = float(np.mean([train.loss_fn(m, train.collate(b, mu, sd))[1] for b in vb])) if vb else float('nan')
        if vs < best:
            best = vs; torch.save(dict(model=m.state_dict(), mu=mu, sd=sd, spk={s: i for i, s in enumerate(sorted(sp))}), out / 'best.pt')
        log(f'epoch {ep+1:3}/{epochs} train_l1={tot/max(k0,1):.4f} val_l1={vs:.4f} best={best:.4f} {time.time()-t0:.0f}s', flush=True)
    return out, best

if __name__ == '__main__':
    kw = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:] if '=' in a)
    run(kw.pop('speakers'), **{k: (int(v) if v.isdigit() else v) for k, v in kw.items()})
