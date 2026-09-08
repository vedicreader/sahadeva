"Train sahadeva on CPU: durations come from the aligner, so there is no attention to learn."
import json, math, sys, time
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from . import cfg, data
from .text import n_vocab
from .model import Sahadeva

__all__ = ['batches', 'norm_stats', 'one_cycle', 'run']

def norm_stats(items):
    "Per-mel-bin mean/std over the training clips."
    m = np.concatenate([it['mel'][::7] for it in items]).astype(np.float32)
    return m.mean(0), m.std(0) + 1e-5

def _pad(xs, n, v=0):
    return torch.stack([F.pad(torch.as_tensor(np.asarray(x)), (0,) * 2 * (np.asarray(x).ndim - 1) + (0, n - len(x)), value=v) for x in xs])

def collate(items, mu, sd):
    nt, nf = max(len(i['tok']) for i in items), max(len(i['mel']) for i in items)
    mel = _pad([(i['mel'].astype(np.float32) - mu) / sd for i in items], nf)
    return dict(tok=_pad([i['tok'] for i in items], nt).long(), dur=_pad([i['dur'] for i in items], nt).long(),
                spk=torch.tensor([i['spk'] for i in items]), mel=mel,
                tmask=_pad([np.ones(len(i['tok'])) for i in items], nt).float(),
                fmask=_pad([np.ones(len(i['mel'])) for i in items], nf).float())

def batches(items, budget=cfg.BATCH_FRAMES, shuffle=True, seed=0):
    "Length-bucketed batches under a frame budget, so padding stays cheap on CPU."
    idx = sorted(range(len(items)), key=lambda i: len(items[i]['mel']))
    out, cur = [], []
    for i in idx:
        n = len(items[i]['mel'])
        if cur and (len(cur) + 1) * n > budget: out.append(cur); cur = []
        cur.append(i)
    if cur: out.append(cur)
    if shuffle: np.random.default_rng(seed).shuffle(out)
    return [[items[i] for i in b] for b in out]

def one_cycle(step, total, lr=cfg.LR, warm=0.25, floor=0.02):
    "One-cycle LR: linear warmup then cosine decay, stepped per batch rather than per epoch."
    p = min(max(step / max(total, 1), 0.0), 1.0)
    if p < warm: return lr * (0.02 + 0.98 * p / warm)
    q = (p - warm) / max(1 - warm, 1e-9)
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * q)))

def loss_fn(m, b):
    pre, post, logd, _ = m(b['tok'], b['spk'], b['dur'])
    fm, tm = b['fmask'][..., None], b['tmask']
    n = fm.sum().clamp(min=1)
    l1 = ((pre - b['mel']).abs() * fm).sum() / n / cfg.N_MEL + ((post - b['mel']).abs() * fm).sum() / n / cfg.N_MEL
    ld = (((logd - torch.log(b['dur'].float() + 1)) ** 2) * tm).sum() / tm.sum().clamp(min=1)
    return l1 + 0.5 * ld, l1.item(), ld.item()

def run(epochs=cfg.EPOCHS, out=None, threads=4, log=print, resume=True):
    torch.set_num_threads(threads); torch.manual_seed(cfg.SEED)
    out = Path(out or cfg.RUN_DIR); out.mkdir(parents=True, exist_ok=True)
    tr, va, te, spk = data.load_split()
    mu, sd = norm_stats(tr)
    log(f'train={len(tr)} val={len(va)} test={len(te)} speakers={len(spk)} '
        f'hours={sum(len(i["mel"]) for i in tr)/cfg.FPS/3600:.2f}')
    m = Sahadeva(n_vocab(), len(spk))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.LR, weight_decay=1e-6)
    ck, ep0, best = out / 'ckpt.pt', 0, math.inf
    if resume and ck.is_file():
        st = torch.load(ck, map_location='cpu', weights_only=False)
        m.load_state_dict(st['model']); opt.load_state_dict(st['opt']); ep0, best = st['epoch'], st['best']
        mu, sd = st['mu'], st['sd']; log(f'resumed at epoch {ep0} (best {best:.4f})')
    vb, nb_ep = batches(va, shuffle=False), len(batches(tr, shuffle=False))
    save = lambda e, b: torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), epoch=e, best=b, mu=mu, sd=sd,
                                        spk=spk, cfg=dict(d=cfg.D_MODEL, n_mel=cfg.N_MEL)), ck)
    for ep in range(ep0, epochs):
        m.train(); t0, tot, nb = time.time(), 0.0, 0
        for k, b in enumerate(batches(tr, seed=ep)):
            for g in opt.param_groups: g['lr'] = one_cycle(ep * nb_ep + k, epochs * nb_ep)
            b = collate(b, mu, sd)
            loss, l1, ld = loss_fn(m, b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += l1; nb += 1
        m.eval(); vs = 0.0
        with torch.no_grad():
            for b in vb: vs += loss_fn(m, collate(b, mu, sd))[1]
        vs /= max(len(vb), 1)
        if vs < best: best = vs; save(ep + 1, best); torch.save(dict(model=m.state_dict(), mu=mu, sd=sd, spk=spk), out / 'best.pt')
        else: save(ep + 1, best)
        log(f'epoch {ep+1:3}/{epochs} train_l1={tot/max(nb,1):.4f} val_l1={vs:.4f} best={best:.4f} {time.time()-t0:.0f}s', flush=True)
    return m

if __name__ == '__main__':
    kw = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:] if '=' in a)
    run(epochs=int(kw.get('epochs', cfg.EPOCHS)), threads=int(kw.get('threads', 4)))
