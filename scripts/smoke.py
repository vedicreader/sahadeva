"""End-to-end plumbing check on a tiny subset: real code paths, real model config, ~1 minute.

Run this before any long training job. It exercises every stage a full run touches — split,
collate, forward, backward, checkpoint, reload, duration prediction, vocoding, page — and
asserts the invariants that have actually broken in practice.
"""
import shutil, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sd import cfg, data, train, synth, eval as ev
from sd.model import Sahadeva
from sd.text import n_vocab, VOCAB

OK, BAD = '  ok  ', ' FAIL '
fails = []
def check(cond, msg, extra=''):
    print(f'[{OK if cond else BAD}] {msg}{" — " + str(extra) if extra else ""}')
    if not cond: fails.append(msg)

def main(epochs=10, n_train=48, n_val=8, n_synth=2):
    t0 = time.time()
    out = Path(cfg.RUN_DIR).parent / 'smoke'
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True, exist_ok=True)

    print('\n== 1. data')
    tr, va, te, spk = data.load_split()
    check(len(tr) > 0 and len(te) > 0, 'split non-empty', f'train={len(tr)} val={len(va)} test={len(te)} spk={len(spk)}')
    it = tr[0]
    check(len(it['tok']) == len(it['dur']), 'tokens and durations same length', f"{len(it['tok'])}")
    check(int(it['dur'].sum()) == len(it['mel']), 'durations sum to mel frames', f"{int(it['dur'].sum())} == {len(it['mel'])}")
    check(it['mel'].shape[1] == cfg.N_MEL, 'mel width matches vocoder contract', it['mel'].shape)
    check(int(it['tok'].max()) < n_vocab(), 'token ids inside vocab', f"max={int(it['tok'].max())} vocab={n_vocab()}")
    bad = [i for i in tr[:500] if int(i['dur'].sum()) != len(i['mel'])]
    check(not bad, 'duration/frame invariant holds across 500 clips', f'{len(bad)} violations')

    print('\n== 2. model')
    torch.manual_seed(0)
    m = Sahadeva(n_vocab(), len(spk))
    check(sum(p.numel() for p in m.parameters()) > 0, 'model builds', f'{sum(p.numel() for p in m.parameters())/1e6:.2f}M params')
    check(torch.get_num_threads() > 0, 'flush_denormal set at import (see sd/model.py)', f'threads={torch.get_num_threads()}')

    print('\n== 3. train loop')
    sub, subv = tr[:n_train], va[:n_val] or tr[n_train:n_train + n_val]
    mu, sd = train.norm_stats(sub)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.LR)
    losses, tstep = [], time.time()
    for ep in range(epochs):
        m.train(); tot, nb = 0.0, 0
        for b in train.batches(sub, seed=ep):
            bb = train.collate(b, mu, sd)
            loss, l1, ld = train.loss_fn(m, bb)
            check(np.isfinite(loss.item()), f'loss finite (epoch {ep})', loss.item()) if not np.isfinite(loss.item()) else None
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += l1; nb += 1
        losses.append(tot / max(nb, 1))
    dt = time.time() - tstep
    check(all(np.isfinite(losses)), 'all losses finite', [round(x, 3) for x in losses])
    check(losses[-1] < losses[0], 'loss decreased over 10 epochs', f'{losses[0]:.3f} -> {losses[-1]:.3f}')
    print(f'         {dt:.1f}s for {epochs} epochs on {len(sub)} clips')

    print('\n== 4. checkpoint round-trip')
    ck = out / 'best.pt'
    torch.save(dict(model=m.state_dict(), mu=mu, sd=sd, spk=spk), ck)
    m2, mu2, sd2, spk2 = synth.load(ck)
    check(spk2 == spk, 'speaker map survives reload')
    a = m2(torch.as_tensor(sub[0]['tok'])[None], torch.tensor([sub[0]['spk']]), torch.as_tensor(sub[0]['dur'])[None].long())[1]
    m.eval()
    b = m(torch.as_tensor(sub[0]['tok'])[None], torch.tensor([sub[0]['spk']]), torch.as_tensor(sub[0]['dur'])[None].long())[1]
    check(torch.allclose(a, b, atol=1e-5), 'reloaded model matches in-memory model')

    print('\n== 5. inference paths')
    it = te[0]
    mel_gt, d_gt = synth.mel_from_tokens(m2, mu2, sd2, it['tok'], it['spk'], dur=it['dur'])
    check(mel_gt.shape == (int(it['dur'].sum()), cfg.N_MEL), 'gt-duration mel has exact expected length', mel_gt.shape)
    mel_pr, d_pr = synth.mel_from_tokens(m2, mu2, sd2, it['tok'], it['spk'])
    check(mel_pr.shape[0] == int(d_pr.sum()) and d_pr.min() >= 1, 'predicted durations expand consistently',
          f'frames={mel_pr.shape[0]} min_dur={int(d_pr.min())}')
    toks = synth.text_tokens('अ॒ग्निमी॑ळे पु॒रोहि॑तं')
    check(len(toks) > 5 and toks.max() < n_vocab(), 'text-only tokenisation works (svara kept)', f'{len(toks)} tokens')

    print('\n== 6. vocoder')
    w = synth.vocode(mel_gt)
    exp = mel_gt.shape[0] * cfg.HOP
    check(abs(len(w) - exp) < cfg.HOP * 4, 'waveform length matches mel frames', f'{len(w)} vs {exp}')
    check(np.isfinite(w).all() and np.abs(w).max() > 1e-3, 'waveform finite and non-silent', f'peak={np.abs(w).max():.3f}')
    ref = synth.vocode(it['mel'].astype(np.float32))
    check(np.abs(ref).max() > 1e-3, 'copy-synth of a real clip is non-silent', f'peak={np.abs(ref).max():.3f}')

    print('\n== 7. eval + page')
    rows = ev.evaluate(ck=ck, out=out, limit=n_synth, per_spk=None, log=lambda *a, **k: None)
    check(len(rows) == n_synth, 'eval wrote metrics rows', len(rows))
    wavs = sorted(out.glob('*.wav'))
    check(len(wavs) == n_synth * 4, 'four wav versions per clip', f'{len(wavs)} files')
    check(all(np.isfinite(r['mel_l1']) for r in rows), 'mel L1 finite', [round(r['mel_l1'], 3) for r in rows])
    pg = ev.page(rows, d=out)
    check(pg.exists() and pg.stat().st_size > 10_000, 'comparison page written', f'{pg.stat().st_size/1e6:.2f} MB')

    print(f'\n{"ALL PASSED" if not fails else str(len(fails)) + " FAILED: " + "; ".join(fails)}  ({time.time()-t0:.0f}s)')
    return 1 if fails else 0

if __name__ == '__main__':
    kw = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:] if '=' in a)
    sys.exit(main(**{k: int(v) for k, v in kw.items()}))
