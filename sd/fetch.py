"Download audio_alignment source audio (Internet Archive etc.) and build standardised align records."
import json, subprocess, sys
from pathlib import Path
from . import cfg, corpus

def _get(url, dst, tries=4):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 10_000: return True
    for k in range(tries):
        r = subprocess.run(['curl', '-sSL', '--retry', '3', '--max-time', '900', '-o', str(dst), url], capture_output=True, text=True)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000: return True
        print(f'  retry {k+1} {url}: {r.stderr.strip()[:120]}', flush=True)
    dst.unlink(missing_ok=True); return False

def entries(corpus_nm, n):
    d = json.load(open(cfg.ALIGN_REF / 'data' / corpus_nm / 'data.json', encoding='utf-8'))['data']
    d = [e for e in d if e.get('word_alignment')]
    return d if n < 0 else d[:n]

def fetch_corpus(corpus_nm, n, speaker):
    out = []
    for e in entries(corpus_nm, n):
        aud = cfg.AUD_DIR / corpus_nm / Path(e['audio_url'].split('?')[0]).name
        if not _get(e['audio_url'], aud): print(f'  SKIP {e["id"]} (download failed)', flush=True); continue
        rec = corpus.from_audio_alignment(corpus_nm, e, speaker, aud)
        if not rec['lines']: print(f'  SKIP {e["id"]} (no aligned words)', flush=True); continue
        corpus.save(rec); out.append(rec)
        print(f'  {corpus_nm}/{e["id"]}: {len(rec["lines"])} lines, {aud.stat().st_size/1e6:.1f}MB', flush=True)
    return out

def main():
    recs = []
    for nm, n, spk in cfg.MANIFEST:
        if not n: continue
        print(f'== {nm} ({n} files, speaker={spk})', flush=True)
        recs += fetch_corpus(nm, n, spk)
    for spk, a in sorted(corpus.stats(recs).items()): print(f'{spk:14} files={a["files"]:3} lines={a["lines"]:6} hours={a["secs"]/3600:5.2f}')

if __name__ == '__main__': main()
