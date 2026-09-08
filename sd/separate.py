"Vocal separation via audio_separator, run out-of-process so its torch stack stays isolated."
import json, subprocess, sys
from pathlib import Path
from . import cfg, corpus

__all__ = ['separate', 'vocals_pth', 'separated_records']

_RUNNER = r'''
import json, logging, sys, warnings
warnings.filterwarnings("ignore")
from audio_separator.separator import Separator
spec = json.loads(sys.argv[1])
s = Separator(output_dir=spec["out"], output_format="wav", log_level=logging.ERROR)
s.load_model(spec["model"])
for src, stem in spec["jobs"]:
    outs = s.separate(src)
    print(json.dumps(dict(src=src, stem=stem, outs=outs)), flush=True)
'''

def vocals_pth(audio, out=None): return Path(out or cfg.SEP_DIR) / f'{Path(audio).stem}.vocals.wav'

def separate(paths, out=None, model=None, log=print):
    "Write a mono vocals wav per input. Skips files already done; returns {src: vocals path}."
    out = Path(out or cfg.SEP_DIR); out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not vocals_pth(p, out).is_file()]
    done = {str(p): vocals_pth(p, out) for p in paths if vocals_pth(p, out).is_file()}
    if not todo: return done
    spec = dict(out=str(out), model=model or cfg.SEP_MODEL, jobs=[[str(p), Path(p).stem] for p in todo])
    pr = subprocess.Popen([cfg.SEP_PY, '-c', _RUNNER, json.dumps(spec)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for ln in pr.stdout:
        try: r = json.loads(ln)
        except ValueError: continue
        vox = next((o for o in r['outs'] if 'Vocals' in o), None)
        if not vox: log(f'  no vocals stem for {r["stem"]}'); continue
        src = Path(out) / vox if not Path(vox).is_absolute() else Path(vox)
        dst = vocals_pth(r['src'], out)
        subprocess.run(['ffmpeg', '-y', '-v', 'quiet', '-i', str(src), '-ac', '1', '-ar', '44100', str(dst)], check=True)
        Path(src).unlink(missing_ok=True)
        for o in r['outs']:
            q = Path(out) / o if not Path(o).is_absolute() else Path(o)
            if 'Instrumental' in str(q): q.unlink(missing_ok=True)
        done[r['src']] = dst; log(f'  separated {Path(r["src"]).name} → {dst.name}')
    pr.wait()
    return done

def separated_records(speakers=None, out=None, dst=None, log=print):
    "Re-point align records at their separated vocals, so the clip builder is otherwise unchanged."
    recs = [r for r in corpus.load_all() if not speakers or r['speaker'] in speakers]
    got = separate(sorted({r['audio'] for r in recs}), out=out, log=log)
    n = 0
    for r in recs:
        v = got.get(r['audio'])
        if not v: continue
        r['audio'] = str(v); corpus.save(r, dst); n += 1
    log(f'{n} records re-pointed at vocals')
    return recs

def main():
    kw = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:] if '=' in a)
    sp = kw.get('speakers', '').split(',') if kw.get('speakers') else None
    separated_records(speakers=sp, dst=kw.get('dst'))

if __name__ == '__main__': main()
