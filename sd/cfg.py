"Paths, audio/feature params, model dims and corpus manifest for sahadeva."
import os
from pathlib import Path

ROOT = Path(os.getenv('SD_ROOT', Path(__file__).resolve().parent.parent))
DATA = Path(os.getenv('SD_DATA', ROOT / 'data'))
ALIGN_REF = Path(os.getenv('SD_ALIGN_REF', '/tmp/claude-0/-home-user/5011b077-7330-5109-b3d1-7d72c56c0b30/scratchpad/audio_alignment'))
VR_ROOT = Path(os.getenv('SD_VR_ROOT', '/home/user/vedicreader'))

AUD_DIR, ALN_DIR, CLIP_DIR, FEAT_DIR, RUN_DIR, OUT_DIR = (DATA / x for x in ['audio', 'align', 'clips', 'feats', 'runs', 'out'])

# === audio / features (must match microsoft/speecht5_hifigan: 16kHz, 80 log-mel, 256 hop) ===
SR          = 16000
HOP         = 256
N_MEL       = 80
FPS         = SR / HOP           # 62.5 frames/sec
VOCODER     = 'microsoft/speecht5_hifigan'

# === clipping ===
CLIP_MIN_MS, CLIP_MAX_MS = 1500, 7000
PAD_MS      = 60                 # context kept around a line
GAP_SIL_MS  = 90                 # inter-word gap that becomes an explicit <sil> token

# === model ===
D_MODEL     = 256
ENC_DILS    = (1, 2, 4, 8)
DEC_DILS    = (1, 2, 4, 8, 1, 2, 4, 1)
POST_BLOCKS = 5
KERNEL      = 5
DROPOUT     = 0.1

# === train ===
BATCH_FRAMES = 6000              # dynamic batching budget
LR          = 2e-3
EPOCHS      = int(os.getenv('SD_EPOCHS', 80))
SEED        = 1234

# === corpora: (corpus, n_files, speaker) — n_files=0 skips, -1 takes all ===
MANIFEST = [
    ('ramayana',        24, 'ghanapati'),
    ('kumarasambhavam',  8, 'vedabhoomi'),
    ('meghaduta',        6, 'vedabhoomi'),
    ('yogasutra',        4, 'jayashree'),
    ('tarkasangraha',    1, 'av'),
]
# vedicreader library titles are their own speakers (one reciter per recording)
VR_SPK_PREFIX = 'vr'
