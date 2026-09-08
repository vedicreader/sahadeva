# sahadeva

Sanskrit TTS trained on forced-aligned Vedic recitation.

Text and audio are paired by a **word-level forced aligner**, not by hand. The aligner gives every phone a
measured duration, so the acoustic model never has to learn attention — which is what makes it trainable on
a CPU in a few hours.

```
word-level alignment ─→ phone tokens + frame durations ─→ acoustic model ─→ HiFi-GAN ─→ wav
   (aeneas, mplain L3)        (sd/g2p.py, sd/data.py)      (sd/model.py)     (pretrained, frozen)
```

## Quick start

```sh
uv venv --python 3.12 && uv pip install -e .
uv run sd-fetch     # pull audio_alignment corpora (Internet Archive) + their aeneas alignments
uv run sd-build     # clips → phone tokens, per-token frame durations, log-mel
uv run sd-train     # CPU training
uv run sd-eval      # test-set wavs + a self-contained comparison page
```

`ffmpeg` must be on `PATH`. Everything else is pip-installable.

## Data

Two sources, one schema. `sd/corpus.py` normalises both into:

```json
{"source": "...", "corpus": "...", "id": "...", "speaker": "...", "audio": "...",
 "lines": [{"i": 0, "text": "...", "s": 0, "e": 2450,
            "words": [{"t": "अस्त्युत्तरस्यां", "s": 0, "e": 2970}]}]}
```

All times are integer milliseconds.

| Source | How it arrives | Notes |
|---|---|---|
| [`avinashvarna/audio_alignment`](https://github.com/avinashvarna/audio_alignment) | aeneas `mplain`/levels=3 JSON, audio from Internet Archive | word fragment ids `pNsNwN` are regrouped into sentences |
| `vedicreader` | Align Studio sidecar JSON (`data/align/sidecar/*.words.json`) | produced by the enhanced aeneas backend in `vr/align/engine.py` |

Recordings in the current manifest (`sd/cfg.py: MANIFEST`): Rāmāyaṇa (Kanda 1), Kumārasambhavam, Meghadūta,
Yogasūtra, Tarkasaṅgraha, plus six vedicreader titles — **8.7 h of speech across 10 reciters**.

## Pipeline

**`sd/g2p.py`** — Devanagari → Sanskrit phones, deterministic and dependency-free. Inherent `a` is inserted
unless a virama or matra follows. Vedic svara marks (`॑` udātta, `॒` anudātta) survive as their own tokens,
so the model can learn the chant melody rather than averaging it away.

**`sd/data.py`** — builds clips. Lines longer than `CLIP_MAX_MS` are split at their widest inter-word gap;
short ones are merged. Each word's measured span is divided across its phones by weight (long vowels 2.0,
short 1.25, consonants 1.0, svara marks 0.25), then quantised to frames summing exactly to the clip's mel
length. Inter-word gaps above `GAP_SIL_MS` become explicit `<sil>` tokens carrying the pause duration — this
is what reproduces the rhythm of recitation.

**`sd/model.py`** — 4.5M-parameter non-autoregressive acoustic model: dilated-conv encoder, duration
predictor, length regulator, dilated-conv decoder, mel postnet, learned per-reciter speaker embedding.
Loss is L1 on mel (pre- and post-net) plus MSE on log duration.

**Vocoder** — `microsoft/speecht5_hifigan`, pretrained and frozen. Its 16 kHz / 80-bin / 256-hop log-mel is
the model's output format. Copy-synthesis through it measures mel L1 0.081 (corr 0.991), so the vocoder is
not the quality ceiling — the acoustic model is.

## Evaluation

`sd-eval` holds out whole clips per reciter and writes four versions of each:

| Version | What it isolates |
|---|---|
| `orig` | the source recording |
| `copysynth` | original mel → vocoder — the ceiling |
| `gen` | model from text alone, predicted durations |
| `gen_gtdur` | model with the aligner's durations — spectral quality without rhythm error |

Metrics: mel L1 against the original, duration RMSE in frames, and total length ratio.
`data/out/sahadeva_test_set.html` inlines every clip as base64 audio, so it opens offline.

## Layout

```
sd/
├── cfg.py      # paths, audio/feature params, model dims, corpus manifest
├── corpus.py   # vedicreader XML + audio_alignment JSON → one align schema
├── fetch.py    # Internet Archive audio download
├── g2p.py      # Devanagari → Sanskrit phone tokens
├── data.py     # clipping, durations, log-mel, splits
├── model.py    # acoustic model
├── train.py    # CPU training loop
├── synth.py    # inference + HiFi-GAN
└── eval.py     # test-set comparison + report
```
