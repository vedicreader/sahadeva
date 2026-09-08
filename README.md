# sahadeva

Sanskrit chant TTS: a svara-aware text frontend, a forced-aligned corpus over VedicReader and
`audio_alignment`, and two training tracks.

The differentiator is **Vedic accent**. Vagdhenu — the current open SOTA for Sanskrit chant — is
trained on Bhāgavatam, which is unaccented classical verse, so its frontend passes svara marks
through as inert characters. The VedicReader corpus is Taittirīya Kṛṣṇa Yajurveda, where svara
*is* the melody. That gap is what this repo is built around.

## Two tracks

| | Track A — F5/IndicF5 fine-tune | Track B — from scratch |
|---|---|---|
| model | IndicF5, LoRA/full fine-tune | 4.5M-param conv acoustic model |
| vocoder | BigVGAN-v2 (F5 stock) | `microsoft/speecht5_hifigan`, frozen |
| needs | GPU + **gated** IndicF5 access | CPU only, no gated weights |
| status | runbook in `configs/finetune.md` | trains and synthesises today |
| quality ceiling | far higher | limited; see below |

Track A is the right answer where a GPU and IndicF5 access exist. Track B exists because neither
is guaranteed, and a pipeline you cannot run end-to-end is a pipeline you cannot debug.

## Quick start

```sh
uv venv --python 3.12 && uv pip install -e ".[data,train,dev]"
pytest                                      # 50 tests, no corpus required

# Track B
sd-fetch          # audio_alignment corpora (Internet Archive) + their aeneas alignments
sd-build          # clips -> phone tokens, per-token frame durations, log-mel, QC
sd-train          # CPU training
sd-eval           # test-set wavs + a self-contained original-vs-generated page

# Track A
sd-ingest --vr-root ../vedicreader --out data/manifest.jsonl
sd-segment --manifest data/manifest.jsonl
sd-qc --manifest data/manifest.jsonl --out data/manifest.qc.jsonl
sd-splits --manifest data/manifest.qc.jsonl
```

`ffmpeg` must be on `PATH`. VedicReader mp3s live in git-lfs; `git lfs pull` them first.

## On forced alignment

An earlier revision of this README argued that forced alignment was unnecessary, because every
timed line in VedicReader carries `timestamp_fixed="true"`. The flag claim is correct — all six
timed titles are fully marked. The conclusion was not.

Measured against a fresh word-level alignment, like-for-like on contiguous spans, the committed
timings are internally inconsistent. **Rate CV** is the spread of ms-per-syllable across a
title's lines; a reciter holds a fairly steady tempo, so a high spread means lines are carrying
audio that is not theirs:

| title | committed | word-level aeneas |
|---|---|---|
| rudram_namakam | 2.236 | **0.568** |
| lalitha_sahasranamam | 1.858 | **0.452** |
| vishnu_sahasranamam | 1.348 | **0.397** |
| lalitha_trishati | 1.808 | **1.090** |
| gayathri_dhyaanam | 1.304 | **1.077** |
| rudram_chamakam | 0.519 | 0.536 *(committed wins)* |

`timestamp_fixed` records intent, not verified accuracy. Alignment is also not optional for the
`audio_alignment` corpora, which have no hand timings at all — and those are what take the
dataset from 2.13 h to 8.7 h. The aligner itself lives in vedicreader (`vr/align/engine.py`,
`uv run vr-align`); this repo consumes its sidecar output.

## Data

Two sources, one schema. `sd/corpus.py` normalises both into:

```json
{"source": "...", "corpus": "...", "id": "...", "speaker": "...", "audio": "...",
 "lines": [{"i": 0, "text": "...", "s": 0, "e": 2450,
            "words": [{"t": "अस्त्युत्तरस्यां", "s": 0, "e": 2970}]}]}
```

All times are integer milliseconds.

| Source | How it arrives |
|---|---|
| [`avinashvarna/audio_alignment`](https://github.com/avinashvarna/audio_alignment) | aeneas `mplain`/levels=3 JSON, audio from Internet Archive; word ids `pNsNwN` regrouped into sentences |
| `vedicreader` | Align Studio sidecars, `data/align/sidecar/*.words.json` |

Current manifest (`sd/cfg.py: MANIFEST`): Rāmāyaṇa Kanda 1, Kumārasambhavam, Meghadūta,
Yogasūtra, Tarkasaṅgraha, plus six vedicreader titles — **6,894 clips / 10 reciters**, of which
6,449 (8.0 h) pass QC.

It is ten speakers, not one voice. Each recording is separately sourced and labelled as its own
speaker; `consent` defaults to `unverified`. None should become a cloning target without
recorded provenance.

## Pipeline

**`dhvani/`** — the text frontend, and the highest-leverage module. Any Brahmic → Devanagari →
SLP1 → Kannada, svara extraction under the Taittirīya reading (U+0951 is *svarita*, despite its
Unicode name), sandhi policies, and laghu/guru scansion. Adapted from `prathoshap/vagdhenu`
(Apache-2.0).

**`sd/text.py`** — Track B tokens. SLP1 characters are the phones; svara rides the same grid as
its own token, attached after the akshara it marks. Built on dhvani, so accent semantics match
Track A rather than diverging from it. 50 tokens cover the corpus with no OOV.

**`sd/data.py`** — clipping. Lines longer than `CLIP_MAX_MS` split at their widest inter-word
gap; short ones merge. Each word's measured span divides across its phones by weight, quantised
to frames summing exactly to the clip's mel length. Inter-word gaps above `GAP_SIL_MS` become
explicit `<sil>` tokens carrying the pause — that is what reproduces the rhythm of recitation.
Clips are QC'd through `sdata.qc` and flagged, never deleted; `load_split` decides.

**`sd/model.py`** — dilated-conv encoder, duration predictor, length regulator, dilated-conv
decoder, mel postnet, per-reciter speaker embedding. Because the aligner supplies every phone's
duration there is no attention to learn, which is what makes CPU training viable.

**Vocoder** — `microsoft/speecht5_hifigan`, frozen. Copy-synthesis through it measures mel L1
0.081 (corr 0.991), so the vocoder is not the ceiling; the acoustic model is.

## Evaluation

`sahaeval/` — svara adherence (with its calibration ceiling documented in the module docstring;
read it before interpreting any number), conjunct accuracy, CER/WER.

`sd-eval` holds out whole clips per reciter and writes four versions of each:

| version | what it isolates |
|---|---|
| `orig` | the source recording |
| `copysynth` | original mel → vocoder — the ceiling |
| `gen` | model from text alone, predicted durations |
| `gen_gtdur` | model with the aligner's durations — spectral quality without rhythm error |

## Known limitations

- **Corpus SNR is mediocre** — median ~15 dB. These are sourced recordings, not studio takes.
  This caps achievable MOS regardless of model quality.
- **Accent is thin.** Only Rudram Namakam carries real Vedic accent, ~0.34 h. Enough for voice
  adaptation; thin for teaching a tonal system.
- The svara metric warps syllables uniformly across the voiced span. Force-aligning the
  synthesized output and measuring per syllable would be strictly better — and the aligner to
  do it with is now in the repo's reach.
- Track B's duration weights are a vowel-length heuristic. `dhvani.meter.weights` already
  computes true laghu/guru, including position-by-conjunct; wiring that in is the obvious next
  improvement.

## Layout

```
dhvani/     text frontend — translit, svara, sandhi, meter
sdata/      Track A corpus: ingest -> segment -> qc -> splits
sd/         Track B: corpus standardisation, fetch, tokens, clips, model, train, synth, eval
train/      F5 dataset prep + vocab preflight
infer/      rendering, reference selection, device routing
sahaeval/   svara adherence, conjunct accuracy, CER/WER
configs/    Track A training runbook
```

## Attribution

`dhvani/translit.py` and `dhvani/sandhi.py` adapt `prathoshap/vagdhenu` `src/prep_text.py`
(Apache-2.0). Dataset and training formats follow F5-TTS 1.1.22 (MIT). IndicF5 is MIT but
**gated**. The Track B vocoder is `microsoft/speecht5_hifigan` (MIT).

Synthesizing sacred recitation in an identifiable voice carries impersonation risk. Use your own
or a consented voice; the manifest carries `consent` and `license` fields so that provenance is
recorded rather than assumed.
