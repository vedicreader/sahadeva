# sahadeva

Sanskrit chant TTS: a svara-aware text frontend, a data pipeline over the VedicReader corpus,
and Track A training/inference against F5/IndicF5.

The differentiator is **Vedic accent**. Vagdhenu — the current open SOTA for Sanskrit chant — is
trained on Bhāgavatam, which is unaccented classical verse, so its frontend passes svara marks
through as inert characters. The VedicReader corpus is Taittirīya Kṛṣṇa Yajurveda, where svara
*is* the melody. That gap is what this repo is built around.

## Quick start

```sh
pip install -e ".[data,dev]"
pytest                                      # 50 tests, no corpus required

sd-ingest  --vr-root ../vedicreader --out data/manifest.jsonl
sd-segment --manifest data/manifest.jsonl
sd-qc      --manifest data/manifest.jsonl --out data/manifest.qc.jsonl
sd-splits  --manifest data/manifest.qc.jsonl
```

Then `configs/finetune.md` for training. The VedicReader mp3s live in git-lfs; fetch them with
`git lfs pull --include='static/vedic_texts/**/*.mp3'` before segmenting.

## What the corpus actually is

Measured, not assumed:

| | utterances | audio |
|---|---|---|
| 5 stotras + 1 japa, all timed | 1,334 | **2.13 h** |
| in the 3–15s window | 1,234 (93%) | |
| accented (Vedic svara) | 233 | |
| namavalis | — | text only, no recordings |

Three properties that shaped every design decision here:

**The alignment problem does not exist.** VedicReader's XML carries hand-corrected line timings
marked `timestamp_fixed="true"`, on every line. These are better than anything MFA or aeneas
would produce, so there is no forced-alignment stage — the single largest chunk of a normal TTS
data pipeline is replaced by reading a column.

**It is six speakers, not one voice.** The recordings are separately sourced, so each title is
labelled as its own speaker and `consent` defaults to `unverified`. Mixing them into a
single-speaker target would produce a blurry averaged voice, and none of them should become a
cloning target without recorded provenance. A consented target voice slots into the same
pipeline unchanged.

**Only Rudram Namakam carries real accent** — 227 of the 233 accented utterances, about 0.34 h.
That is thin for teaching a tonal distinction, and it is the main risk to the svara result.

## Layout

```
dhvani/     text frontend — translit, svara, sandhi, meter        (the highest-leverage module)
sdata/      ingest -> segment -> qc -> splits
train/      F5 dataset prep + vocab preflight
infer/      rendering, reference selection, device routing
sahaeval/   svara adherence, conjunct accuracy, CER/WER
configs/    training runbook
```

## Where this diverges from the original plan

Four corrections, all from reading vagdhenu's actual source rather than its description:

1. **Sandhi defaults to off.** The plan specifies resolving pronunciation-affecting sandhi.
   Vagdhenu's `prep_text.py` says they A/B'd exactly that and found it *worse* — "plain >
   resolved for satva" — and their 4.6-MOS champion applies no sandhi at all. `SandhiT.PLAIN`
   is the default; the other policies are opt-in arms.
2. **No aeneas, no MFA.** The plan budgets forced alignment via `avinashvarna/audio_alignment`.
   The timings already exist and are human-verified.
3. **`indic_transliteration`, not `sanskrit-tokenizer`.** That is what vagdhenu actually uses.
4. **Svara is unhandled everywhere upstream** — not in the plan, not in vagdhenu. It is the
   thing this corpus is uniquely able to support.

## Notes on svara

- Follows the **Taittirīya** reading, in which U+0951 is *svarita*. Unicode names that codepoint
  `DEVANAGARI STRESS SIGN UDATTA`, and taking the name literally inverts the melody. Both
  readings are pinned in tests.
- U+0951 and U+0952 are **in the IndicF5 vocab**, so accent survives tokenization and no vocab
  extension is needed — but IndicF5 trained on IndicVoices-R, which has no accented Vedic text,
  so those embeddings are effectively untrained. Teaching them is what the fine-tune is for.
- U+1CDA (dīrgha svarita) is **absent** from that vocab; dhvani folds it to U+0951, keeping the
  tonal category and losing only length.
- `sahaeval.svara` is calibrated against human ground truth: the real Namakam recording scores
  r≈0.13 and 70% ordering. That is the metric's ceiling, not the reciter's. Read the module
  docstring before interpreting any number from it.

## Known limitations

- **Corpus SNR is mediocre** — median ~15 dB by the QC estimate, against the "studio-clean"
  the plan assumes. The estimate understates for continuous chant, since the p10 energy floor
  is quiet speech rather than silence, but these are sourced recordings and not studio takes.
  This caps achievable MOS regardless of model quality.
- **2.13 h total, 0.34 h accented.** Enough for voice adaptation; thin for teaching a tonal system.
- Accent marks inside conjunct clusters do not round-trip exactly (2 of 2,619 lines). The label
  channel stays correct; only exact glyph position is lost. Pinned as a test.
- The svara metric warps syllables uniformly across the voiced span. Force-aligning the
  synthesized output and measuring per syllable would be strictly better.

## Attribution

`dhvani/translit.py` and `dhvani/sandhi.py` adapt `prathoshap/vagdhenu` `src/prep_text.py`
(Apache-2.0). Dataset and training formats follow F5-TTS 1.1.22 (MIT). IndicF5 is MIT but
**gated** — request access before running anything in `configs/finetune.md`.

Synthesizing sacred recitation in an identifiable voice carries impersonation risk. Use your own
or a consented voice; the manifest carries `consent` and `license` fields so that provenance is
recorded rather than assumed.
