# Track A fine-tune runbook

Every flag below is from `f5_tts/train/finetune_cli.py` (F5-TTS 1.1.22). Defaults in that CLI
are tuned for Emilia-scale pretraining — tens of thousands of hours — and several of them are
actively wrong for a 2-hour corpus. The ones that matter are called out.

## 0. Prerequisites

IndicF5 is **gated**. Request access at <https://huggingface.co/ai4bharat/IndicF5> before
anything else; the model listing is public but file reads are not.

```sh
pip install f5-tts                      # do not use uv sync here, see plan §8
huggingface-cli login
huggingface-cli download ai4bharat/IndicF5 --local-dir ckpt/indicf5
```

## 1. Data

```sh
sd-ingest  --vr-root ../vedicreader --out data/manifest.jsonl
sd-segment --manifest data/manifest.jsonl
sd-qc      --manifest data/manifest.jsonl --out data/manifest.qc.jsonl
sd-splits  --manifest data/manifest.qc.jsonl
sd-prep    --split data/train.jsonl --out data/f5/metadata.csv --vocab ckpt/indicf5/checkpoints/vocab.txt
```

`sd-prep`'s vocab report is not optional reading. F5-TTS's fine-tune path copies the pretrained
vocab and silently drops characters it does not know — for accented text that means the svara
marks vanish without a warning, and you would not find out until you listened to a checkpoint
several hours later.

Then hand off to F5-TTS's own preparation, which writes the arrow dataset and duration index:

```sh
python -m f5_tts.train.datasets.prepare_csv_wavs data/f5/metadata.csv data/f5/sahadeva_char
```

## 2. Sizing the run

Useful arithmetic, since the defaults assume a corpus four orders of magnitude larger.

At 24 kHz with hop 256, one second is ~94 mel frames. The usable corpus is ~2.0h after QC and
hard-set holdout, so roughly **680k frames per epoch**.

| `--batch_size_per_gpu` (frames) | updates/epoch | epochs for ~2k updates |
|---|---|---|
| 3200 (CLI default) | ~212 | ~10 |
| 6000 | ~113 | ~18 |
| 8000 | ~85 | ~24 |

Target **1.5–2.5k updates total**. Past that, a 2h corpus overfits: the voice sharpens for a
while and then the model starts reproducing the training verses' exact phrasing regardless of
what you asked it to say.

## 3. The run

```sh
accelerate launch -m f5_tts.train.finetune_cli \
  --exp_name F5TTS_v1_Base \
  --dataset_name sahadeva_char \
  --finetune --pretrain ckpt/indicf5/model.safetensors \
  --tokenizer custom --tokenizer_path ckpt/indicf5/checkpoints/vocab.txt \
  --learning_rate 1e-5 \
  --batch_size_type frame --batch_size_per_gpu 6000 \
  --grad_accumulation_steps 2 \
  --max_samples 32 \
  --num_warmup_updates 200 \
  --epochs 18 \
  --save_per_updates 500 --last_per_updates 250 \
  --log_samples --logger tensorboard
```

Flags that differ from the CLI defaults, and why:

- `--num_warmup_updates 200` — the default is **20,000**, which is more updates than this entire
  run has. Left alone, the learning rate never leaves warmup and nothing trains.
- `--learning_rate 1e-5` — bottom of the plan's 1e-5–4e-5 range. With 2h and a voice you are
  adapting rather than teaching from scratch, higher rates wash out the pretrained phonetics.
- `--tokenizer custom` — required. The default `pinyin` tokenizer will mangle Kannada.
- `--max_samples 32` — halved from 64; with frame-based batching this caps sequences per batch
  and keeps memory predictable on a single card.

## 4. Memory

At 16 GB, `--batch_size_per_gpu 6000` with `--grad_accumulation_steps 2` is a reasonable start.
If it OOMs, in this order: halve `batch_size_per_gpu` and double `grad_accumulation_steps`
(keeps the effective batch identical), then drop `--max_samples` to 16, then add
`--bnb_optimizer` for the 8-bit optimizer.

Note that the effective batch is what governs convergence, so trading batch size for
accumulation is close to free — it costs wall-clock, not quality.

BigVGAN-v2 vocoder fine-tuning is deliberately **not** part of this runbook. Vagdhenu treats it
as mandatory for long vowels, and they are right, but it is a separate training job with its own
memory profile and 2h of source audio is thin for it. Use the stock vocoder first and establish
whether the vocoder is actually your bottleneck before spending a GPU-week on it.

## 5. Arms worth running

The corpus supports a genuine A/B that nothing upstream has run, because no one else has an
accented Sanskrit TTS corpus with hand-verified timings:

| arm | prep command | question |
|---|---|---|
| svara-inline | `sd-ingest` (default) | can the model learn accent from in-vocab marks? |
| svara-stripped | `sd-ingest --no-svara` | how much do the marks actually buy? |

Both arms share segmented audio — only the text field differs — so the second arm costs one
ingest and one training run, not a re-cut of the corpus. Score both with
`python -m sahaeval.svara`, which is the metric this comparison exists to move.

Keep the two checkpoints. The stripped arm is the honest baseline; without it, "the model
learned svara" is an assertion rather than a result.
