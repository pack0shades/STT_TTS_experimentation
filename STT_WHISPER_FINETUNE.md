# Whisper fine-tuning notes (dataset generation + next steps)

This repo currently focuses on TTS comparisons. To fine-tune Whisper for your sales voice-bot, you mainly need a dataset of:

- **audio**: customer speech recordings
- **text**: the exact transcript (what was said)

## Synthetic dataset (bootstrapping)

If you don’t have real customer audio yet, you can bootstrap with synthetic speech.
This repo includes:

- `stt_prompts_gen_retail_sales.py` — generates ~20k retail/sales-style prompts (text only)
- `stt_dataset_gen_melotts.py` — uses MeloTTS to generate WAVs + JSONL manifests

**Generate a dataset:**

Using your workspace venv:

- `python` path: `/home/pragay/WWAI/.venv/bin/python`

Suggested (more realistic) command for a sales voice-bot (adds mild noise + phone-call simulation + a bit of silence padding):

- prompts: `data/sales_prompts_en.txt`
- output: `datasets/sales_synth_melotts/`

  /home/pragay/WWAI/.venv/bin/python stt_dataset_gen_melotts.py \
    --prompts data/sales_prompts_en.txt \
    --out-dir datasets/sales_synth_melotts \
    --language EN \
    --sr 16000 \
    --n 500 \
    --val-ratio 0.05 \
    --speed-min 0.9 --speed-max 1.1 \
    --snr-db-min 18 --snr-db-max 30 \
    --pad-start-ms-min 0 --pad-start-ms-max 400 \
    --pad-end-ms-min 0 --pad-end-ms-max 250 \
    --telephone

If you want a larger generated prompt set (e.g. 20k), run:

  /home/pragay/WWAI/.venv/bin/python stt_prompts_gen_retail_sales.py \
    --out data/sales_prompts_generated_en.txt \
    --n 20000 \
    --seed 0 \
    --min-words 4 --max-words 12 \
    --dedupe \
    --balance

…then use the generated file as the `--prompts` input:

  /home/pragay/WWAI/.venv/bin/python stt_dataset_gen_melotts.py \
    --prompts data/sales_prompts_generated_en.txt \
    --out-dir datasets/sales_synth_melotts \
    --language EN \
    --sr 16000 \
    --n 20000 \
    --val-ratio 0.05 \
    --speed-min 0.9 --speed-max 1.1 \
    --snr-db-min 18 --snr-db-max 30 \
    --pad-start-ms-min 0 --pad-start-ms-max 400 \
    --pad-end-ms-min 0 --pad-end-ms-max 250 \
    --telephone

- prompts: `data/sales_prompts_en.txt`
- output: `datasets/sales_synth_melotts/`
- artifacts:
  - `wavs/000000.wav`, ...
  - `train.jsonl`
  - `val.jsonl`

The JSONL format is one record per line:

- `audio`: path to wav file (absolute or relative)
- `text`: transcript
- extra metadata: `speaker_id`, `speed`, `sr`, etc.

## Strong recommendation (quality)

Whisper is usually most improved by training on *real* speech in the target conditions:

- real accents + speaking styles
- phone call / headset microphone quality
- background noise (office, car, street)
- disfluencies ("uh", "um", self-corrections)

Use TTS mainly to:

- teach domain vocabulary (plan names, product terms)
- generate lots of coverage quickly

…but always mix it with real audio when possible.

## Next step after you have train/val JSONL

Typical HuggingFace flow:

1. `datasets.load_dataset("json", data_files={...})`
2. `cast_column("audio", Audio(sampling_rate=16000))`
3. preprocess with `WhisperProcessor` (feature extractor + tokenizer)
4. train with `Seq2SeqTrainer`

Model choice depends on your latency/quality goals:

- `openai/whisper-small` is a common sweet spot
- `openai/whisper-tiny` is fast but lower accuracy

## Practical tips

- Keep transcripts *exact* (punctuation optional, but be consistent)
- Use 16 kHz mono WAV to match Whisper preprocessing defaults
- Hold out a validation set with realistic noise conditions
- Track WER (word error rate) before/after fine-tuning

If you want, tell me:

- which Whisper checkpoint you want to fine-tune (`tiny`/`base`/`small`/etc.)
- language(s) and accents
- target deployment (CPU, GPU, mobile)

…and I can add a training script in this repo that reads the generated JSONL and fine-tunes via Transformers.
