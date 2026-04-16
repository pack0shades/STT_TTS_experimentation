"""Generate a synthetic speech->text dataset for Whisper fine-tuning using VibeVoice.

This is the VibeVoice parallel to `stt_dataset_gen_melotts.py`: it turns prompt
text into WAVs plus train/val JSONL manifests.

Example:
    python stt_dataset_gen_vibevoice.py \
        --prompts data/sales_prompts_en.txt \
        --out-dir datasets/sales_synth_vibevoice \
        --speaker-name Emma \
        --n 200 \
        --val-ratio 0.05 \
        --sr 16000 \
        --snr-db-min 18 --snr-db-max 30 \
        --telephone
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

import numpy as np

import tts_common as common
from stt_dataset_gen_melotts import (
    _add_white_noise_snr,
    _apply_gain_db,
    _bandpass_fft,
    _pad_silence,
    _resample_linear,
    _simulate_telephony,
)
from tts_vibevoice import DEFAULT_MODEL_PATH, _load_vibevoice, synthesize_vibevoice


logger = logging.getLogger(__name__)


# -----------------------------
# Prompts loading
# -----------------------------


def _iter_prompts_txt(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            yield t


def _iter_prompts_jsonl(path: str) -> Iterable[str]:
    """Read JSONL with either {"text": ...} or {"prompt": ...}."""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text") or obj.get("prompt")
            if t and str(t).strip():
                yield str(t).strip()


def _iter_prompts(path: str) -> list[str]:
    ext = str(Path(path).suffix).lower()
    if ext == ".jsonl":
        return list(_iter_prompts_jsonl(path))
    return list(_iter_prompts_txt(path))


def main(argv: Optional[list[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=str, required=True, help="TXT (one utterance per line) or JSONL with {text: ...}")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--language", type=str, default="EN", help="Metadata only; VibeVoice realtime is primarily English")

    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--vibevoice-repo", type=str, default="")
    p.add_argument("--speaker-name", type=str, default="Emma")
    p.add_argument("--voice-preset", type=str, default="", help="Path to a VibeVoice streaming .pt voice preset")
    p.add_argument("--voices-dir", type=str, default="", help="Directory containing streaming_model .pt voice presets")
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--ddpm-steps", type=int, default=5)

    p.add_argument("--gpu", action="store_true", help="Force GPU")
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--sr", type=int, default=16000, help="Target sample rate for saved WAVs")
    p.add_argument("--n", type=int, default=0, help="Limit number of samples (0 = all prompts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.05)

    p.add_argument("--speed", type=float, default=1.0, help="Accepted for MeloTTS CLI compatibility; ignored")
    p.add_argument("--speed-min", type=float, default=0.0, help="Accepted for MeloTTS CLI compatibility; ignored")
    p.add_argument("--speed-max", type=float, default=0.0, help="Accepted for MeloTTS CLI compatibility; ignored")

    p.add_argument("--gain-db-min", type=float, default=0.0)
    p.add_argument("--gain-db-max", type=float, default=0.0)

    p.add_argument("--snr-db-min", type=float, default=0.0, help="If >0, add white noise with random SNR")
    p.add_argument("--snr-db-max", type=float, default=0.0)

    p.add_argument("--pad-start-ms-min", type=int, default=0)
    p.add_argument("--pad-start-ms-max", type=int, default=0)
    p.add_argument("--pad-end-ms-min", type=int, default=0)
    p.add_argument("--pad-end-ms-max", type=int, default=0)

    p.add_argument(
        "--bandpass-low-hz",
        type=float,
        default=0.0,
        help="If >0 along with bandpass-high-hz, apply a simple FFT bandpass before saving.",
    )
    p.add_argument("--bandpass-high-hz", type=float, default=0.0)
    p.add_argument(
        "--telephone",
        action="store_true",
        help="Simulate phone-call audio (300-3400 Hz band + 8k narrowband + mu-law artifacts)",
    )

    p.add_argument("--rel-paths", action="store_true", help="Store audio paths relative to out-dir in JSONL")

    args = p.parse_args(argv)

    if float(args.speed) != 1.0 or float(args.speed_min) > 0.0 or float(args.speed_max) > 0.0:
        logger.warning("VibeVoice realtime does not expose speed control; speed arguments will be ignored")

    if str(args.language).upper() != "EN":
        logger.warning("VibeVoice realtime is primarily intended for English; language=%s is metadata only", args.language)

    if int(args.seed):
        try:
            import torch

            torch.manual_seed(int(args.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed))
        except Exception:
            pass

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    if bool(args.cpu):
        use_gpu = False
    elif bool(args.gpu):
        use_gpu = True
    else:
        use_gpu = common.gpu_is_available()

    prompts = _iter_prompts(str(args.prompts))
    if not prompts:
        raise RuntimeError(f"No prompts found in {args.prompts}")

    if int(args.n) > 0:
        prompts = prompts[: int(args.n)]

    handle, init_s = _load_vibevoice(
        use_gpu=use_gpu,
        model_path=str(args.model_path),
        speaker_name=str(args.speaker_name),
        voice_preset=str(args.voice_preset),
        voices_dir=str(args.voices_dir),
        vibevoice_repo=str(args.vibevoice_repo),
        ddpm_steps=int(args.ddpm_steps),
    )
    common.log_json(logger, {"event": "vibevoice_dataset_ready", "init_s": round(float(init_s), 3)})

    gain_min = float(args.gain_db_min)
    gain_max = float(args.gain_db_max)
    use_gain_range = (gain_min != 0.0 or gain_max != 0.0) and gain_max >= gain_min

    snr_min = float(args.snr_db_min)
    snr_max = float(args.snr_db_max)
    use_snr_range = snr_min > 0 and snr_max > 0 and snr_max >= snr_min

    pad0_min = int(max(0, int(args.pad_start_ms_min)))
    pad0_max = int(max(0, int(args.pad_start_ms_max)))
    pad1_min = int(max(0, int(args.pad_end_ms_min)))
    pad1_max = int(max(0, int(args.pad_end_ms_max)))
    use_pad = (pad0_max > 0 or pad1_max > 0) and (pad0_max >= pad0_min) and (pad1_max >= pad1_min)

    bp_lo = float(args.bandpass_low_hz)
    bp_hi = float(args.bandpass_high_hz)
    use_bandpass = bp_lo > 0.0 and bp_hi > 0.0 and bp_hi > bp_lo

    idxs = list(range(len(prompts)))
    rng.shuffle(idxs)
    val_n = int(round(float(len(idxs)) * float(max(0.0, min(0.5, float(args.val_ratio))))))
    val_set = set(idxs[:val_n])

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    n_train = 0
    n_val = 0
    n_fail = 0

    t_all0 = perf_counter()

    with open(train_path, "w", encoding="utf-8") as f_train, open(val_path, "w", encoding="utf-8") as f_val:
        for i, text in enumerate(prompts):
            sample_id = f"{i:06d}"

            try:
                audio, model_sr, gen_s = synthesize_vibevoice(
                    handle=handle,
                    text=text,
                    cfg_scale=float(args.cfg_scale),
                )

                if use_gain_range:
                    audio = _apply_gain_db(audio, rng.uniform(gain_min, gain_max))
                if use_snr_range:
                    audio = _add_white_noise_snr(audio, rng.uniform(snr_min, snr_max), rng)

                if bool(args.telephone):
                    audio = _simulate_telephony(audio, sr=int(model_sr))

                if use_bandpass:
                    audio = _bandpass_fft(audio, int(model_sr), low_hz=bp_lo, high_hz=bp_hi)

                if use_pad:
                    pad0 = rng.randrange(pad0_min, pad0_max + 1) if pad0_max >= pad0_min else 0
                    pad1 = rng.randrange(pad1_min, pad1_max + 1) if pad1_max >= pad1_min else 0
                    audio = _pad_silence(audio, int(model_sr), pad_start_ms=int(pad0), pad_end_ms=int(pad1))

                audio = _resample_linear(audio, int(model_sr), int(args.sr))
                wav_path = wav_dir / f"{sample_id}.wav"
                common.save_wav_pcm16(str(wav_path), audio, int(args.sr))

                duration_s = float(audio.size) / float(int(args.sr)) if audio.size else 0.0

                rec = {
                    "id": sample_id,
                    "audio": str(wav_path if not args.rel_paths else wav_path.relative_to(out_dir)),
                    "text": text,
                    "language": str(args.language),
                    "model": "vibevoice-realtime-0.5b",
                    "model_path": str(args.model_path),
                    "speaker_name": str(args.speaker_name),
                    "cfg_scale": float(args.cfg_scale),
                    "ddpm_steps": int(args.ddpm_steps),
                    "sr": int(args.sr),
                    "duration_s": round(duration_s, 4),
                    "gen_s": round(float(gen_s), 4),
                }

                out_f = f_val if i in val_set else f_train
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if i in val_set:
                    n_val += 1
                else:
                    n_train += 1

                if (i + 1) % 25 == 0:
                    common.log_json(
                        logger,
                        {
                            "event": "progress",
                            "done": i + 1,
                            "total": len(prompts),
                            "train": n_train,
                            "val": n_val,
                            "fail": n_fail,
                        },
                    )

            except Exception as e:
                n_fail += 1
                logger.exception("Failed to synthesize sample %s: %s", sample_id, e)

    total_s = perf_counter() - t_all0
    common.log_json(
        logger,
        {
            "event": "done",
            "out_dir": str(out_dir),
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "n_total": len(prompts),
            "n_train": n_train,
            "n_val": n_val,
            "n_fail": n_fail,
            "wall_s": round(float(total_s), 3),
        },
    )

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
