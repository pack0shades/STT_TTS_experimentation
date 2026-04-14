"""Generate a synthetic speech->text dataset for Whisper fine-tuning using MeloTTS.

Why this exists
--------------
You said MeloTTS is already working in this repo and you want to generate voice data.
This script turns a list of *texts* (sales/customer-like utterances) into:
- WAV files (default 16 kHz mono PCM16)
- JSONL manifests (train.jsonl / val.jsonl) with {"audio": "/abs/or/rel/path.wav", "text": "..."}

Notes / caveats
---------------
Training Whisper on pure TTS audio can help bootstrap domain vocabulary, but it will
NOT fully match real customer speech (accents, mics, noise, disfluencies). You’ll
get best results by mixing this synthetic set with real recordings + augmentation.

Example
-------
    python stt_dataset_gen_melotts.py \
        --prompts data/sales_prompts_en.txt \
        --out-dir datasets/sales_synth_melotts \
        --language EN \
        --n 200 \
        --val-ratio 0.05 \
        --sr 16000 \
        --speed-min 0.9 --speed-max 1.1 \
            --snr-db-min 18 --snr-db-max 30 \
            --pad-start-ms-min 0 --pad-start-ms-max 400 \
            --pad-end-ms-min 0 --pad-end-ms-max 250 \
            --telephone

Then for HuggingFace:
    load_dataset("json", data_files={"train": ".../train.jsonl", "validation": ".../val.jsonl"})
    ... cast_column("audio", Audio(sampling_rate=16000))

"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

import numpy as np

import tts_common as common

logger = logging.getLogger(__name__)


# -----------------------------
# Audio helpers
# -----------------------------


def _pad_silence(
    audio: np.ndarray,
    sr: int,
    *,
    pad_start_ms: int = 0,
    pad_end_ms: int = 0,
) -> np.ndarray:
    audio = common.to_mono_float32(audio)
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f"Invalid sr={sr}")

    n0 = int(round(float(pad_start_ms) * float(sr) / 1000.0))
    n1 = int(round(float(pad_end_ms) * float(sr) / 1000.0))
    if n0 <= 0 and n1 <= 0:
        return audio

    parts = []
    if n0 > 0:
        parts.append(np.zeros((n0,), dtype=np.float32))
    parts.append(audio)
    if n1 > 0:
        parts.append(np.zeros((n1,), dtype=np.float32))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _bandpass_fft(audio: np.ndarray, sr: int, *, low_hz: float, high_hz: float) -> np.ndarray:
    """Very simple bandpass via FFT masking.

    This is not a perfect filter (it can ring a bit), but it's deterministic,
    dependency-free, and good enough to mimic narrowband/phone-like audio.
    """

    audio = common.to_mono_float32(audio)
    sr = int(sr)
    if audio.size == 0:
        return audio

    low_hz = float(low_hz)
    high_hz = float(high_hz)
    if not (0.0 < low_hz < high_hz < (0.5 * sr)):
        raise ValueError(f"Invalid bandpass low_hz={low_hz} high_hz={high_hz} for sr={sr}")

    n = int(audio.size)
    spec = np.fft.rfft(audio.astype(np.float32, copy=False))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec = spec * mask
    out = np.fft.irfft(spec, n=n).astype(np.float32, copy=False)
    return np.clip(out, -1.0, 1.0)


def _mu_law_roundtrip(audio: np.ndarray, *, mu: int = 255) -> np.ndarray:
    """Apply μ-law companding + 8-bit quantization + expansion.

    This approximates telephone/codec artifacts without external deps.
    """

    audio = common.to_mono_float32(audio)
    if audio.size == 0:
        return audio

    mu = int(mu)
    mu_f = float(mu)
    x = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

    y = np.sign(x) * (np.log1p(mu_f * np.abs(x)) / np.log1p(mu_f))
    # quantize to 8-bit (256 levels) in [-1, 1]
    yq = np.round((y + 1.0) * 0.5 * 255.0) / 255.0 * 2.0 - 1.0
    out = np.sign(yq) * (1.0 / mu_f) * (np.expm1(np.abs(yq) * np.log1p(mu_f)))
    return np.clip(out.astype(np.float32, copy=False), -1.0, 1.0)


def _simulate_telephony(audio: np.ndarray, *, sr: int) -> np.ndarray:
    """Make audio sound more like a phone call (narrowband + μ-law-ish artifacts)."""

    sr = int(sr)
    audio = common.to_mono_float32(audio)
    if audio.size == 0:
        return audio

    # Phone: ~300-3400 Hz band + narrowband resampling + μ-law artifacts.
    a = _bandpass_fft(audio, sr, low_hz=300.0, high_hz=min(3400.0, 0.5 * sr - 1.0))
    # If we're at 16k, simulate 8k PSTN then back.
    if sr != 8000:
        a = _resample_linear(a, sr, 8000)
        a = _resample_linear(a, 8000, sr)
    a = _mu_law_roundtrip(a, mu=255)
    return np.clip(a, -1.0, 1.0).astype(np.float32, copy=False)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample 1D float32 audio using linear interpolation (dependency-free).

    This is good enough for dataset generation and avoids pulling in scipy/librosa.
    """

    audio = common.to_mono_float32(np.asarray(audio, dtype=np.float32))

    src_sr = int(src_sr)
    dst_sr = int(dst_sr)
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError(f"Invalid sample rates src_sr={src_sr} dst_sr={dst_sr}")

    if audio.size == 0 or src_sr == dst_sr:
        return audio

    src_n = int(audio.size)
    dst_n = int(round(float(src_n) * float(dst_sr) / float(src_sr)))
    dst_n = max(1, dst_n)

    # np.interp requires float64 x-coords; keep it simple.
    x_old = np.linspace(0.0, 1.0, num=src_n, endpoint=True, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=dst_n, endpoint=True, dtype=np.float64)

    y = np.interp(x_new, x_old, audio.astype(np.float64, copy=False)).astype(np.float32, copy=False)
    return np.clip(y, -1.0, 1.0)


def _apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    g = float(10.0 ** (float(gain_db) / 20.0))
    return np.clip(audio * g, -1.0, 1.0).astype(np.float32, copy=False)


def _add_white_noise_snr(audio: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    """Add white noise to achieve an approximate SNR in dB."""

    audio = common.to_mono_float32(audio)
    if audio.size == 0:
        return audio

    # Signal power
    sig = float(np.mean(np.square(audio, dtype=np.float64)))
    if sig <= 1e-12:
        return audio

    snr_db = float(snr_db)
    noise_power = sig / (10.0 ** (snr_db / 10.0))
    noise = rng.normalvariate(0.0, 1.0)  # warm-up
    _ = noise

    n = rng
    # Generate in numpy with a deterministic seed derived from rng state
    seed = n.randrange(0, 2**32 - 1)
    nrng = np.random.default_rng(seed)
    w = nrng.standard_normal(audio.shape, dtype=np.float32)
    w_power = float(np.mean(np.square(w, dtype=np.float64)))
    if w_power <= 0:
        return audio

    w = w * np.float32(np.sqrt(noise_power / w_power))
    return np.clip(audio + w, -1.0, 1.0).astype(np.float32, copy=False)


# -----------------------------
# Prompts loading
# -----------------------------

def _iter_prompts_txt(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            if t.startswith("#"):
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


# -----------------------------
# MeloTTS wrapper
# -----------------------------

@dataclass(frozen=True)
class MeloHandle:
    tts: object
    sr: int
    spk2id: dict[str, int]


def _load_melotts(*, use_gpu: bool, language: str) -> MeloHandle:
    try:
        from melo.api import TTS  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "MeloTTS not importable. Install the MeloTTS repo/package so `from melo.api import TTS` works."
        ) from e

    device = "cuda" if use_gpu else "cpu"
    t0 = perf_counter()
    tts = TTS(language=str(language), device=device)
    init_s = perf_counter() - t0

    spk2id = {}
    try:
        raw = getattr(getattr(getattr(tts, "hps"), "data"), "spk2id", None)
        if isinstance(raw, dict):
            spk2id = {str(k): int(v) for k, v in raw.items()}
    except Exception:
        spk2id = {}

    try:
        sr = int(getattr(getattr(tts, "hps"), "data").sampling_rate)
    except Exception as e:
        raise RuntimeError("Could not read MeloTTS sampling_rate from tts.hps.data") from e

    common.log_json(logger, {"event": "melotts_loaded", "device": device, "sr": sr, "init_s": round(init_s, 3)})
    return MeloHandle(tts=tts, sr=sr, spk2id=spk2id)


def _pick_speaker_id(
    *,
    spk2id: dict[str, int],
    speaker: str,
    speaker_id: int,
    rng: random.Random,
    random_speaker: bool,
) -> int:
    if int(speaker_id) >= 0:
        return int(speaker_id)

    if speaker and speaker in spk2id:
        return int(spk2id[speaker])

    if spk2id:
        keys = sorted(spk2id.keys())
        if random_speaker:
            k = keys[int(rng.randrange(0, len(keys)))]
            return int(spk2id[k])
        # stable default
        return int(spk2id[keys[0]])

    return 0


def _synthesize_one(
    *,
    handle: MeloHandle,
    text: str,
    spk_id: int,
    speed: float,
    quiet: bool,
) -> np.ndarray:
    audio = getattr(handle.tts, "tts_to_file")(
        text,
        int(spk_id),
        output_path=None,
        speed=float(speed),
        quiet=bool(quiet),
    )
    return common.to_mono_float32(np.asarray(audio, dtype=np.float32))


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[list[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=str, required=True, help="TXT (one utterance per line) or JSONL with {text: ...}")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--language", type=str, default="EN")

    p.add_argument("--gpu", action="store_true", help="Force GPU")
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--speaker", type=str, default="", help="Speaker name (if model provides spk2id)")
    p.add_argument("--speaker-id", type=int, default=-1)
    p.add_argument("--random-speaker", action="store_true", help="Randomize speaker if spk2id available")

    p.add_argument("--sr", type=int, default=16000, help="Target sample rate for saved WAVs")
    p.add_argument("--n", type=int, default=0, help="Limit number of samples (0 = all prompts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.05)

    p.add_argument("--speed", type=float, default=1.0, help="Fixed speed (overrides speed-min/max if provided)")
    p.add_argument("--speed-min", type=float, default=0.0)
    p.add_argument("--speed-max", type=float, default=0.0)

    p.add_argument("--gain-db-min", type=float, default=0.0)
    p.add_argument("--gain-db-max", type=float, default=0.0)

    p.add_argument("--snr-db-min", type=float, default=0.0, help="If >0, add white noise with random SNR")
    p.add_argument("--snr-db-max", type=float, default=0.0)

    p.add_argument(
        "--pad-start-ms-min",
        type=int,
        default=0,
        help="Random leading silence (ms). Useful to mimic natural call timing.",
    )
    p.add_argument("--pad-start-ms-max", type=int, default=0)
    p.add_argument("--pad-end-ms-min", type=int, default=0, help="Random trailing silence (ms).")
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
        help="Simulate phone-call audio (300-3400 Hz band + 8k narrowband + μ-law artifacts)",
    )

    p.add_argument("--rel-paths", action="store_true", help="Store audio paths relative to out-dir in JSONL")

    args = p.parse_args(argv)

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

    handle = _load_melotts(use_gpu=use_gpu, language=str(args.language))

    # Decide speed/gain ranges
    speed_min = float(args.speed_min) if float(args.speed_min) > 0 else 0.0
    speed_max = float(args.speed_max) if float(args.speed_max) > 0 else 0.0
    use_speed_range = speed_min > 0 and speed_max > 0 and speed_max >= speed_min

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
    use_bandpass = (bp_lo > 0.0 and bp_hi > 0.0 and bp_hi > bp_lo)

    # Precompute split indices (deterministic shuffle)
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

            spk_id = _pick_speaker_id(
                spk2id=handle.spk2id,
                speaker=str(args.speaker),
                speaker_id=int(args.speaker_id),
                rng=rng,
                random_speaker=bool(args.random_speaker),
            )

            if use_speed_range:
                speed = rng.uniform(speed_min, speed_max)
            else:
                speed = float(args.speed)

            try:
                t0 = perf_counter()
                audio = _synthesize_one(handle=handle, text=text, spk_id=spk_id, speed=speed, quiet=True)
                gen_s = perf_counter() - t0

                if use_gain_range:
                    audio = _apply_gain_db(audio, rng.uniform(gain_min, gain_max))
                if use_snr_range:
                    audio = _add_white_noise_snr(audio, rng.uniform(snr_min, snr_max), rng)

                if bool(args.telephone):
                    audio = _simulate_telephony(audio, sr=handle.sr)

                if use_bandpass:
                    audio = _bandpass_fft(audio, handle.sr, low_hz=bp_lo, high_hz=bp_hi)

                if use_pad:
                    pad0 = rng.randrange(pad0_min, pad0_max + 1) if pad0_max >= pad0_min else 0
                    pad1 = rng.randrange(pad1_min, pad1_max + 1) if pad1_max >= pad1_min else 0
                    audio = _pad_silence(audio, handle.sr, pad_start_ms=int(pad0), pad_end_ms=int(pad1))

                # Resample and save
                audio = _resample_linear(audio, handle.sr, int(args.sr))
                wav_path = wav_dir / f"{sample_id}.wav"
                common.save_wav_pcm16(str(wav_path), audio, int(args.sr))

                duration_s = float(audio.size) / float(int(args.sr)) if audio.size else 0.0

                rec = {
                    "id": sample_id,
                    "audio": str(wav_path if not args.rel_paths else wav_path.relative_to(out_dir)),
                    "text": text,
                    "language": str(args.language),
                    "speaker_id": int(spk_id),
                    "speed": round(float(speed), 4),
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
